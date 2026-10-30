"""
Industrial GPU Cluster Sizing & SRE Capacity Model
Calculates required GPU node counts, hardware capital expenditure, and TCO for production SLOs.
"""
import math
from typing import Dict, Any, Optional

# Hardware architectural specifications for frontier inference GPUs
GPU_SPECS: Dict[str, Dict[str, float]] = {
    "H100": {
        "tflops": 1979.0,        # FP8 dense Tensor Core TFLOPS
        "vram_gb": 80,           # HBM3 capacity (GB)
        "cost_gpu": 30000,       # Estimated acquisition CapEx per GPU ($)
        "decode_tok_s": 150.0,    # Baseline decode token throughput per GPU
    },
    "B200": {
        "tflops": 4500.0,        # FP4/FP8 dense Tensor Core TFLOPS
        "vram_gb": 192,          # HBM3e capacity (GB)
        "cost_gpu": 40000,       # Estimated acquisition CapEx per GPU ($)
        "decode_tok_s": 350.0,    # ~2.33x higher memory bandwidth (8 TB/s vs 3.35 TB/s)
    },
}

def calculate_cluster_tco(
    target_qps: float,
    avg_prompt_tokens: int,
    avg_gen_tokens: int,
    gpu_type: str = "H100",
    sla_target: float = 0.0,
    amortization_years: float = 3.0,
    annual_opex_ratio: float = 0.20,
    custom_specs: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Calculates GPU cluster capacity, node requirements, CapEx, and TCO.

    Args:
        target_qps: Average query arrival rate in queries per second.
        avg_prompt_tokens: Mean input prompt token count per query.
        avg_gen_tokens: Mean output generated token count per query.
        gpu_type: Target GPU accelerator type ("H100" or "B200").
        sla_target: Production SLA percentile target (e.g., 0.95, 0.99 for P99, 0.999).
                    Values in [0.0, 1.0). Higher targets increase capacity headroom
                    to absorb queueing bursts and prevent tail latency spikes.
        amortization_years: Hardware depreciation window in years (typically 3.0).
        annual_opex_ratio: Operational expenditure ratio as a fraction of CapEx per year
                           (covering datacenter lease, cooling, power, and networking).
        custom_specs: Optional dictionary overriding default GPU hardware specifications.

    Returns:
        Dict containing cluster sizing metrics, required nodes/GPUs, CapEx, TCO,
        and amortized cost per million tokens.
    """
    if target_qps < 0:
        raise ValueError(f"target_qps must be non-negative, got {target_qps}")
    if avg_prompt_tokens < 0 or avg_gen_tokens < 0:
        raise ValueError(
            f"Token counts must be non-negative, got prompt={avg_prompt_tokens}, gen={avg_gen_tokens}"
        )
    if sla_target < 0.0 or sla_target >= 1.0:
        raise ValueError(f"sla_target must be in range [0.0, 1.0), got {sla_target}")
    if amortization_years <= 0:
        raise ValueError(f"amortization_years must be positive, got {amortization_years}")
    if annual_opex_ratio < 0:
        raise ValueError(f"annual_opex_ratio must be non-negative, got {annual_opex_ratio}")

    specs = custom_specs if custom_specs is not None else GPU_SPECS
    if gpu_type not in specs:
        raise ValueError(f"Unsupported gpu_type '{gpu_type}'. Supported: {list(specs.keys())}")
    spec = specs[gpu_type]

    # Active token bandwidth demand:
    # In disaggregated or coupled serving, prefill tokens require ~5% of steady-state
    # decode compute cycles due to compute-bound parallelism.
    total_tokens_per_sec = target_qps * (avg_prompt_tokens * 0.05 + avg_gen_tokens)

    # Production SRE SLA capacity headroom:
    # Operating a cluster near 100% saturation causes queueing delays to diverge.
    # Higher SLA targets (P95, P99, P99.9) lower the target utilization ceiling.
    if sla_target <= 0.0:
        headroom = 1.0
    else:
        # Utilization ceiling formula for queueing stability under bursty traffic:
        utilization_ceiling = max(0.40, 1.0 - (sla_target - 0.5) * 0.8) if sla_target > 0.5 else 1.0
        headroom = 1.0 / utilization_ceiling

    effective_tokens_per_sec = total_tokens_per_sec * headroom
    decode_tok_s = spec.get("decode_tok_s", 150.0)

    # Handle zero QPS / zero throughput edge cases safely
    if effective_tokens_per_sec > 0 and decode_tok_s > 0:
        gpus_for_throughput = math.ceil(effective_tokens_per_sec / decode_tok_s)
        # Standard industrial server nodes host 8 SXM5/SXM6 GPUs
        nodes = math.ceil(gpus_for_throughput / 8.0)
        total_gpus = nodes * 8
    else:
        gpus_for_throughput = 0
        nodes = 0
        total_gpus = 0

    total_capex = total_gpus * spec["cost_gpu"]
    total_tco = total_capex * (1.0 + annual_opex_ratio * amortization_years)

    # Amortized token economics:
    lifetime_seconds = amortization_years * 365.25 * 86400.0
    total_lifetime_tokens = total_tokens_per_sec * lifetime_seconds

    if total_lifetime_tokens > 0:
        cost_per_million = (total_tco / total_lifetime_tokens) * 1_000_000.0
        capex_per_million = (total_capex / total_lifetime_tokens) * 1_000_000.0
    else:
        cost_per_million = 0.0
        capex_per_million = 0.0

    return {
        "Target_QPS": target_qps,
        "GPU_Type": gpu_type,
        "Required_Nodes": nodes,
        "Total_GPUs": total_gpus,
        "Hardware_CapEx_USD": total_capex,
        "Total_TCO_USD": round(total_tco, 2),
        "Total_Tokens_Per_Sec": total_tokens_per_sec,
        "Cost_Per_Million_Tokens_USD": round(cost_per_million, 4),
        "CapEx_Per_Million_Tokens_USD": round(capex_per_million, 4),
        "SLA_Target": sla_target,
        "Headroom_Multiplier": round(headroom, 4),
    }

if __name__ == "__main__":
    print("=" * 80)
    print("  CHAPTER 10: INDUSTRIAL GPU FLEET SIZING & TCO")
    print("=" * 80)
    for qps in [50.0, 200.0]:
        tco_h100 = calculate_cluster_tco(qps, 2000, 200, "H100")
        tco_b200 = calculate_cluster_tco(qps, 2000, 200, "B200")
        print(f"Target Workload: {qps} QPS (2k in / 200 out)")
        print(
            f"  H100 Fleet: {tco_h100['Total_GPUs']} GPUs ({tco_h100['Required_Nodes']} nodes) -> "
            f"${tco_h100['Hardware_CapEx_USD']:,} (${tco_h100['Cost_Per_Million_Tokens_USD']:.2f}/M tok)"
        )
        print(
            f"  B200 Fleet: {tco_b200['Total_GPUs']} GPUs ({tco_b200['Required_Nodes']} nodes) -> "
            f"${tco_b200['Hardware_CapEx_USD']:,} (${tco_b200['Cost_Per_Million_Tokens_USD']:.2f}/M tok)"
        )
    print("=" * 80)
