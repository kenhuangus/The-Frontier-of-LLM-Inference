"""
Industrial GPU Cluster Sizing & SRE Capacity Model
Calculates required GPU node counts and hardware capital expenditure for production SLOs.
"""
import math
from typing import Dict

def calculate_cluster_tco(
    target_qps: float,
    avg_prompt_tokens: int,
    avg_gen_tokens: int,
    gpu_type: str = "H100"
) -> Dict[str, any]:
    specs = {
        "H100": {"tflops": 1979.0, "vram_gb": 80, "cost_gpu": 30000},
        "B200": {"tflops": 4500.0, "vram_gb": 192, "cost_gpu": 40000}
    }
    spec = specs[gpu_type]
    
    # Compute active token bandwidth
    total_tokens_per_sec = target_qps * (avg_prompt_tokens * 0.05 + avg_gen_tokens)
    gpus_for_throughput = math.ceil(total_tokens_per_sec / 150.0) # ~150 tok/s per GPU decode
    # Round to 8-GPU nodes
    nodes = math.ceil(gpus_for_throughput / 8.0)
    total_gpus = nodes * 8
    total_capex = total_gpus * spec["cost_gpu"]
    
    return {
        "Target_QPS": target_qps,
        "GPU_Type": gpu_type,
        "Required_Nodes": nodes,
        "Total_GPUs": total_gpus,
        "Hardware_CapEx_USD": total_capex
    }

if __name__ == "__main__":
    print("=" * 80)
    print("  CHAPTER 10: INDUSTRIAL GPU FLEET SIZING & TCO")
    print("=" * 80)
    for qps in [50.0, 200.0]:
        tco_h100 = calculate_cluster_tco(qps, 2000, 200, "H100")
        tco_b200 = calculate_cluster_tco(qps, 2000, 200, "B200")
        print(f"Target Workload: {qps} QPS (2k in / 200 out)")
        print(f"  H100 Fleet: {tco_h100['Total_GPUs']} GPUs ({tco_h100['Required_Nodes']} nodes) -> ${tco_h100['Hardware_CapEx_USD']:,}")
        print(f"  B200 Fleet: {tco_b200['Total_GPUs']} GPUs ({tco_b200['Required_Nodes']} nodes) -> ${tco_b200['Hardware_CapEx_USD']:,}")
    print("=" * 80)
