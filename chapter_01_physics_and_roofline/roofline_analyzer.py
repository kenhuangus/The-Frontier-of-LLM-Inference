"""
Roofline Model and Arithmetic Intensity Analyzer
Calculates theoretical hardware ceilings, operational intensity ridge points,
and Model Flops Utilization (MFU) across modern accelerator architectures.
"""
from typing import Dict

def analyze_roofline(
    hardware_name: str,
    peak_tflops: float,
    memory_bandwidth_tbps: float,
    operational_intensity: float
) -> Dict[str, float]:
    """
    Computes attainable performance and hardware balance metrics under the Roofline model.
    """
    ridge_point = peak_tflops / memory_bandwidth_tbps
    bandwidth_bound_tflops = operational_intensity * memory_bandwidth_tbps
    attainable_tflops = min(peak_tflops, bandwidth_bound_tflops)
    is_memory_bound = operational_intensity < ridge_point
    return {
        "Hardware": hardware_name,
        "Ridge_Point_FLOP_per_Byte": round(ridge_point, 2),
        "Operational_Intensity": round(operational_intensity, 2),
        "Attainable_TFLOPS": round(attainable_tflops, 2),
        "Is_Memory_Bound": is_memory_bound
    }

def classify_inference_regime(
    hardware_name: str,
    peak_tflops: float,
    attainable_tflops: float,
    is_memory_bound: bool
) -> Dict[str, any]:
    """Classifies execution regime into Memory-Bound (GEMV) vs Compute-Bound (GEMM)."""
    mfu_percent = (attainable_tflops / peak_tflops) * 100.0
    regime = "Memory-Bandwidth Bound (GEMV decode)" if is_memory_bound else "Compute Bound (GEMM prefill)"
    return {
        "Hardware": hardware_name,
        "Attainable_TFLOPS": round(attainable_tflops, 2),
        "MFU_Percent": round(mfu_percent, 2),
        "Execution_Regime": regime
    }

if __name__ == "__main__":
    print("=" * 80)
    print("  CHAPTER 1: ROOFLINE INFERENCE REGIME BENCHMARK (H100 vs B200)")
    print("=" * 80)
    
    benchmarks = [
        ("NVIDIA H100 SXM (Decode B=1)", 1979.0, 3.35, 1.5),
        ("NVIDIA H100 SXM (Prefill 4k)", 1979.0, 3.35, 450.0),
        ("NVIDIA B200 NVL (Decode B=1)", 4500.0, 8.0, 1.5),
        ("NVIDIA B200 NVL (Prefill 4k)", 4500.0, 8.0, 450.0),
        ("Google TPU v5p (Decode B=1)", 459.0, 4.8, 1.5),
    ]
    
    for name, tflops, bw, intensity in benchmarks:
        metrics = analyze_roofline(name, tflops, bw, intensity)
        regime = classify_inference_regime(name, tflops, metrics["Attainable_TFLOPS"], metrics["Is_Memory_Bound"])
        print(f"\n[{name}]")
        print(f"  Ridge Point: {metrics['Ridge_Point_FLOP_per_Byte']} FLOP/B | Intensity: {metrics['Operational_Intensity']} FLOP/B")
        print(f"  Attainable Compute: {regime['Attainable_TFLOPS']} TFLOPS | MFU: {regime['MFU_Percent']}%")
        print(f"  Regime: {regime['Execution_Regime']}")
    print("=" * 80)
