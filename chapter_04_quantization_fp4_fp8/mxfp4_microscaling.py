"""
NVIDIA Blackwell NV-FP4 Microscaling Simulator (E2M1 Format)
Simulates two-level microscaling: vector block scaling and E2M1 FP4 numeric quantizer.
"""
import math
from typing import List, Tuple

# E2M1 representable positive magnitudes: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

def quantize_e2m1(val: float) -> float:
    sign = -1.0 if val < 0.0 else 1.0
    abs_v = abs(val)
    best_match = E2M1_VALUES[0]
    min_dist = float('inf')
    for candidate in E2M1_VALUES:
        dist = abs(abs_v - candidate)
        if dist < min_dist:
            min_dist = dist
            best_match = candidate
    return sign * best_match

def quantize_mxfp4_block(tensor: List[float], block_size: int = 16) -> Tuple[List[float], float]:
    """Quantizes a block using NV-FP4 microscaling."""
    max_val = max(abs(x) for x in tensor) or 1e-6
    # Scale factor maps max value to 6.0 (E2M1 max representable value)
    block_scale = max_val / 6.0
    
    quantized = []
    for x in tensor:
        scaled_x = x / block_scale
        q = quantize_e2m1(scaled_x)
        dequant = q * block_scale
        quantized.append(dequant)
    return quantized, block_scale

if __name__ == "__main__":
    print("=" * 80)
    print("  CHAPTER 4: BLACKWELL NV-FP4 (E2M1) MICROSCALING QUANTIZER")
    print("=" * 80)
    sample_weights = [-2.8, -1.4, 0.05, 0.45, 1.12, 2.34, 3.98, -0.72]
    dequantized, scale = quantize_mxfp4_block(sample_weights)
    
    print(f"Block Scale Factor: {scale:.4f}")
    print(f"{'Original FP16':<15} | {'Dequantized FP4':<15} | {'Abs Error':<10}")
    print("-" * 45)
    mse = 0.0
    for orig, deq in zip(sample_weights, dequantized):
        err = abs(orig - deq)
        mse += err**2
        print(f"{orig:>15.4f} | {deq:>15.4f} | {err:>10.4f}")
    print("-" * 45)
    print(f"Quantization SNR / RMSE: {math.sqrt(mse / len(sample_weights)):.4f}")
    print("=" * 80)
