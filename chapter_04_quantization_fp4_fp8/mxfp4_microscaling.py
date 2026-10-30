"""
NVIDIA Blackwell NV-FP4 Microscaling Simulator (E2M1 Format)
Simulates two-level microscaling: vector block scaling and E2M1 FP4 numeric quantizer.
Supports OCP Microscaling (MX) specifications with block size 32 alignment,
scale factor clamping, SNR computation, and extreme value stability.
"""
import math
from typing import List, Tuple

# E2M1 representable positive magnitudes: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

# OCP Microscaling Formats (MX) 8-bit scale factor (E8M0) dynamic range limits
# Minimum non-zero scale: 2^-127; Maximum scale: 2^127
MIN_SCALE = 2.0 ** -127
MAX_SCALE = 2.0 ** 127

def quantize_e2m1(val: float) -> float:
    """
    Quantizes a single scalar value to the nearest E2M1 FP4 numeric representation.
    Handles NaN (mapped to 0.0) and +/- Inf (clamped to +/- 6.0).
    """
    if math.isnan(val):
        return 0.0
    if math.isinf(val):
        return -6.0 if val < 0.0 else 6.0

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

def quantize_mxfp4_block(
    tensor: List[float],
    block_size: int = 32,
    clamp_scale: bool = True
) -> Tuple[List[float], float]:
    """
    Quantizes a vector block using NV-FP4 / OCP MXFP4 microscaling.
    Maps the block maximum absolute value to 6.0 (E2M1 max representable magnitude).
    Handles empty blocks, all-zero blocks, and extreme scale factors.
    """
    if not tensor:
        return [], 0.0

    finite_vals = [abs(x) for x in tensor if not (math.isnan(x) or math.isinf(x))]
    has_inf = any(math.isinf(x) for x in tensor)

    if has_inf:
        max_val = 6.0 * MAX_SCALE
    elif finite_vals:
        max_val = max(finite_vals)
    else:
        max_val = 0.0

    # Handle all-zero blocks: scale factor is zero and all dequantized values are 0.0
    if max_val == 0.0:
        return [0.0] * len(tensor), 0.0

    # Scale factor maps max value to 6.0 (E2M1 max representable value)
    block_scale = max_val / 6.0

    if clamp_scale:
        block_scale = max(MIN_SCALE, min(MAX_SCALE, block_scale))

    quantized = []
    for x in tensor:
        scaled_x = x / block_scale
        q = quantize_e2m1(scaled_x)
        dequant = q * block_scale
        quantized.append(dequant)
    return quantized, block_scale

def quantize_mxfp4_tensor(
    tensor: List[float],
    block_size: int = 32,
    clamp_scale: bool = True
) -> Tuple[List[float], List[float]]:
    """
    Quantizes an arbitrary-length 1D tensor by partitioning it into blocks of block_size.
    Pads unaligned inputs with zero to maintain block-size alignment, then unpads.
    Returns:
        (dequantized_tensor, list_of_block_scales)
    """
    if not tensor:
        return [], []

    orig_len = len(tensor)
    pad_len = (block_size - (orig_len % block_size)) % block_size
    padded_tensor = tensor + [0.0] * pad_len

    dequantized_full: List[float] = []
    scales: List[float] = []

    for i in range(0, len(padded_tensor), block_size):
        chunk = padded_tensor[i : i + block_size]
        q_chunk, scale = quantize_mxfp4_block(chunk, block_size=block_size, clamp_scale=clamp_scale)
        dequantized_full.extend(q_chunk)
        scales.append(scale)

    return dequantized_full[:orig_len], scales

def calculate_snr(original: List[float], quantized: List[float]) -> float:
    """
    Calculates the Signal-to-Quantization-Noise Ratio (SQNR / SNR) in decibels (dB).
    Formula: SNR_dB = 10 * log10(P_signal / P_noise)
    Returns:
        0.0 if signal power is zero or inputs are empty
        float('inf') if noise power is zero (lossless)
    """
    if len(original) != len(quantized):
        raise ValueError(f"Length mismatch: original ({len(original)}) vs quantized ({len(quantized)})")
    if not original:
        return 0.0

    signal_power = sum(x * x for x in original)
    noise_power = sum((orig - deq) ** 2 for orig, deq in zip(original, quantized))

    if signal_power == 0.0:
        return 0.0
    if noise_power == 0.0:
        return float('inf')

    return 10.0 * math.log10(signal_power / noise_power)

if __name__ == "__main__":
    print("=" * 80)
    print("  CHAPTER 4: BLACKWELL NV-FP4 (E2M1) MICROSCALING QUANTIZER")
    print("=" * 80)
    sample_weights = [-2.8, -1.4, 0.05, 0.45, 1.12, 2.34, 3.98, -0.72]
    dequantized, scale = quantize_mxfp4_block(sample_weights, block_size=16)
    
    print(f"Block Scale Factor: {scale:.4f}")
    print(f"{'Original FP16':<15} | {'Dequantized FP4':<15} | {'Abs Error':<10}")
    print("-" * 45)
    mse = 0.0
    for orig, deq in zip(sample_weights, dequantized):
        err = abs(orig - deq)
        mse += err**2
        print(f"{orig:>15.4f} | {deq:>15.4f} | {err:>10.4f}")
    print("-" * 45)
    rmse = math.sqrt(mse / len(sample_weights))
    snr_db = calculate_snr(sample_weights, dequantized)
    print(f"Quantization RMSE: {rmse:.4f}")
    print(f"Quantization SNR:  {snr_db:.2f} dB")
    print("=" * 80)
