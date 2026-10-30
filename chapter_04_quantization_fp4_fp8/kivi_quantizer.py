"""
KIVI 2-Bit Asymmetric KV Cache Quantization
Performs per-channel Key quantization and per-token Value quantization.
Reduces KV cache footprint by up to 4x relative to FP8 and 8x relative to FP16.

Mathematical formulation:
- 2-bit asymmetric quantization maps [min_v, max_v] onto 4 integer levels: {0, 1, 2, 3}.
- scale = (max_v - min_v) / 3.0
- zero_point = min_v
- q = clamp(round((v - zero_point) / scale), 0, 3)
- v_hat = q * scale + zero_point
- Residual error bound: |v - v_hat| <= scale / 2.0 = (max_v - min_v) / 6.0
"""
import math
from typing import List, Tuple

def quantize_2bit_asymmetric(values: List[float]) -> Tuple[List[int], float, float]:
    """
    Quantizes a 1D list of floating point values to 2-bit unsigned integers [0, 3].
    Returns:
        (quantized_codes, scale, zero_point)
    """
    if not values:
        return [], 1.0, 0.0

    min_v = min(values)
    max_v = max(values)

    if max_v > min_v:
        scale = (max_v - min_v) / 3.0
    else:
        # Constant vector: scale defaults to 1.0; code is 0; dequantized matches exactly
        scale = 1.0

    zero_point = min_v

    q_tokens: List[int] = []
    for v in values:
        q = round((v - zero_point) / scale)
        q = max(0, min(3, q))  # Clamp strictly to 2 bits [0, 3]
        q_tokens.append(q)

    return q_tokens, scale, zero_point

def dequantize_2bit(q_tokens: List[int], scale: float, zero_point: float) -> List[float]:
    """Dequantizes 2-bit integer codes back to floating point representation."""
    return [q * scale + zero_point for q in q_tokens]

def quantize_key_per_channel(
    key_tensor: List[List[float]]
) -> Tuple[List[List[int]], List[float], List[float]]:
    """
    Quantizes Key cache per-channel across sequence tokens.
    Input tensor shape: [num_tokens, num_channels]
    Key cache exhibits channel-specific outliers across the temporal sequence dimension;
    grouping along tokens preserves channel-specific outlier distributions.
    Returns:
        (q_keys [T, D], per_channel_scales [D], per_channel_zero_points [D])
    """
    if not key_tensor or not key_tensor[0]:
        return [], [], []

    num_tokens = len(key_tensor)
    num_channels = len(key_tensor[0])

    for row in key_tensor:
        if len(row) != num_channels:
            raise ValueError(f"Inconsistent channel dimension: expected {num_channels}, got {len(row)}")

    q_keys = [[0] * num_channels for _ in range(num_tokens)]
    scales: List[float] = []
    zero_points: List[float] = []

    for d in range(num_channels):
        channel_vals = [key_tensor[t][d] for t in range(num_tokens)]
        q_col, scale, zp = quantize_2bit_asymmetric(channel_vals)
        scales.append(scale)
        zero_points.append(zp)
        for t in range(num_tokens):
            q_keys[t][d] = q_col[t]

    return q_keys, scales, zero_points

def dequantize_key_per_channel(
    q_keys: List[List[int]],
    scales: List[float],
    zero_points: List[float]
) -> List[List[float]]:
    """Dequantizes per-channel quantized Key cache back to floating point [T, D]."""
    if not q_keys or not q_keys[0]:
        return []

    num_tokens = len(q_keys)
    num_channels = len(q_keys[0])

    if len(scales) != num_channels or len(zero_points) != num_channels:
        raise ValueError("Scale/zero_point dimension does not match channel count")

    dequantized = [[0.0] * num_channels for _ in range(num_tokens)]
    for t in range(num_tokens):
        for d in range(num_channels):
            dequantized[t][d] = q_keys[t][d] * scales[d] + zero_points[d]

    return dequantized

def quantize_value_per_token(
    value_tensor: List[List[float]]
) -> Tuple[List[List[int]], List[float], List[float]]:
    """
    Quantizes Value cache per-token across channels.
    Input tensor shape: [num_tokens, num_channels]
    Value cache exhibits token-specific outliers (certain tokens have high magnitudes across channels);
    per-token quantization isolates token dynamic ranges.
    Returns:
        (q_values [T, D], per_token_scales [T], per_token_zero_points [T])
    """
    if not value_tensor or not value_tensor[0]:
        return [], [], []

    num_tokens = len(value_tensor)
    num_channels = len(value_tensor[0])

    for row in value_tensor:
        if len(row) != num_channels:
            raise ValueError(f"Inconsistent channel dimension: expected {num_channels}, got {len(row)}")

    q_values: List[List[int]] = []
    scales: List[float] = []
    zero_points: List[float] = []

    for t in range(num_tokens):
        token_vals = value_tensor[t]
        q_row, scale, zp = quantize_2bit_asymmetric(token_vals)
        q_values.append(q_row)
        scales.append(scale)
        zero_points.append(zp)

    return q_values, scales, zero_points

def dequantize_value_per_token(
    q_values: List[List[int]],
    scales: List[float],
    zero_points: List[float]
) -> List[List[float]]:
    """Dequantizes per-token quantized Value cache back to floating point [T, D]."""
    if not q_values or not q_values[0]:
        return []

    num_tokens = len(q_values)
    num_channels = len(q_values[0])

    if len(scales) != num_tokens or len(zero_points) != num_tokens:
        raise ValueError("Scale/zero_point dimension does not match token count")

    dequantized = [[0.0] * num_channels for _ in range(num_tokens)]
    for t in range(num_tokens):
        for d in range(num_channels):
            dequantized[t][d] = q_values[t][d] * scales[t] + zero_points[t]

    return dequantized

def max_residual_error(original: List[float], dequantized: List[float]) -> float:
    """Computes the maximum absolute residual error between original and dequantized values."""
    if len(original) != len(dequantized):
        raise ValueError("Length mismatch between original and dequantized")
    if not original:
        return 0.0
    return max(abs(o - d) for o, d in zip(original, dequantized))

if __name__ == "__main__":
    print("=" * 80)
    print("  CHAPTER 4: KIVI 2-BIT ASYMMETRIC KV CACHE QUANTIZER")
    print("=" * 80)
    raw_kv_channel = [0.12, -0.45, 0.88, 1.25, -0.05, 0.65, 0.40, -0.30]
    q_vals, scale, zp = quantize_2bit_asymmetric(raw_kv_channel)
    recon = dequantize_2bit(q_vals, scale, zp)
    
    print(f"Raw Channel (FP16): {[round(x, 2) for x in raw_kv_channel]}")
    print(f"Packed 2-Bit Codes: {q_vals}")
    print(f"Dequantized Output: {[round(x, 2) for x in recon]}")
    max_err = max_residual_error(raw_kv_channel, recon)
    bound = (max(raw_kv_channel) - min(raw_kv_channel)) / 6.0
    print(f"Max Residual Error: {max_err:.4f} (Theoretical Bound: <= {bound:.4f})")
    print("Compression: 16 bits -> 2 bits = 8.0x KV Cache Memory Savings!")
    print("=" * 80)
