"""
YaRN (Yet another RoPE extensioN) Spectral Frequency Scaling
Interpolates low-frequency rotary dimensions and preserves high-frequency local components.
"""
import math
from typing import List

def compute_yarn_frequencies(
    dim: int = 128,
    base: float = 10000.0,
    scale: float = 16.0 # 4k -> 64k extension
) -> List[float]:
    inv_freq = []
    for i in range(0, dim, 2):
        freq = 1.0 / (base ** (i / dim))
        # YaRN smooth ramp function
        wavelength = 2.0 * math.pi / freq
        if wavelength > 4096.0:
            # Low frequency: fully interpolated
            adjusted_freq = freq / scale
        else:
            # High frequency: extrapolate / maintain resolution
            adjusted_freq = freq
        inv_freq.append(round(adjusted_freq, 6))
    return inv_freq

if __name__ == "__main__":
    freqs = compute_yarn_frequencies(dim=64, scale=8.0)
    print("YaRN Sample Frequencies (interpolated low-frequency vs preserved high-frequency):")
    print(" ", freqs[:10])
