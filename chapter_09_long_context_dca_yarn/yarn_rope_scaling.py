"""
YaRN (Yet another RoPE extensioN) Spectral Frequency Scaling
Interpolates low-frequency rotary dimensions and preserves high-frequency local components.
Reference: Peng et al. (2023), arXiv:2309.00071
"""
import math
from typing import List, Tuple, Optional

def compute_yarn_frequencies(
    dim: int = 128,
    base: float = 10000.0,
    scale: float = 16.0, # Target extension scale (e.g. 4k -> 64k)
    original_context_len: int = 4096,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    extrapolation_factor: float = 1.0
) -> List[float]:
    """
    Computes YaRN frequency modulation across Rotary Position Embedding dimensions.
    
    Partitions frequencies into three regimes:
      1. High-frequency regime (wavelength <= low): Zero interpolation (ramp = 0.0),
         preserving native resolution for local syntactic tokens.
      2. Low-frequency regime (wavelength >= high): Full linear interpolation (ramp = 1.0),
         scaling frequency by 1.0 / scale to prevent out-of-distribution phase wrap.
      3. Medium-frequency regime (low < wavelength < high): Smooth linear ramp gamma(r).
      
    Edge Cases Handled:
      - scale == 1.0: Exact identity (no scaling applied, native RoPE).
      - dim == 0: Returns empty list [] without division-by-zero error.
      - dim < 0 or odd dim: Raises ValueError.
      - scale <= 0: Raises ValueError.
    """
    if dim < 0:
        raise ValueError("dim must be a non-negative even integer")
    if dim == 0:
        return []
    if dim % 2 != 0:
        raise ValueError("dim must be an even integer")
    if scale <= 0:
        raise ValueError("scale must be strictly positive")
    if base <= 0:
        raise ValueError("base must be strictly positive")
    if beta_fast <= 0 or beta_slow <= 0:
        raise ValueError("beta thresholds must be strictly positive")

    # Scale 1.0 identity: return native base RoPE inverse frequencies
    if scale == 1.0:
        return [round(1.0 / (base ** (i / dim)), 6) for i in range(0, dim, 2)]

    low = original_context_len / beta_fast
    high = original_context_len / beta_slow

    inv_freq = []
    for i in range(0, dim, 2):
        freq = 1.0 / (base ** (i / dim))
        wavelength = 2.0 * math.pi / freq

        # YaRN smooth piecewise linear ramp function gamma(r_i)
        if high > low:
            ramp = max(0.0, min(1.0, (wavelength - low) / (high - low)))
        else:
            ramp = 1.0 if wavelength >= high else 0.0

        # Blend native frequency (extrapolation) and scaled frequency (interpolation)
        adjusted_freq = (1.0 - ramp) * freq + ramp * (freq / scale)
        inv_freq.append(round(adjusted_freq, 6))

    return inv_freq


def compute_yarn_mscale(scale: float = 1.0, attn_factor: float = 1.0) -> float:
    """
    Computes YaRN attention logit temperature calibration factor (1/t).
    Law: 1/t = 0.1 * ln(s) + 1.0 for scale > 1.0.
    Restores pre-trained attention sharpness across long contexts.
    """
    if scale <= 0:
        raise ValueError("scale must be strictly positive")
    if scale <= 1.0:
        return 1.0
    return round((0.1 * math.log(scale) + 1.0) * attn_factor, 6)


def compute_yarn_rotary_embedding(
    pos: int,
    freqs: List[float],
    mscale: float = 1.0
) -> Tuple[List[float], List[float]]:
    """
    Computes cosine and sine rotary embeddings for a given token position.
    
    Handles:
      - pos < 0: Mathematically consistent negative position angles (cos(-theta) = cos(theta),
                 sin(-theta) = -sin(theta)).
      - pos == 0: Identity rotation (cos = 1.0, sin = 0.0).
      - empty freqs: Returns empty lists ([], []).
    """
    if not freqs:
        return [], []

    cos_half = [math.cos(pos * f) * mscale for f in freqs]
    sin_half = [math.sin(pos * f) * mscale for f in freqs]
    # Standard RoPE half-split concatenation [dim // 2, dim // 2] -> [dim]
    return cos_half + cos_half, sin_half + sin_half


def apply_yarn_rope(
    vec: List[float],
    pos: int,
    freqs: List[float],
    mscale: float = 1.0
) -> List[float]:
    """
    Applies YaRN-scaled rotary position embedding to an input embedding vector.
    Rotates vector pairs using rotate_half standard transformation.
    """
    if not vec and not freqs:
        return []
    expected_dim = 2 * len(freqs)
    if len(vec) != expected_dim:
        raise ValueError(f"Vector dim {len(vec)} does not match frequency dim {expected_dim}")

    half = len(freqs)
    x1 = vec[:half]
    x2 = vec[half:]
    # rotate_half(x) = [-x2, x1]
    rotated_half = [-x for x in x2] + x1

    cos_emb, sin_emb = compute_yarn_rotary_embedding(pos, freqs, mscale=1.0)

    # (x * cos) + (rotate_half(x) * sin) scaled by mscale
    return [
        round(((vec[i] * cos_emb[i]) + (rotated_half[i] * sin_emb[i])) * mscale, 6)
        for i in range(len(vec))
    ]


class YaRNScaledRotaryEmbedding:
    """
    Production YaRN Module reference implementation.
    Reference: Peng et al. (2023), arXiv:2309.00071.
    """
    def __init__(
        self,
        dim: int = 128,
        max_position_embeddings: int = 65536,
        original_max_position_embeddings: int = 4096,
        base: float = 10000.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        attn_factor: float = 1.0
    ):
        if dim <= 0 or dim % 2 != 0:
            raise ValueError("dim must be a positive even integer")
        self.dim = dim
        self.max_pos = max_position_embeddings
        self.orig_max_pos = original_max_position_embeddings
        self.base = base
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.attn_factor = attn_factor
        self.scale = max_position_embeddings / original_max_position_embeddings

        self.inv_freq = compute_yarn_frequencies(
            dim=self.dim,
            base=self.base,
            scale=self.scale,
            original_context_len=self.orig_max_pos,
            beta_fast=self.beta_fast,
            beta_slow=self.beta_slow
        )
        self.mscale = compute_yarn_mscale(self.scale, self.attn_factor)

    def get_embedding(self, pos: int) -> Tuple[List[float], List[float]]:
        return compute_yarn_rotary_embedding(pos, self.inv_freq, self.mscale)

    def apply_rope(self, vec: List[float], pos: int) -> List[float]:
        return apply_yarn_rope(vec, pos, self.inv_freq, self.mscale)


if __name__ == "__main__":
    freqs = compute_yarn_frequencies(dim=64, scale=8.0)
    print("YaRN Sample Frequencies (interpolated low-frequency vs preserved high-frequency):")
    print(" ", freqs[:10])
    mscale = compute_yarn_mscale(scale=8.0)
    print(f"YaRN Attention Temperature Multiplier (1/t for 8x): {mscale:.4f}")
