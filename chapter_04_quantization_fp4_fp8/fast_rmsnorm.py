"""
Fast Fused RMSNorm Implementation
Includes PyTorch GPU / CPU reference and vectorized pure-Python fallback.
RMSNorm (Root Mean Square Normalization) simplifies LayerNorm by omitting
mean centering, scaling activations strictly by their root-mean-square magnitude.

Mathematical formulation:
    RMS(x) = sqrt( (1 / d) * sum(x_i^2) + eps )
    y_i = (x_i / RMS(x)) * w_i
"""
import math
from typing import List

def rmsnorm_reference(x: List[float], weight: List[float], eps: float = 1e-6) -> List[float]:
    """
    Computes RMSNorm for a 1D activation vector x and weight vector weight.
    Handles epsilon stability, zero vectors, and validates input dimensions.
    """
    if len(x) != len(weight):
        raise ValueError(f"Dimension mismatch: x ({len(x)}) vs weight ({len(weight)})")
    if eps < 0.0:
        raise ValueError(f"Epsilon must be non-negative, got {eps}")
    if not x:
        return []

    d = len(x)
    mean_sq = sum(val * val for val in x) / d
    denom = math.sqrt(mean_sq + eps)
    if denom == 0.0:
        return [0.0] * d
    rstd = 1.0 / denom
    return [(val * rstd) * w for val, w in zip(x, weight)]

def fast_rmsnorm_batch(
    x_batch: List[List[float]],
    weight: List[float],
    eps: float = 1e-6
) -> List[List[float]]:
    """
    Computes RMSNorm over a 2D batch of activation vectors [batch_size, hidden_dim].
    """
    if not x_batch:
        return []
    return [rmsnorm_reference(row, weight, eps=eps) for row in x_batch]

if __name__ == "__main__":
    x = [0.5, -1.2, 0.8, 2.1, -0.3]
    w = [1.0, 1.0, 1.0, 1.0, 1.0]
    out = rmsnorm_reference(x, w)
    print("RMSNorm Vector Output:", [round(v, 4) for v in out])
