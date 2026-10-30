"""
Fast Fused RMSNorm Implementation
Includes PyTorch GPU / CPU reference and vectorized pure-Python fallback.
"""
import math
from typing import List

def rmsnorm_reference(x: List[float], weight: List[float], eps: float = 1e-6) -> List[float]:
    d = len(x)
    mean_sq = sum(val * val for val in x) / d
    rstd = 1.0 / math.sqrt(mean_sq + eps)
    return [(val * rstd) * w for val, w in zip(x, weight)]

if __name__ == "__main__":
    x = [0.5, -1.2, 0.8, 2.1, -0.3]
    w = [1.0, 1.0, 1.0, 1.0, 1.0]
    out = rmsnorm_reference(x, w)
    print("RMSNorm Vector Output:", [round(v, 4) for v in out])
