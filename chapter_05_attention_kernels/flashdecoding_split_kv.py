"""
FlashDecoding Split-KV Attention Simulator
Partitions long KV sequence across multiple GPU thread blocks, executes parallel
partial softmax reduction, and merges outputs using Log-Sum-Exp (LSE).
"""
import math
from typing import List, Tuple

def flashdecoding_split_k(
    seq_len: int,
    num_splits: int = 4
) -> Tuple[List[float], float]:
    """Simulates Split-KV tree reduction over partial attention tiles."""
    split_size = seq_len // num_splits
    partial_maxes = []
    partial_sums = []
    
    # Stage 1: Parallel Split Computation
    for s in range(num_splits):
        # Simulate local attention logit distribution
        local_max = 12.5 + (s * 0.2)
        local_sum = sum(math.exp(local_max - (local_max + j*0.01)) for j in range(split_size))
        partial_maxes.append(local_max)
        partial_sums.append(local_sum)
        
    # Stage 2: Global LSE Reduction
    global_max = max(partial_maxes)
    global_sum = 0.0
    for l_max, l_sum in zip(partial_maxes, partial_sums):
        global_sum += l_sum * math.exp(l_max - global_max)
        
    global_lse = global_max + math.log(max(global_sum, 1e-9))
    return partial_maxes, round(global_lse, 4)

if __name__ == "__main__":
    print("=" * 80)
    print("  CHAPTER 5: FLASHDECODING SPLIT-KV ATTENTION REDUCTION")
    print("=" * 80)
    seq_len = 65536 # 64k context decode
    splits, lse = flashdecoding_split_k(seq_len, num_splits=8)
    print(f"Context Length: {seq_len:,} tokens | Parallel Splits: 8 threadblocks")
    print(f"Partial Tile Maxima: {[round(m, 2) for m in splits]}")
    print(f"Merged Global Log-Sum-Exp: {lse}")
    print("FlashDecoding converts single-threaded 64k decode into parallel 8x reduction!")
    print("=" * 80)
