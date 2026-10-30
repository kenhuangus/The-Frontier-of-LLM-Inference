"""
FlashDecoding Split-KV Attention Simulator
Partitions long KV sequence across multiple GPU thread blocks, executes parallel
partial softmax reduction, and merges outputs using online Log-Sum-Exp (LSE) resynthesis.
"""
from dataclasses import dataclass
import math
from typing import List, Optional, Tuple


@dataclass
class SplitKPartial:
    """
    Partial attention statistics produced by Stage 1 of Split-KV FlashDecoding.
    Encapsulates (O_s, m_s, l_s) tuple for each split.
    """
    split_id: int
    start_idx: int
    end_idx: int
    num_tokens: int
    max_logit: float
    unnorm_sum: float
    output_vector: Optional[List[float]] = None  # Unnormalized output projection


def online_softmax_merge_pair(a: SplitKPartial, b: SplitKPartial) -> SplitKPartial:
    """
    Merges two partial split results using the Milakov-Gimelshein online softmax
    rescaling formula. Handles maximum exponent shifts and unnormalized sum re-weighting
    with provable numerical stability against floating-point overflow.
    """
    if a.num_tokens == 0:
        return b
    if b.num_tokens == 0:
        return a

    m_merged = max(a.max_logit, b.max_logit)
    w_a = math.exp(a.max_logit - m_merged)
    w_b = math.exp(b.max_logit - m_merged)

    sum_merged = a.unnorm_sum * w_a + b.unnorm_sum * w_b

    vec_merged = None
    if a.output_vector is not None and b.output_vector is not None:
        vec_merged = [
            a_val * w_a + b_val * w_b
            for a_val, b_val in zip(a.output_vector, b.output_vector)
        ]

    return SplitKPartial(
        split_id=min(a.split_id, b.split_id),
        start_idx=min(a.start_idx, b.start_idx),
        end_idx=max(a.end_idx, b.end_idx),
        num_tokens=a.num_tokens + b.num_tokens,
        max_logit=m_merged,
        unnorm_sum=sum_merged,
        output_vector=vec_merged,
    )


def flashdecoding_reduce_tree(partials: List[SplitKPartial]) -> SplitKPartial:
    """
    Reduces S partial split results using a hierarchical binary reduction tree.
    Executes in ceil(log2 S) stages, mirroring shared memory / warp shuffle reductions.
    """
    if not partials:
        raise ValueError("Cannot reduce empty list of partial splits")

    current = partials
    while len(current) > 1:
        next_level = []
        for i in range(0, len(current), 2):
            if i + 1 < len(current):
                next_level.append(online_softmax_merge_pair(current[i], current[i + 1]))
            else:
                next_level.append(current[i])
        current = next_level

    return current[0]


def flashdecoding_reduce_flat(partials: List[SplitKPartial]) -> SplitKPartial:
    """
    Reduces S partial split results using Stage 2 multi-way reduction.
    Finds global maximum across all splits and rescales all sums and output vectors
    simultaneously in global workspace memory.
    """
    if not partials:
        raise ValueError("Cannot reduce empty list of partial splits")

    active = [p for p in partials if p.num_tokens > 0]
    if not active:
        return partials[0]

    global_max = max(p.max_logit for p in active)
    global_sum = 0.0

    d = len(active[0].output_vector) if active[0].output_vector is not None else 0
    global_vec = [0.0] * d if d > 0 else None

    total_tokens = sum(p.num_tokens for p in active)

    for p in active:
        w = math.exp(p.max_logit - global_max)
        global_sum += p.unnorm_sum * w
        if global_vec is not None and p.output_vector is not None:
            for dim in range(d):
                global_vec[dim] += p.output_vector[dim] * w

    return SplitKPartial(
        split_id=0,
        start_idx=min(p.start_idx for p in active),
        end_idx=max(p.end_idx for p in active),
        num_tokens=total_tokens,
        max_logit=global_max,
        unnorm_sum=global_sum,
        output_vector=global_vec,
    )


def flashdecoding_online_softmax_resynthesis(
    partial_outputs: List[List[float]],
    partial_maxes: List[float],
    partial_sums: List[float],
) -> Tuple[List[float], float, float]:
    """
    Online softmax resynthesis across splits (Question 13 derivation):
    Step 1: m_global = max_s(m_s)
    Step 2: w_s = exp(m_s - m_global), l_global = sum_s(w_s * l_s)
    Step 3: O_final = (sum_s w_s * O_s) / l_global
    Global LSE: LSE = m_global + ln(l_global)

    Returns:
    - O_final: Normalized attention output vector [d]
    - m_global: Global logit maximum
    - lse_global: Global Log-Sum-Exp
    """
    if not (len(partial_outputs) == len(partial_maxes) == len(partial_sums)):
        raise ValueError("Mismatched lengths among partial outputs, maxes, and sums")
    if not partial_outputs:
        raise ValueError("Empty partial inputs")

    m_global = max(partial_maxes)
    l_global = 0.0
    d = len(partial_outputs[0])
    o_unnorm = [0.0] * d

    for o_s, m_s, l_s in zip(partial_outputs, partial_maxes, partial_sums):
        w_s = math.exp(m_s - m_global)
        l_global += l_s * w_s
        for dim in range(d):
            o_unnorm[dim] += o_s[dim] * w_s

    denom = max(l_global, 1e-12)
    o_final = [val / denom for val in o_unnorm]
    lse_global = m_global + math.log(denom)

    return o_final, m_global, lse_global


def naive_attention(
    q: List[float],
    k: List[List[float]],
    v: List[List[float]],
) -> Tuple[List[float], float]:
    """
    Exact single-thread reference implementation of scaled dot-product attention
    over a full sequence.
    """
    d = len(q)
    scale = 1.0 / math.sqrt(d)
    scores = [
        sum(qi * ki_val for qi, ki_val in zip(q, ki)) * scale
        for ki in k
    ]
    m = max(scores)
    exp_scores = [math.exp(s - m) for s in scores]
    l = sum(exp_scores)
    lse = m + math.log(max(l, 1e-12))

    out = [0.0] * d
    for exp_s, vi in zip(exp_scores, v):
        for dim in range(d):
            out[dim] += (exp_s / l) * vi[dim]

    return out, lse


def flashdecoding_split_kv_attention(
    q: List[float],
    k: List[List[float]],
    v: List[List[float]],
    num_splits: int = 4,
    use_tree_reduction: bool = False,
) -> Tuple[List[float], float]:
    """
    Simulates full Split-KV FlashDecoding attention with parallel Stage 1 splits
    and Stage 2 hierarchical or flat tree reduction.
    """
    seq_len = len(k)
    if seq_len == 0:
        raise ValueError("Sequence length cannot be empty")
    if num_splits <= 0:
        raise ValueError("num_splits must be positive")

    d = len(q)
    scale = 1.0 / math.sqrt(d)
    partials: List[SplitKPartial] = []

    # Stage 1: Parallel Split Execution across threadblocks
    for s in range(num_splits):
        start = (s * seq_len) // num_splits
        end = ((s + 1) * seq_len) // num_splits
        chunk_len = end - start

        if chunk_len == 0:
            partials.append(
                SplitKPartial(
                    split_id=s,
                    start_idx=start,
                    end_idx=end,
                    num_tokens=0,
                    max_logit=-float("inf"),
                    unnorm_sum=0.0,
                    output_vector=[0.0] * d,
                )
            )
            continue

        split_k = k[start:end]
        split_v = v[start:end]

        scores = [
            sum(qi * ki_val for qi, ki_val in zip(q, ki)) * scale
            for ki in split_k
        ]
        m_s = max(scores)
        l_s = 0.0
        v_s = [0.0] * d

        for s_val, vi in zip(scores, split_v):
            w = math.exp(s_val - m_s)
            l_s += w
            for dim in range(d):
                v_s[dim] += w * vi[dim]

        partials.append(
            SplitKPartial(
                split_id=s,
                start_idx=start,
                end_idx=end,
                num_tokens=chunk_len,
                max_logit=m_s,
                unnorm_sum=l_s,
                output_vector=v_s,
            )
        )

    # Stage 2: Tree Reduction or Flat Reduction
    if use_tree_reduction:
        reduced = flashdecoding_reduce_tree(partials)
    else:
        reduced = flashdecoding_reduce_flat(partials)

    l_final = max(reduced.unnorm_sum, 1e-12)
    o_final = [val / l_final for val in reduced.output_vector]
    lse_final = reduced.max_logit + math.log(l_final)

    return o_final, lse_final


def flashdecoding_split_k(
    seq_len: int,
    num_splits: int = 4,
) -> Tuple[List[float], float]:
    """
    Simulates Split-KV tree reduction over partial attention tiles.
    Partitions context length into num_splits chunks and performs Stage 2 LSE reduction.
    """
    if seq_len <= 0 or num_splits <= 0:
        raise ValueError("seq_len and num_splits must be positive integers")

    partial_maxes: List[float] = []
    partial_sums: List[float] = []

    # Stage 1: Parallel Split Computation across threadblocks
    for s in range(num_splits):
        start = (s * seq_len) // num_splits
        end = ((s + 1) * seq_len) // num_splits
        split_size = end - start

        if split_size > 0:
            local_max = 12.5 + (s * 0.2)
            # Evaluate exponential sums within the split
            local_sum = sum(
                math.exp(local_max - (local_max + j * 0.01))
                for j in range(split_size)
            )
            partial_maxes.append(local_max)
            partial_sums.append(local_sum)
        else:
            partial_maxes.append(-float("inf"))
            partial_sums.append(0.0)

    # Stage 2: Global LSE Reduction
    valid_splits = [
        (m, l) for m, l in zip(partial_maxes, partial_sums) if l > 0.0
    ]

    if not valid_splits:
        return partial_maxes, 0.0

    global_max = max(m for m, _ in valid_splits)
    global_sum = sum(l * math.exp(m - global_max) for m, l in valid_splits)
    global_lse = global_max + math.log(max(global_sum, 1e-9))

    return partial_maxes, round(global_lse, 4)


if __name__ == "__main__":
    print("=" * 80)
    print("  CHAPTER 5: FLASHDECODING SPLIT-KV ATTENTION REDUCTION")
    print("=" * 80)
    seq_len = 65536  # 64k context decode
    splits, lse = flashdecoding_split_k(seq_len, num_splits=8)
    print(f"Context Length: {seq_len:,} tokens | Parallel Splits: 8 threadblocks")
    print(f"Partial Tile Maxima: {[round(m, 2) for m in splits]}")
    print(f"Merged Global Log-Sum-Exp: {lse}")
    print("FlashDecoding converts single-threaded 64k decode into parallel 8x reduction!")
    print("=" * 80)
