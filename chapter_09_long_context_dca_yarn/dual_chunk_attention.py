"""
Alibaba Qwen Dual-Chunk Attention (DCA) Factorization
Decomposes quadratic global attention into intra-chunk local attention
and inter-chunk representative attention for 1M+ token contexts.
"""
import math
from typing import List, Tuple, Optional

def dual_chunk_attention_simulation(
    seq_len: int = 131072,
    chunk_size: int = 4096,
    exact: bool = True
) -> Tuple[int, int, float]:
    """
    Computes computational savings of DCA versus standard full attention.
    Full attention: seq_len * seq_len
    DCA: intra-chunk local attention + inter-chunk representative interactions.
    
    Parameters:
        seq_len: Total sequence length in tokens.
        chunk_size: Chunk size in tokens (default 4096).
        exact: If True, accounts for remainder/uneven chunk lengths.
        
    Returns:
        (full_attention_flops, dca_flops, savings_percentage)
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    if seq_len == 0:
        return 0, 0, 0.0

    full_attention_flops = seq_len * seq_len

    # Edge case: sequence fits entirely within a single chunk
    if seq_len <= chunk_size:
        dca_flops = seq_len * seq_len
        savings = 0.0
        return full_attention_flops, dca_flops, savings

    if exact:
        num_full_chunks = seq_len // chunk_size
        remainder = seq_len % chunk_size

        # Intra-chunk FLOPs: full chunks squared plus remainder squared
        intra_chunk_flops = num_full_chunks * (chunk_size * chunk_size)
        if remainder > 0:
            intra_chunk_flops += remainder * remainder

        # Inter-chunk FLOPs: pairwise compressed inter-chunk interactions (64 tokens per chunk)
        full_chunk_pairs = (num_full_chunks * (num_full_chunks - 1)) // 2
        inter_chunk_flops = full_chunk_pairs * (chunk_size * 64)
        if remainder > 0:
            inter_chunk_flops += num_full_chunks * (remainder * 64)
    else:
        num_chunks = math.ceil(seq_len / chunk_size)
        intra_chunk_flops = num_chunks * (chunk_size * chunk_size)
        inter_chunk_flops = (num_chunks * (num_chunks - 1) // 2) * (chunk_size * 64)

    dca_flops = intra_chunk_flops + inter_chunk_flops
    savings = (full_attention_flops - dca_flops) / full_attention_flops * 100.0
    return full_attention_flops, dca_flops, round(savings, 2)


def get_chunk_assignment(token_idx: int, chunk_size: int) -> Tuple[int, int]:
    """
    Determines the macro chunk index and local offset for a token position.
    
    Parameters:
        token_idx: Zero-based global token index.
        chunk_size: Non-overlapping chunk size.
        
    Returns:
        (chunk_index, chunk_offset)
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if token_idx < 0:
        raise ValueError("token_idx must be non-negative")
    return token_idx // chunk_size, token_idx % chunk_size


def classify_attention_relation(q_idx: int, k_idx: int, chunk_size: int) -> str:
    """
    Classifies the DCA interaction between query token and key token.
    
    Relations:
        - "future_masked": Causal violation (k_idx > q_idx).
        - "intra": Both tokens reside in the same chunk (i == j).
        - "successive": Key resides in immediately preceding chunk (i == j + 1).
        - "inter": Key resides in earlier distant chunk (i > j + 1).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if q_idx < 0 or k_idx < 0:
        raise ValueError("token indices must be non-negative")

    if k_idx > q_idx:
        return "future_masked"

    q_chunk, _ = get_chunk_assignment(q_idx, chunk_size)
    k_chunk, _ = get_chunk_assignment(k_idx, chunk_size)

    if q_chunk == k_chunk:
        return "intra"
    elif q_chunk == k_chunk + 1:
        return "successive"
    else:
        return "inter"


def map_dca_positions(
    q_idx: int,
    k_idx: int,
    chunk_size: int,
    pretrained_len: int = 32768
) -> Tuple[int, int]:
    """
    Computes DCA RoPE position remapping for query and key tokens.
    
    Keys reuse positions within [0, chunk_size - 1].
    Queries are remapped based on intra, successive, or inter-chunk relationship
    such that all relative positions reside within the pretrained envelope.
    
    Returns:
        (q_remapped_pos, k_remapped_pos)
    """
    relation = classify_attention_relation(q_idx, k_idx, chunk_size)
    if relation == "future_masked":
        raise ValueError(f"Causal violation: query {q_idx} cannot attend to future key {k_idx}")

    _, q_off = get_chunk_assignment(q_idx, chunk_size)
    _, k_off = get_chunk_assignment(k_idx, chunk_size)

    k_pos = k_off

    if relation == "intra":
        q_pos = q_off
    elif relation == "successive":
        q_pos = q_off + chunk_size
    else: # inter-chunk
        q_pos = min(pretrained_len - 1, 2 * chunk_size + q_off)

    return q_pos, k_pos


def compute_dca_attention_scores(
    q_tokens: List[List[float]],
    k_tokens: List[List[float]],
    chunk_size: int,
    scale: Optional[float] = None,
    causal: bool = True
) -> List[List[float]]:
    """
    Computes DCA attention probability matrix between query and key token embeddings.
    Applies causal masking and softmax normalization.
    
    Parameters:
        q_tokens: Query embeddings [seq_len_q, dim].
        k_tokens: Key embeddings [seq_len_k, dim].
        chunk_size: DCA chunk size.
        scale: Optional scaling factor (default: 1.0 / sqrt(dim)).
        causal: If True, enforces causal lower-triangular mask.
        
    Returns:
        Normalized attention probability matrix [seq_len_q, seq_len_k].
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not q_tokens or not k_tokens:
        return []

    seq_len_q = len(q_tokens)
    seq_len_k = len(k_tokens)
    dim = len(q_tokens[0])

    for q in q_tokens:
        if len(q) != dim:
            raise ValueError("All query vectors must have identical dimensions")
    for k in k_tokens:
        if len(k) != dim:
            raise ValueError("All key vectors must match query dimension")

    if scale is None:
        scale = 1.0 / math.sqrt(dim) if dim > 0 else 1.0

    scores: List[List[float]] = []

    for i in range(seq_len_q):
        row: List[float] = []
        for j in range(seq_len_k):
            if causal and j > i:
                row.append(-1e9)
            else:
                dot = sum(q_tokens[i][d] * k_tokens[j][d] for d in range(dim))
                row.append(dot * scale)

        # Numerically stable softmax
        max_score = max(row)
        if max_score <= -1e8:
            # Entire row is masked
            scores.append([0.0] * seq_len_k)
        else:
            exp_scores = [math.exp(s - max_score) if s > -1e8 else 0.0 for s in row]
            sum_exp = sum(exp_scores)
            if sum_exp > 0:
                scores.append([s / sum_exp for s in exp_scores])
            else:
                scores.append([0.0] * seq_len_k)

    return scores


class DualChunkAttention:
    """
    High-level Dual-Chunk Attention abstraction.
    Partitions long-context sequences into intra-chunk and inter-chunk regimes.
    """
    def __init__(self, dim: int = 128, chunk_size: int = 4096, pretrained_len: int = 32768):
        if dim <= 0:
            raise ValueError("dim must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.dim = dim
        self.chunk_size = chunk_size
        self.pretrained_len = pretrained_len

    def get_assignment(self, token_idx: int) -> Tuple[int, int]:
        return get_chunk_assignment(token_idx, self.chunk_size)

    def classify_relation(self, q_idx: int, k_idx: int) -> str:
        return classify_attention_relation(q_idx, k_idx, self.chunk_size)

    def remap_positions(self, q_idx: int, k_idx: int) -> Tuple[int, int]:
        return map_dca_positions(q_idx, k_idx, self.chunk_size, self.pretrained_len)

    def compute_attention(
        self,
        q_tokens: List[List[float]],
        k_tokens: List[List[float]],
        causal: bool = True
    ) -> List[List[float]]:
        return compute_dca_attention_scores(q_tokens, k_tokens, self.chunk_size, causal=causal)

    def simulate_savings(self, seq_len: int) -> Tuple[int, int, float]:
        return dual_chunk_attention_simulation(seq_len, self.chunk_size)


if __name__ == "__main__":
    print("=" * 80)
    print("  CHAPTER 9: ALIBABA QWEN DUAL-CHUNK ATTENTION (DCA) FACTORIZATION")
    print("=" * 80)
    for length in [32768, 131072, 1048576]:
        full, dca, savings = dual_chunk_attention_simulation(length, chunk_size=4096)
        print(f"Context: {length:>8,} tokens | Full Attn: {full:>16,} | DCA: {dca:>14,} | Savings: {savings:>6.2f}%")
    print("=" * 80)
