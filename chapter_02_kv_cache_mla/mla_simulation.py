"""
DeepSeek Multi-Head Latent Attention (MLA) Simulator
Implements low-rank latent KV vector compression, decoupled RoPE keys,
and matrix absorption during autoregressive decoding.
"""
import math
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class MLAModelConfig:
    name: str = "DeepSeek-V4-Pro"
    d_model: int = 7168
    n_heads: int = 128
    d_head: int = 128
    n_layers: int = 60
    d_c: int = 512       # KV latent compression dimension
    d_h_r: int = 64      # Decoupled RoPE head dimension
    bytes_per_elem: float = 2.0 # FP16

class MLALayerSimulation:
    def __init__(self, config: MLAModelConfig, layer_idx: int = 0):
        self.cfg = config
        self.layer_idx = layer_idx
        self.cached_latent_kv: List[List[float]] = [] # c_t^{KV} of dim d_c
        self.cached_rope_keys: List[List[float]] = [] # k_t^R of dim d_h_r

    def compute_token_kv_bytes(self) -> int:
        """Physical bytes stored in VRAM per token for this layer (c_t^{KV} + k_t^R)."""
        dims_per_token = self.cfg.d_c + self.cfg.d_h_r
        return int(dims_per_token * self.cfg.bytes_per_elem)

    def compute_mha_token_kv_bytes(self) -> int:
        """Standard MHA bytes stored in VRAM per token for this layer (2 * n_heads * d_head * bytes)."""
        return int(2 * self.cfg.n_heads * self.cfg.d_head * self.cfg.bytes_per_elem)

    def compute_gqa_token_kv_bytes(self, n_kv_heads: int) -> int:
        """Grouped-Query Attention (GQA) bytes stored in VRAM per token for this layer."""
        return int(2 * n_kv_heads * self.cfg.d_head * self.cfg.bytes_per_elem)

    def compute_vram_savings_vs_mha(self) -> float:
        """Percentage reduction in KV cache memory compared to standard Multi-Head Attention."""
        mha = self.compute_mha_token_kv_bytes()
        mla = self.compute_token_kv_bytes()
        return ((mha - mla) / mha) * 100.0

    def compute_vram_savings_vs_gqa(self, n_kv_heads: int) -> float:
        """Percentage reduction in KV cache memory compared to Grouped-Query Attention."""
        gqa = self.compute_gqa_token_kv_bytes(n_kv_heads)
        mla = self.compute_token_kv_bytes()
        return ((gqa - mla) / gqa) * 100.0

    def project_latent_kv(self, hidden_state: List[float], W_DKV: List[List[float]]) -> List[float]:
        """
        Computes low-rank KV latent compression: c_t^{KV} = W^{DKV} h_t.
        W_DKV shape: (d_c, d_model), hidden_state length: d_model.
        """
        if len(hidden_state) != self.cfg.d_model:
            raise ValueError(f"hidden_state dimension {len(hidden_state)} != d_model {self.cfg.d_model}")
        if len(W_DKV) != self.cfg.d_c or len(W_DKV[0]) != self.cfg.d_model:
            raise ValueError(f"W_DKV shape must be ({self.cfg.d_c}, {self.cfg.d_model})")
        
        # Matrix-vector multiply
        return [sum(W_DKV[i][j] * hidden_state[j] for j in range(self.cfg.d_model)) for i in range(self.cfg.d_c)]

    def project_rope_key(self, hidden_state: List[float], W_KR: List[List[float]]) -> List[float]:
        """
        Computes decoupled RoPE key: k_t^R = W^{KR} h_t (prior to applying RoPE rotary embedding).
        W_KR shape: (d_h_r, d_model), hidden_state length: d_model.
        """
        if len(hidden_state) != self.cfg.d_model:
            raise ValueError(f"hidden_state dimension {len(hidden_state)} != d_model {self.cfg.d_model}")
        if len(W_KR) != self.cfg.d_h_r or len(W_KR[0]) != self.cfg.d_model:
            raise ValueError(f"W_KR shape must be ({self.cfg.d_h_r}, {self.cfg.d_model})")

        return [sum(W_KR[i][j] * hidden_state[j] for j in range(self.cfg.d_model)) for i in range(self.cfg.d_h_r)]

    def absorb_query_matrix(self, query_head: List[float], W_UK: List[List[float]]) -> List[float]:
        """
        Simulates matrix absorption during decoding:
        Instead of uncompressing c_j^{KV} for all historical tokens: K_j = W^{UK} c_j^{KV},
        the projection matrix W^{UK} is absorbed into the single active query vector:
        q_{t, absorbed} = q_t W^{UK}.
        query_head shape: (d_head,), W_UK shape: (d_head, d_c).
        Returns q_{t, absorbed} of dimension d_c.
        """
        if len(query_head) != self.cfg.d_head:
            raise ValueError(f"query_head dimension {len(query_head)} != d_head {self.cfg.d_head}")
        if len(W_UK) != self.cfg.d_head or len(W_UK[0]) != self.cfg.d_c:
            raise ValueError(f"W_UK shape must be ({self.cfg.d_head}, {self.cfg.d_c})")

        # q_absorbed_k = sum_{i=0}^{d_head-1} q_t[i] * W_UK[i][k] for k in range(d_c)
        return [sum(query_head[i] * W_UK[i][k] for i in range(self.cfg.d_head)) for k in range(self.cfg.d_c)]

    def compute_absorbed_attention_score(
        self,
        q_absorbed: List[float],
        q_rope: List[float],
        c_kv: List[float],
        k_rope: List[float]
    ) -> float:
        """
        Computes attention score using decoupled absorption:
        Score = (q_{t, absorbed}^T c_j^{KV} + q_{t, RoPE}^T k_{j, RoPE}^R) / sqrt(d_head + d_h_r)
        """
        content_score = sum(q * c for q, c in zip(q_absorbed, c_kv))
        position_score = sum(qr * kr for qr, kr in zip(q_rope, k_rope))
        scale = math.sqrt(self.cfg.d_head + self.cfg.d_h_r)
        return (content_score + position_score) / scale

    def decode_step(self, token_idx: int) -> float:
        """
        Simulates an absorbed decode attention step over historical cached tokens
        without decompressing c_t^{KV} into high-dimensional standard key/values.
        """
        num_cached = len(self.cached_latent_kv)
        # FLOPs: 2 * (num_cached + 1) * (d_c + d_h_r) * n_heads
        step_flops = 2 * (num_cached + 1) * (self.cfg.d_c + self.cfg.d_h_r) * self.cfg.n_heads
        
        # Store newly generated token's low-rank vector
        self.cached_latent_kv.append([0.05] * self.cfg.d_c)
        self.cached_rope_keys.append([0.02] * self.cfg.d_h_r)
        return step_flops

    def reset(self) -> None:
        """Clears the KV cache history."""
        self.cached_latent_kv.clear()
        self.cached_rope_keys.clear()

if __name__ == "__main__":
    print("=" * 80)
    print("  CHAPTER 2: DEEPSEEK MULTI-HEAD LATENT ATTENTION (MLA) SIMULATOR")
    print("=" * 80)
    cfg = MLAModelConfig()
    mla_layer = MLALayerSimulation(cfg)
    
    bytes_per_tok = mla_layer.compute_token_kv_bytes()
    # Standard MHA would store: 2 * n_heads * d_head * bytes = 2 * 128 * 128 * 2 = 65,536 bytes
    mha_bytes = 2 * cfg.n_heads * cfg.d_head * int(cfg.bytes_per_elem)
    savings = ((mha_bytes - bytes_per_tok) / mha_bytes) * 100.0
    
    print(f"Model Configuration: {cfg.name} ({cfg.n_layers} Layers, {cfg.n_heads} Heads)")
    print(f"  Standard MHA VRAM per token/layer : {mha_bytes:,} Bytes")
    print(f"  DeepSeek MLA VRAM per token/layer : {bytes_per_tok:,} Bytes (d_c={cfg.d_c}, d_h_r={cfg.d_h_r})")
    print(f"  Total Memory Reduction            : {savings:.2f}% VRAM saved!\n")
    
    print("Simulating 100 autoregressive decode steps with tensor absorption:")
    total_flops = 0
    for t in range(100):
        flops = mla_layer.decode_step(t)
        total_flops += flops
    print(f"  Cached Tokens in VRAM: {len(mla_layer.cached_latent_kv)}")
    print(f"  Total Decode Attention FLOPs: {total_flops:,.0f} FLOPs")
    print("=" * 80)
