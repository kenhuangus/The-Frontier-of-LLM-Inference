"""
DeepSeek Multi-Head Latent Attention (MLA) Simulator
Implements low-rank latent KV vector compression, decoupled RoPE keys,
and matrix absorption during autoregressive decoding.
"""
import math
from typing import List, Dict
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
        """Physical bytes stored in VRAM per token for this layer."""
        dims_per_token = self.cfg.d_c + self.cfg.d_h_r
        return int(dims_per_token * self.cfg.bytes_per_elem)

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
