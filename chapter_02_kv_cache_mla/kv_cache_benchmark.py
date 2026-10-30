"""
Comprehensive Architecture KV Cache Benchmark (MHA vs GQA vs MLA)
Calculates per-token memory footprint, multi-context VRAM sizing,
memory-bound decode latency, and autoregressive generation throughput.
"""
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class ArchConfig:
    name: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_head: int
    is_mla: bool = False
    d_c: int = 512
    d_h_r: int = 64

def compute_token_kv_bytes(arch: ArchConfig, bytes_per_elem: float = 2.0) -> int:
    """Computes total KV cache bytes stored per token across all layers."""
    if arch.is_mla:
        return int((arch.d_c + arch.d_h_r) * arch.n_layers * bytes_per_elem)
    else:
        return int(2 * arch.n_layers * arch.n_kv_heads * arch.d_head * bytes_per_elem)

def compute_context_vram_gb(arch: ArchConfig, context_len: int, bytes_per_elem: float = 2.0) -> float:
    """Computes total KV cache VRAM footprint in GiB for a given context length."""
    tok_bytes = compute_token_kv_bytes(arch, bytes_per_elem)
    return (tok_bytes * context_len) / (1024 ** 3)

def calculate_decode_latency_ms(
    arch: ArchConfig,
    context_len: int,
    batch_size: int = 1,
    memory_bandwidth_tb_s: float = 3.35,
    bytes_per_elem: float = 2.0
) -> float:
    """
    Computes theoretical memory-bound attention KV cache read latency (in milliseconds)
    per decode step under given GPU HBM bandwidth (default 3.35 TB/s for NVIDIA H100 SXM5).
    """
    tok_bytes = compute_token_kv_bytes(arch, bytes_per_elem)
    total_bytes = tok_bytes * context_len * batch_size
    bandwidth_bytes_per_sec = memory_bandwidth_tb_s * (10 ** 12)
    latency_sec = total_bytes / bandwidth_bytes_per_sec
    return latency_sec * 1000.0

def calculate_generation_throughput(
    arch: ArchConfig,
    context_len: int,
    batch_size: int = 1,
    memory_bandwidth_tb_s: float = 3.35,
    bytes_per_elem: float = 2.0
) -> float:
    """
    Computes theoretical attention-bound generation throughput (tokens/second)
    constrained by KV cache memory bandwidth during autoregressive decoding.
    """
    latency_ms = calculate_decode_latency_ms(arch, context_len, batch_size, memory_bandwidth_tb_s, bytes_per_elem)
    if latency_ms <= 0:
        return 0.0
    return (batch_size * 1000.0) / latency_ms

def run_benchmark():
    models = [
        ArchConfig("Llama-1 65B (MHA)", 80, 64, 64, 128),
        ArchConfig("Llama-3 70B (GQA-8)", 80, 64, 8, 128),
        ArchConfig("Llama-3 405B (GQA-8)", 126, 128, 8, 128),
        ArchConfig("DeepSeek-V3/V4 (MLA)", 60, 128, 128, 128, is_mla=True, d_c=512, d_h_r=64)
    ]
    contexts = [4096, 32768, 131072, 1048576]
    
    print(f"{'Model Architecture':<25} | {'Bytes/Tok':<10} | {'4k Ctx':<9} | {'32k Ctx':<9} | {'128k Ctx':<9} | {'1M Ctx':<9}")
    print("-" * 85)
    for m in models:
        tok_bytes = compute_token_kv_bytes(m)
        m4k = compute_context_vram_gb(m, 4096)
        m32k = compute_context_vram_gb(m, 32768)
        m128k = compute_context_vram_gb(m, 131072)
        m1m = compute_context_vram_gb(m, 1048576)
        
        tok_str = f"{tok_bytes/1024:.1f} KB" if tok_bytes >= 1024 else f"{tok_bytes} B"
        print(f"{m.name:<25} | {tok_str:<10} | {m4k:>7.2f} GB | {m32k:>7.2f} GB | {m128k:>7.2f} GB | {m1m:>7.2f} GB")
    print("-" * 85)

    print("\nAttention Decode Latency & Throughput (128k Context, Batch=1, H100 3.35 TB/s):")
    print(f"{'Model Architecture':<25} | {'Decode Latency':<16} | {'Max Throughput':<16}")
    print("-" * 65)
    for m in models:
        lat = calculate_decode_latency_ms(m, context_len=131072, batch_size=1, memory_bandwidth_tb_s=3.35)
        thr = calculate_generation_throughput(m, context_len=131072, batch_size=1, memory_bandwidth_tb_s=3.35)
        print(f"{m.name:<25} | {lat:>10.2f} ms     | {thr:>10.1f} tok/s")
    print("-" * 65)

if __name__ == "__main__":
    run_benchmark()
