"""
KV Cache Sizing and Cluster Fleet Capacity Model
Formulates dynamic memory footprint across Multi-Head Attention (MHA),
Grouped-Query Attention (GQA), and DeepSeek Multi-Head Latent Attention (MLA).
"""
from typing import Dict

def calculate_kv_cache_bytes_per_token(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    precision_bytes: int = 2
) -> int:
    """Dynamic KV cache bytes per token for standard MHA / GQA."""
    return 2 * num_layers * num_kv_heads * head_dim * precision_bytes

def calculate_mla_bytes_per_token(
    num_layers: int,
    latent_dim: int = 512,
    rope_dim: int = 64,
    precision_bytes: int = 2
) -> int:
    """DeepSeek Multi-Head Latent Attention (MLA) compressed KV cache footprint."""
    return num_layers * (latent_dim + rope_dim) * precision_bytes

def size_cluster_fleet(
    model_param_billions: float,
    context_tokens: int,
    concurrency: int,
    use_mla: bool = False,
    gpu_vram_gb: float = 80.0
) -> Dict[str, float]:
    """Calculates required GPU count for weights + dynamic KV cache."""
    weights_vram_gb = model_param_billions * 2.0 # FP16 weights
    if use_mla:
        kv_bytes_per_tok = calculate_mla_bytes_per_token(num_layers=80)
    else:
        kv_bytes_per_tok = calculate_kv_cache_bytes_per_token(num_layers=80, num_kv_heads=8, head_dim=128)
    
    total_kv_gb = (kv_bytes_per_tok * context_tokens * concurrency) / (1024**3)
    total_required_vram_gb = (weights_vram_gb + total_kv_gb) * 1.2 # 20% runtime headroom
    num_gpus = int(total_required_vram_gb / gpu_vram_gb) + 1
    return {
        "Weights_VRAM_GB": round(weights_vram_gb, 2),
        "Total_KV_Cache_GB": round(total_kv_gb, 2),
        "Total_Required_VRAM_GB": round(total_required_vram_gb, 2),
        "Minimum_GPUs_80GB": num_gpus
    }

if __name__ == "__main__":
    print("=" * 80)
    print("  CHAPTER 1: KV CACHE SIZING & FLEET CAPACITY CALCULATOR")
    print("=" * 80)
    
    gqa = calculate_kv_cache_bytes_per_token(num_layers=80, num_kv_heads=8, head_dim=128)
    mla = calculate_mla_bytes_per_token(num_layers=80, latent_dim=512, rope_dim=64)
    savings = ((gqa - mla) / gqa) * 100.0
    
    print(f"GQA-8 Footprint per token : {gqa:,} Bytes ({gqa/1024:.2f} KB)")
    print(f"MLA Footprint per token   : {mla:,} Bytes ({mla/1024:.2f} KB)")
    print(f"DeepSeek MLA VRAM Savings : {savings:.2f}% reduction per token!\n")
    
    fleet_gqa = size_cluster_fleet(70.0, 32768, 64, use_mla=False)
    fleet_mla = size_cluster_fleet(70.0, 32768, 64, use_mla=True)
    print("Cluster Sizing (Llama-3-70B, 32k context, 64 concurrent streams):")
    print(f"  Standard GQA : {fleet_gqa['Total_KV_Cache_GB']} GB KV cache -> {fleet_gqa['Minimum_GPUs_80GB']} H100 GPUs")
    print(f"  DeepSeek MLA : {fleet_mla['Total_KV_Cache_GB']} GB KV cache -> {fleet_mla['Minimum_GPUs_80GB']} H100 GPUs")
    print("=" * 80)
