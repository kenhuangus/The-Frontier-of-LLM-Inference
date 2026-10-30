"""
Comprehensive Architecture KV Cache Benchmark (MHA vs GQA vs MLA)
"""
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
        if m.is_mla:
            tok_bytes = (m.d_c + m.d_h_r) * m.n_layers * 2
        else:
            tok_bytes = 2 * m.n_layers * m.n_kv_heads * m.d_head * 2
            
        m4k = (tok_bytes * 4096) / (1024**3)
        m32k = (tok_bytes * 32768) / (1024**3)
        m128k = (tok_bytes * 131072) / (1024**3)
        m1m = (tok_bytes * 1048576) / (1024**3)
        
        tok_str = f"{tok_bytes/1024:.1f} KB" if tok_bytes >= 1024 else f"{tok_bytes} B"
        print(f"{m.name:<25} | {tok_str:<10} | {m4k:>7.2f} GB | {m32k:>7.2f} GB | {m128k:>7.2f} GB | {m1m:>7.2f} GB")
    print("-" * 85)

if __name__ == "__main__":
    run_benchmark()
