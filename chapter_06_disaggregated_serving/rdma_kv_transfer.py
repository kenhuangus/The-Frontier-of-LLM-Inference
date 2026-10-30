"""
Mooncake 4-Tier RDMA KV Cache Transfer Simulator
Models RoCEv2 / InfiniBand GPUDirect transfer of prefilled KV cache blocks.
"""
def calculate_rdma_transfer_time_ms(
    kv_cache_bytes: int,
    network_bandwidth_gbps: float = 400.0,
    base_latency_us: float = 5.0
) -> float:
    bw_bytes_per_sec = (network_bandwidth_gbps * 1e9) / 8.0
    transfer_sec = (kv_cache_bytes / bw_bytes_per_sec) + (base_latency_us * 1e-6)
    return transfer_sec * 1000.0

if __name__ == "__main__":
    # 4000 token prompt under DeepSeek MLA = 4000 * 576 * 2 bytes = 4.6 MB
    kv_bytes = 4000 * 576 * 2
    transfer_ms = calculate_rdma_transfer_time_ms(kv_bytes, network_bandwidth_gbps=400.0)
    print(f"KV Cache Payload: {kv_bytes / 1024**2:.2f} MB")
    print(f"RDMA Transfer Time over 400Gbps RoCEv2: {transfer_ms:.3f} ms (sub-millisecond handoff!)")
