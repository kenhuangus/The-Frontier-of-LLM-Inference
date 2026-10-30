"""
Mooncake 4-Tier RDMA KV Cache Transfer Simulator
Models RoCEv2 / InfiniBand GPUDirect transfer of prefilled KV cache blocks.
Evaluates serialization delay, base NIC latency, MTU chunking overhead,
and effective line-rate bandwidth utilization across payload regimes.
"""
import math
from typing import Dict, Optional, Union

def calculate_rdma_transfer_time_ms(
    kv_cache_bytes: int,
    network_bandwidth_gbps: float = 400.0,
    base_latency_us: float = 5.0,
    mtu_bytes: Optional[int] = None,
    header_overhead_bytes: int = 64
) -> float:
    """Calculates RDMA transfer time in milliseconds for a KV cache payload.
    
    Supports both ideal line-rate transport and packetized MTU chunking
    (accounting for RoCEv2 Ethernet/IP/UDP/BTH header framing overhead).
    """
    if kv_cache_bytes < 0:
        raise ValueError(f"kv_cache_bytes cannot be negative, got {kv_cache_bytes}")
    if network_bandwidth_gbps <= 0:
        raise ValueError(f"network_bandwidth_gbps must be positive, got {network_bandwidth_gbps}")
    if base_latency_us < 0:
        raise ValueError(f"base_latency_us cannot be negative, got {base_latency_us}")
    if mtu_bytes is not None and mtu_bytes <= 0:
        raise ValueError(f"mtu_bytes must be positive, got {mtu_bytes}")
    if header_overhead_bytes < 0:
        raise ValueError(f"header_overhead_bytes cannot be negative, got {header_overhead_bytes}")

    if kv_cache_bytes == 0:
        return 0.0

    bw_bytes_per_sec = (network_bandwidth_gbps * 1e9) / 8.0

    if mtu_bytes is None:
        wire_bytes = float(kv_cache_bytes)
    else:
        num_packets = math.ceil(kv_cache_bytes / mtu_bytes)
        wire_bytes = float(kv_cache_bytes + (num_packets * header_overhead_bytes))

    transfer_sec = (wire_bytes / bw_bytes_per_sec) + (base_latency_us * 1e-6)
    return transfer_sec * 1000.0

def calculate_effective_bandwidth_gbps(
    kv_cache_bytes: int,
    transfer_time_ms: float
) -> float:
    """Calculates achieved goodput bandwidth in Gigabits per second."""
    if kv_cache_bytes <= 0 or transfer_time_ms <= 0:
        return 0.0
    transfer_sec = transfer_time_ms / 1000.0
    bits_transferred = kv_cache_bytes * 8.0
    return (bits_transferred / transfer_sec) / 1e9

def simulate_chunked_rdma_transfer(
    kv_cache_bytes: int,
    network_bandwidth_gbps: float = 400.0,
    base_latency_us: float = 5.0,
    mtu_bytes: int = 4096,
    header_overhead_bytes: int = 64
) -> Dict[str, Union[float, int]]:
    """Simulates packetized RoCEv2 transfer and reports transmission metrics."""
    if kv_cache_bytes < 0:
        raise ValueError(f"kv_cache_bytes cannot be negative, got {kv_cache_bytes}")
    if mtu_bytes <= 0:
        raise ValueError(f"mtu_bytes must be positive, got {mtu_bytes}")

    if kv_cache_bytes == 0:
        return {
            "kv_cache_bytes": 0,
            "payload_mb": 0.0,
            "network_bandwidth_gbps": network_bandwidth_gbps,
            "base_latency_us": base_latency_us,
            "mtu_bytes": mtu_bytes,
            "num_packets": 0,
            "wire_bytes": 0,
            "transfer_time_ms": 0.0,
            "effective_bandwidth_gbps": 0.0,
            "line_rate_efficiency": 0.0
        }

    num_packets = math.ceil(kv_cache_bytes / mtu_bytes)
    wire_bytes = kv_cache_bytes + (num_packets * header_overhead_bytes)
    transfer_ms = calculate_rdma_transfer_time_ms(
        kv_cache_bytes=kv_cache_bytes,
        network_bandwidth_gbps=network_bandwidth_gbps,
        base_latency_us=base_latency_us,
        mtu_bytes=mtu_bytes,
        header_overhead_bytes=header_overhead_bytes
    )
    effective_bw = calculate_effective_bandwidth_gbps(kv_cache_bytes, transfer_ms)
    efficiency = effective_bw / network_bandwidth_gbps if network_bandwidth_gbps > 0 else 0.0

    return {
        "kv_cache_bytes": kv_cache_bytes,
        "payload_mb": round(kv_cache_bytes / (1024 ** 2), 4),
        "network_bandwidth_gbps": network_bandwidth_gbps,
        "base_latency_us": base_latency_us,
        "mtu_bytes": mtu_bytes,
        "num_packets": num_packets,
        "wire_bytes": wire_bytes,
        "transfer_time_ms": round(transfer_ms, 4),
        "effective_bandwidth_gbps": round(effective_bw, 3),
        "line_rate_efficiency": round(efficiency, 4)
    }

if __name__ == "__main__":
    # 4000 token prompt under DeepSeek MLA = 4000 * 576 * 2 bytes = 4.6 MB
    kv_bytes = 4000 * 576 * 2
    transfer_ms = calculate_rdma_transfer_time_ms(kv_bytes, network_bandwidth_gbps=400.0)
    print(f"KV Cache Payload: {kv_bytes / 1024**2:.2f} MB")
    print(f"RDMA Transfer Time over 400Gbps RoCEv2: {transfer_ms:.3f} ms (sub-millisecond handoff!)")
