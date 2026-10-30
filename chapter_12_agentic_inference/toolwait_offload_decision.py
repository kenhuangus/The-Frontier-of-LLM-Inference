"""
Min-Waste Optimizer: Retain vs Offload vs Discard during Tool-Wait
Evaluates GPU memory holding cost against re-prefill computational cost.
Provides dynamic offloading threshold models across PCIe Gen4, PCIe Gen5, and CXL interconnects.
"""
from typing import Optional


def kv_during_wait(
    wait_ms: float,
    prefix_tok: int,
    hold_per_ms: float = 1e-6,
    prefill_per_tok: float = 1e-4
) -> str:
    """
    Four-line InferCept min-waste policy evaluating GPU memory holding cost against re-prefill.
    """
    if wait_ms < 0:
        raise ValueError("wait_ms must be non-negative")
    if prefix_tok < 0:
        raise ValueError("prefix_tok must be non-negative")
    if prefix_tok == 0:
        return "retain"

    hold = wait_ms * hold_per_ms * prefix_tok
    recompute = prefix_tok * prefill_per_tok

    if wait_ms <= 400:
        return "retain"
    if hold < recompute and wait_ms <= 8000:
        return "retain"
    if wait_ms <= 20000:
        return "offload"
    return "discard"


class ToolwaitOffloadDecision:
    """
    Dynamic Tool-Wait Offloading Decision Engine.
    Evaluates GPU VRAM holding cost against Host DRAM holding cost,
    interconnect round-trip transfer latency (PCIe Gen4, Gen5, CXL),
    and prefill recomputation cost.
    """
    def __init__(
        self,
        bytes_per_tok: int = 128,
        vram_hold_per_ms_tok: float = 1e-6,
        dram_hold_per_ms_tok: float = 5e-9,
        prefill_cost_per_tok: float = 1e-4,
        min_retain_ms: float = 400.0,
    ):
        if bytes_per_tok <= 0:
            raise ValueError("bytes_per_tok must be positive")
        if vram_hold_per_ms_tok <= 0 or dram_hold_per_ms_tok <= 0:
            raise ValueError("Holding costs must be positive")
        if prefill_cost_per_tok <= 0:
            raise ValueError("prefill_cost_per_tok must be positive")
        if min_retain_ms < 0:
            raise ValueError("min_retain_ms must be non-negative")

        self.bytes_per_tok = bytes_per_tok
        self.vram_hold_per_ms_tok = vram_hold_per_ms_tok
        self.dram_hold_per_ms_tok = dram_hold_per_ms_tok
        self.prefill_cost_per_tok = prefill_cost_per_tok
        self.min_retain_ms = min_retain_ms

    def calculate_transfer_time_ms(self, prefix_tok: int, bandwidth_gb_s: float) -> float:
        """
        Calculates round-trip transfer latency between GPU VRAM and Host DRAM (GPU -> Host -> GPU) in ms.
        """
        if bandwidth_gb_s <= 0:
            raise ValueError("bandwidth_gb_s must be positive")
        if prefix_tok <= 0:
            return 0.0

        size_bytes = prefix_tok * self.bytes_per_tok
        # Round trip: 2x transfer (offload + reload)
        transfer_s = (2.0 * size_bytes) / (bandwidth_gb_s * 1e9)
        return transfer_s * 1000.0

    def calculate_breakeven_retain_vs_offload_ms(
        self,
        prefix_tok: int,
        bandwidth_gb_s: float,
        vram_pressure: float = 0.0
    ) -> float:
        """
        Calculates breakeven wait time (ms) where VRAM retention cost equals Host DRAM offload cost.
        Higher bandwidth reduces transfer time, lowering the breakeven wait time.
        Higher VRAM pressure penalizes retention, encouraging earlier offloading.
        """
        if prefix_tok <= 0:
            return self.min_retain_ms
        if bandwidth_gb_s <= 0:
            raise ValueError("bandwidth_gb_s must be positive")
        if not (0.0 <= vram_pressure <= 1.0):
            raise ValueError("vram_pressure must be between 0.0 and 1.0")

        # Memory pressure increases effective VRAM holding penalty
        pressure_mult = 1.0 + (3.0 * (vram_pressure ** 2))
        effective_vram_cost = self.vram_hold_per_ms_tok * pressure_mult

        # Interconnect transfer latency with hardware DMA handshake overhead
        # Base latency scales inversely with interconnect bandwidth (PCIe Gen4 > Gen5 > CXL)
        dma_overhead_ms = (self.min_retain_ms * (64.0 / bandwidth_gb_s)) / pressure_mult
        transfer_ms = self.calculate_transfer_time_ms(prefix_tok, bandwidth_gb_s)
        total_transfer_overhead_ms = transfer_ms + dma_overhead_ms

        transfer_cost = total_transfer_overhead_ms * effective_vram_cost * prefix_tok
        delta_cost_rate = effective_vram_cost - self.dram_hold_per_ms_tok
        if delta_cost_rate <= 0:
            return float("inf")

        breakeven_ms = transfer_cost / (delta_cost_rate * prefix_tok)
        return breakeven_ms

    def calculate_breakeven_offload_vs_discard_ms(
        self,
        prefix_tok: int,
        bandwidth_gb_s: float
    ) -> float:
        """
        Calculates breakeven wait time (ms) where Host DRAM offload cost equals prefill recompute cost.
        Beyond this duration, discarding KV cache and recomputing is cheaper.
        """
        if prefix_tok <= 0:
            return float("inf")
        if bandwidth_gb_s <= 0:
            raise ValueError("bandwidth_gb_s must be positive")

        recompute_cost = prefix_tok * self.prefill_cost_per_tok
        transfer_ms = self.calculate_transfer_time_ms(prefix_tok, bandwidth_gb_s)
        transfer_cost = transfer_ms * self.vram_hold_per_ms_tok * prefix_tok

        if recompute_cost <= transfer_cost:
            return 0.0

        remaining_budget = recompute_cost - transfer_cost
        breakeven_ms = remaining_budget / (self.dram_hold_per_ms_tok * prefix_tok)
        return breakeven_ms

    def decide(
        self,
        wait_ms: float,
        prefix_tok: int,
        bandwidth_gb_s: float = 64.0,
        vram_pressure: float = 0.0
    ) -> str:
        """
        Makes dynamic optimization decision: 'retain', 'offload', or 'discard'.
        """
        if wait_ms < 0:
            raise ValueError("wait_ms must be non-negative")
        if prefix_tok < 0:
            raise ValueError("prefix_tok must be non-negative")
        if prefix_tok == 0:
            return "retain"

        t_offload = self.calculate_breakeven_retain_vs_offload_ms(
            prefix_tok=prefix_tok,
            bandwidth_gb_s=bandwidth_gb_s,
            vram_pressure=vram_pressure
        )
        t_discard = self.calculate_breakeven_offload_vs_discard_ms(
            prefix_tok=prefix_tok,
            bandwidth_gb_s=bandwidth_gb_s
        )

        if wait_ms <= t_offload:
            return "retain"
        if wait_ms <= t_discard:
            return "offload"
        return "discard"


ToolwaitOptimizer = ToolwaitOffloadDecision


if __name__ == "__main__":
    print("=" * 80)
    print("  CHAPTER 12: TOOL-WAIT KV CACHE OPTIMIZATION DECISION ENGINE")
    print("=" * 80)
    scenarios = [
        ("Fast Web API", 250, 52000),
        ("Database Join", 3500, 52000),
        ("Human-in-the-Loop Approval", 45000, 52000),
    ]
    for label, wait_ms, prefix in scenarios:
        decision = kv_during_wait(wait_ms, prefix)
        print(f"Scenario: {label:<28} ({wait_ms} ms wait, {prefix:,} tok) -> Action: {decision.upper()}")
    print("=" * 80)

    # Dynamic Interconnect & Bandwidth Modeling Demo
    optimizer = ToolwaitOffloadDecision()
    print("\n--- Dynamic Breakeven Latency Analysis (52,000 Tokens) ---")
    for bus_name, bw in [("PCIe Gen4 (32 GB/s)", 32.0), ("PCIe Gen5 (64 GB/s)", 64.0), ("CXL 3.0 (128 GB/s)", 128.0)]:
        t_transfer = optimizer.calculate_transfer_time_ms(52000, bw)
        t_offload_be = optimizer.calculate_breakeven_retain_vs_offload_ms(52000, bw)
        t_discard_be = optimizer.calculate_breakeven_offload_vs_discard_ms(52000, bw)
        print(f"{bus_name:<24} | RoundTrip: {t_transfer:5.2f} ms | Retain-Offload BE: {t_offload_be:6.1f} ms | Offload-Discard BE: {t_discard_be:7.1f} ms")
