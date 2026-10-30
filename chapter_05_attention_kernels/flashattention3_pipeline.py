"""
FlashAttention-3 Producer-Consumer Warp Specialization Simulator
Models asynchronous TMA (Tensor Memory Accelerator) loads, mbarrier synchronization,
and WGMMA consumer execution on NVIDIA Hopper architecture.
"""
from dataclasses import dataclass, field
import math
import time
from typing import Dict, List, Optional, Tuple


@dataclass
class MBarrierState:
    """
    Simulates hardware transaction barrier (cuda::barrier / mbarrier).
    Tracks transaction arrival tokens from TMA and phase parity for synchronization.
    """
    stage_id: int
    expected_bytes: int = 0
    arrived_bytes: int = 0
    phase: int = 0  # Phase parity bit (0 or 1)

    def arrive_and_expect_tx(self, tx_bytes: int) -> None:
        """Producer issues TMA and registers expected transaction byte count."""
        self.expected_bytes += tx_bytes

    def complete_tx(self, tx_bytes: int) -> None:
        """TMA hardware signals byte arrival directly into shared memory."""
        self.arrived_bytes += tx_bytes

    def is_ready(self) -> bool:
        """Returns True if all expected transaction bytes have arrived."""
        return self.expected_bytes > 0 and self.arrived_bytes >= self.expected_bytes

    def reset_for_next_phase(self) -> None:
        """Consumer releases stage and flips mbarrier phase bit."""
        self.expected_bytes = 0
        self.arrived_bytes = 0
        self.phase ^= 1


@dataclass
class FA3SimulationResult:
    """Detailed cycle metrics from the FlashAttention-3 pipeline simulation."""
    total_tiles: int
    pipeline_stages: int
    total_cycles: int
    overlap_efficiency: float
    effective_wgmma_tflops: float
    producer_stall_cycles: int
    consumer_stall_cycles: int
    elapsed_us: float
    producer_issue_cycles: List[int] = field(default_factory=list)
    tma_arrival_cycles: List[int] = field(default_factory=list)
    consumer_start_cycles: List[int] = field(default_factory=list)
    consumer_finish_cycles: List[int] = field(default_factory=list)


class FA3PipelineSimulator:
    """
    Hopper FlashAttention-3 Producer-Consumer Pipeline Simulator.

    Decouples threadblocks into:
    1. Producer Warp (Warp 0, 32 threads): Computes TMA descriptors and initiates
       asynchronous bulk transfers from GMEM to SMEM with zero register staging.
    2. Consumer Warpgroup (Warps 1-3, 96 threads): Issues asynchronous Warpgroup
       MMA (WGMMA) matrix multiplications and SIMD online softmax reductions.

    Tracks multi-stage circular SMEM buffers and mbarrier synchronization.
    """

    def __init__(
        self,
        num_stages: int = 4,
        tma_latency_cycles: int = 250,
        wgmma_latency_cycles: int = 240,
        tma_issue_cycles: int = 5,
        tile_bytes: int = 65536,  # 128x128 FP16 tile (64 KB)
        peak_wgmma_tflops: float = 750.0,
    ):
        if num_stages < 1:
            raise ValueError("num_stages must be at least 1")
        if tma_latency_cycles < 0 or wgmma_latency_cycles < 0:
            raise ValueError("Latencies must be non-negative")

        self.num_stages = num_stages
        self.tma_latency_cycles = tma_latency_cycles
        self.wgmma_latency_cycles = wgmma_latency_cycles
        self.tma_issue_cycles = tma_issue_cycles
        self.tile_bytes = tile_bytes
        self.peak_wgmma_tflops = peak_wgmma_tflops

        # Initialize hardware mbarriers for circular stages
        self.full_barriers = [MBarrierState(stage_id=i) for i in range(num_stages)]
        self.empty_barriers = [MBarrierState(stage_id=i) for i in range(num_stages)]

    def run(self, num_tiles: int) -> FA3SimulationResult:
        """
        Executes discrete-event pipeline simulation across num_tiles.
        Tracks producer issue cycles, TMA transfer arrivals, consumer GEMMs,
        and barrier stalls.
        """
        if num_tiles < 0:
            raise ValueError("num_tiles must be non-negative")

        t_wall_start = time.perf_counter()

        if num_tiles == 0:
            return FA3SimulationResult(
                total_tiles=0,
                pipeline_stages=self.num_stages,
                total_cycles=0,
                overlap_efficiency=1.0,
                effective_wgmma_tflops=0.0,
                producer_stall_cycles=0,
                consumer_stall_cycles=0,
                elapsed_us=0.0,
            )

        producer_issue = [0] * num_tiles
        tma_arrival = [0] * num_tiles
        consumer_start = [0] * num_tiles
        consumer_finish = [0] * num_tiles

        producer_stall_cycles = 0
        consumer_stall_cycles = 0

        # Simulate concurrent producer and consumer timelines
        for i in range(num_tiles):
            stage = i % self.num_stages

            # Earliest cycle producer can issue tile i
            earliest_producer = 0 if i == 0 else producer_issue[i - 1] + self.tma_issue_cycles

            # Buffer fullness check: stage cannot be overwritten until Consumer frees it
            if i >= self.num_stages:
                stage_free_cycle = consumer_finish[i - self.num_stages]
                if stage_free_cycle > earliest_producer:
                    producer_stall_cycles += (stage_free_cycle - earliest_producer)
                    issue_cycle = stage_free_cycle
                else:
                    issue_cycle = earliest_producer
            else:
                issue_cycle = earliest_producer

            producer_issue[i] = issue_cycle
            # Hardware mbarrier tracks transaction
            self.full_barriers[stage].arrive_and_expect_tx(self.tile_bytes)
            self.full_barriers[stage].complete_tx(self.tile_bytes)
            tma_arrival[i] = issue_cycle + self.tma_latency_cycles

            # Earliest cycle consumer warpgroup can execute tile i
            earliest_consumer = 0 if i == 0 else consumer_finish[i - 1]
            data_ready_cycle = tma_arrival[i]

            # Stage synchronization check: consumer waits for TMA arrival on full_barrier
            if data_ready_cycle > earliest_consumer:
                consumer_stall_cycles += (data_ready_cycle - earliest_consumer)
                compute_start = data_ready_cycle
            else:
                compute_start = earliest_consumer

            consumer_start[i] = compute_start
            consumer_finish[i] = compute_start + self.wgmma_latency_cycles
            self.full_barriers[stage].reset_for_next_phase()

        total_cycles = consumer_finish[-1]
        compute_cycles = num_tiles * self.wgmma_latency_cycles

        # Hardware overlap efficiency: ratio of active compute cycles to total timeline cycles
        overlap_efficiency = compute_cycles / total_cycles if total_cycles > 0 else 1.0
        effective_wgmma_tflops = self.peak_wgmma_tflops * overlap_efficiency

        elapsed_us = (time.perf_counter() - t_wall_start) * 1e6

        return FA3SimulationResult(
            total_tiles=num_tiles,
            pipeline_stages=self.num_stages,
            total_cycles=total_cycles,
            overlap_efficiency=round(overlap_efficiency, 4),
            effective_wgmma_tflops=round(effective_wgmma_tflops, 2),
            producer_stall_cycles=producer_stall_cycles,
            consumer_stall_cycles=consumer_stall_cycles,
            elapsed_us=round(elapsed_us, 4),
            producer_issue_cycles=producer_issue,
            tma_arrival_cycles=tma_arrival,
            consumer_start_cycles=consumer_start,
            consumer_finish_cycles=consumer_finish,
        )


def simulate_fa3_pipeline(
    num_tiles: int = 16,
    num_stages: int = 4,
    tma_latency_cycles: int = 250,
    wgmma_latency_cycles: int = 240,
) -> Dict[str, float]:
    """
    Simulates Hopper FA-3 double-buffered and multi-stage asynchronous pipeline.
    Producer warps load next tile via TMA while Consumer warps compute WGMMA GEMMs.

    Returns dictionary matching benchmark interface:
    - Total_Tiles: Total tiles processed
    - Overlap_Efficiency: Ratio of overlapped execution to total wall clock cycles
    - Hopper_WGMMA_TFLOPS: Effective compute throughput achieved
    - Total_Cycles: Simulated timeline cycles
    - Producer_Stall_Cycles: Cycles stalled waiting for empty SMEM buffers
    - Consumer_Stall_Cycles: Cycles stalled waiting for TMA arrival
    """
    simulator = FA3PipelineSimulator(
        num_stages=num_stages,
        tma_latency_cycles=tma_latency_cycles,
        wgmma_latency_cycles=wgmma_latency_cycles,
    )
    result = simulator.run(num_tiles=num_tiles)
    return {
        "Total_Tiles": result.total_tiles,
        "Pipeline_Stages": result.pipeline_stages,
        "Total_Cycles": result.total_cycles,
        "Overlap_Efficiency": result.overlap_efficiency,
        "Hopper_WGMMA_TFLOPS": result.effective_wgmma_tflops,
        "Producer_Stall_Cycles": result.producer_stall_cycles,
        "Consumer_Stall_Cycles": result.consumer_stall_cycles,
        "Execution_Time_us": result.elapsed_us,
    }


if __name__ == "__main__":
    res = simulate_fa3_pipeline(32)
    print("=" * 80)
    print("  CHAPTER 5: FLASHATTENTION-3 WARP SPECIALIZATION PIPELINE SIMULATION")
    print("=" * 80)
    print(f"Total Tiles Processed: {res['Total_Tiles']}")
    print(f"Pipeline Stages: {res['Pipeline_Stages']}")
    print(f"Total Simulated Cycles: {res['Total_Cycles']}")
    print(f"Producer-Consumer Overlap Efficiency: {res['Overlap_Efficiency']*100:.1f}%")
    print(f"Effective Hopper WGMMA Throughput: {res['Hopper_WGMMA_TFLOPS']:.1f} TFLOPS")
    print(f"Producer Stall Cycles: {res['Producer_Stall_Cycles']}")
    print(f"Consumer Stall Cycles: {res['Consumer_Stall_Cycles']}")
    print("=" * 80)
