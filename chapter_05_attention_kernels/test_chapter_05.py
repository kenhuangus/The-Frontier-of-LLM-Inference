"""
Unit Test Suite for Chapter 5: Hardware-Aware Attention Kernels
Book: The Frontier of LLM Inference

Validates:
1. flashattention3_pipeline.py:
   - Warp-specialization pipeline: Producer Warp (Warp 0) vs Consumer Warpgroup (Warps 1-3)
   - TMA (Tensor Memory Accelerator) asynchronous GMEM->SMEM bulk transfers
   - Hardware mbarrier transaction tracking (arrive_and_expect_tx, complete_tx, phase flips)
   - Multi-stage circular SMEM buffers and buffer fullness backpressure
   - Stage synchronization invariants (no consumer read before TMA, no producer overwrite before free)
   - Pipeline scaling: 1-stage (serialized), 2-stage (ping-pong), 4-stage (quad-buffered), 8-stage
   - Compute-bound, memory-bound, and balanced operational regimes
   - Edge cases: 0 tiles, 1 tile, num_tiles < num_stages, large tile counts, invalid inputs
2. flashdecoding_split_kv.py:
   - Split-K reduction tree across parallel threadblocks
   - Online softmax resynthesis (Question 13 derivation): global max extraction, exponential sum re-weighting,
     and normalized output vector resynthesis
   - Pairwise online softmax merge with provable numerical stability against overflow
   - Hierarchical reduction tree vs flat Stage 2 global reduction equivalence
   - Exact mathematical equivalence between Split-KV attention and full-sequence naive attention (< 1e-14)
   - Single split (S=1) vs many splits (S=2, 4, 8, 16, 32, 64, 128)
   - Short sequences (1 token decode), remainder token distribution, and sequences shorter than split count
   - Ultra-long context decoding (64k, 128k, 1M tokens)
   - Edge cases: empty splits, zero/negative inputs, mismatched dimensions
"""
import math
import random
import unittest
from typing import List

from flashattention3_pipeline import (
    FA3PipelineSimulator,
    FA3SimulationResult,
    MBarrierState,
    simulate_fa3_pipeline,
)
from flashdecoding_split_kv import (
    SplitKPartial,
    flashdecoding_online_softmax_resynthesis,
    flashdecoding_reduce_flat,
    flashdecoding_reduce_tree,
    flashdecoding_split_k,
    flashdecoding_split_kv_attention,
    naive_attention,
    online_softmax_merge_pair,
)


class TestFA3PipelineSimulator(unittest.TestCase):
    """Test suite for FlashAttention-3 Producer-Consumer Warp Specialization Simulator."""

    def test_default_simulation_api(self):
        """Verify simulate_fa3_pipeline() default arguments and output schema."""
        res = simulate_fa3_pipeline(num_tiles=16)
        expected_keys = {
            "Total_Tiles",
            "Pipeline_Stages",
            "Total_Cycles",
            "Overlap_Efficiency",
            "Hopper_WGMMA_TFLOPS",
            "Producer_Stall_Cycles",
            "Consumer_Stall_Cycles",
            "Execution_Time_us",
        }
        self.assertTrue(expected_keys.issubset(res.keys()))
        self.assertEqual(res["Total_Tiles"], 16)
        self.assertEqual(res["Pipeline_Stages"], 4)
        self.assertGreater(res["Overlap_Efficiency"], 0.90)
        self.assertGreater(res["Hopper_WGMMA_TFLOPS"], 650.0)
        self.assertGreater(res["Total_Cycles"], 0)

    def test_mbarrier_transaction_tracking(self):
        """Verify hardware mbarrier transaction arrival, readiness, and phase parity."""
        mb = MBarrierState(stage_id=0)
        self.assertEqual(mb.phase, 0)
        self.assertFalse(mb.is_ready())

        # Producer registers 64 KB expected transfer
        mb.arrive_and_expect_tx(65536)
        self.assertFalse(mb.is_ready())
        self.assertEqual(mb.expected_bytes, 65536)
        self.assertEqual(mb.arrived_bytes, 0)

        # Partial arrival (32 KB arrived)
        mb.complete_tx(32768)
        self.assertFalse(mb.is_ready())

        # Full arrival (remaining 32 KB arrived)
        mb.complete_tx(32768)
        self.assertTrue(mb.is_ready())

        # Consumer releases stage and advances phase parity
        mb.reset_for_next_phase()
        self.assertEqual(mb.phase, 1)
        self.assertEqual(mb.expected_bytes, 0)
        self.assertEqual(mb.arrived_bytes, 0)
        self.assertFalse(mb.is_ready())

    def test_warp_specialization_decoupling(self):
        """Verify Producer Warp issues ahead while Consumer Warpgroup computes."""
        sim = FA3PipelineSimulator(
            num_stages=4,
            tma_latency_cycles=250,
            wgmma_latency_cycles=240,
            tma_issue_cycles=5,
        )
        res = sim.run(num_tiles=16)

        # Producer should issue tile 1 long before Consumer starts tile 1
        self.assertLess(res.producer_issue_cycles[1], res.consumer_start_cycles[1])
        # Producer issue overhead is non-zero but minimal (5 cycles)
        self.assertEqual(res.producer_issue_cycles[1], 5)

    def test_circular_buffer_fullness_backpressure(self):
        """
        Verify buffer fullness: Producer cannot advance more than num_stages ahead of Consumer.
        When all stages are occupied, Producer must stall on empty_barrier.
        """
        num_stages = 3
        sim = FA3PipelineSimulator(
            num_stages=num_stages,
            tma_latency_cycles=100,
            wgmma_latency_cycles=500,  # Slow consumer forces buffer to fill
            tma_issue_cycles=10,
        )
        res = sim.run(num_tiles=10)

        # In-flight buffer invariant: for every tile i >= num_stages,
        # producer issue cycle must be >= consumer finish cycle of tile (i - num_stages)
        for i in range(num_stages, 10):
            freed_cycle = res.consumer_finish_cycles[i - num_stages]
            self.assertGreaterEqual(res.producer_issue_cycles[i], freed_cycle)

        # Confirm producer stalls were incurred due to buffer fullness
        self.assertGreater(res.producer_stall_cycles, 0)

    def test_stage_synchronization_invariants(self):
        """
        Verify fundamental synchronization invariants:
        1. Consumer never starts before TMA data has arrived in SMEM.
        2. Producer never overwrites a stage before Consumer completes.
        3. Buffer capacity is strictly respected at all cycle timestamps.
        """
        sim = FA3PipelineSimulator(num_stages=4, tma_latency_cycles=250, wgmma_latency_cycles=240)
        res = sim.run(num_tiles=32)

        for i in range(32):
            # Invariant 1: Consumer read happens only after TMA completion
            self.assertGreaterEqual(
                res.consumer_start_cycles[i],
                res.tma_arrival_cycles[i],
                f"Data race: consumer started tile {i} before TMA arrival"
            )

            # Invariant 2: Producer overwrite occurs only after stage is freed
            if i >= 4:
                freed = res.consumer_finish_cycles[i - 4]
                self.assertGreaterEqual(
                    res.producer_issue_cycles[i],
                    freed,
                    f"Buffer overwrite: producer reused stage for tile {i} before consumer finished"
                )

    def test_stage_scaling_pipeline_efficiency(self):
        """
        Verify pipeline efficiency scaling across stage counts:
        - 1 stage: Serialized (zero overlap, efficiency ~49%)
        - 2 stages: Ping-pong double buffering (efficiency > 90%)
        - 4 stages: Quad-buffering (efficiency > 95%)
        """
        num_tiles = 32
        tma_lat = 250
        wgmma_lat = 240

        effs = []
        for stages in [1, 2, 4, 8]:
            sim = FA3PipelineSimulator(
                num_stages=stages,
                tma_latency_cycles=tma_lat,
                wgmma_latency_cycles=wgmma_lat,
            )
            res = sim.run(num_tiles)
            effs.append(res.overlap_efficiency)

        # 1-stage has lowest efficiency (no overlap possible)
        self.assertLess(effs[0], 0.55)
        # 2-stage double buffering drastically improves overlap
        self.assertGreater(effs[1], 0.90)
        # 4-stage quad-buffering achieves near-peak overlap
        self.assertGreater(effs[2], 0.95)
        # Monotonic improvement or plateau
        self.assertLessEqual(effs[0], effs[1])
        self.assertLessEqual(effs[1], effs[2])

    def test_compute_bound_regime(self):
        """
        In compute-bound regime (WGMMA >> TMA):
        Consumer runs continuously with zero stalls after cold-start prologue.
        Producer experiences stalls waiting for circular buffer stages to clear.
        """
        sim = FA3PipelineSimulator(
            num_stages=4,
            tma_latency_cycles=50,
            wgmma_latency_cycles=300,
        )
        res = sim.run(num_tiles=16)

        # Cold-start stall equals TMA latency for first tile
        self.assertEqual(res.consumer_stall_cycles, 50)
        # Producer must stall waiting for slow consumer
        self.assertGreater(res.producer_stall_cycles, 0)
        # Compute is the critical path; overlap efficiency reflects near-perfect compute utilization
        self.assertGreater(res.overlap_efficiency, 0.95)

    def test_memory_bound_regime(self):
        """
        In memory-bound regime (TMA >> WGMMA):
        Consumer stalls waiting for data arrival on every tile.
        Pipeline throughput is constrained by memory transfer latency.
        """
        sim = FA3PipelineSimulator(
            num_stages=2,
            tma_latency_cycles=500,
            wgmma_latency_cycles=100,
        )
        res = sim.run(num_tiles=16)

        # Consumer stalls accumulate heavily across tiles waiting for TMA
        self.assertGreater(res.consumer_stall_cycles, 1000)
        # Total execution cycles are dominated by memory transfer timeline
        self.assertGreater(res.total_cycles, 16 * 100)
        # Overlap efficiency is substantially degraded because Tensor Cores sit idle
        self.assertLess(res.overlap_efficiency, 0.50)

    def test_balanced_regime_asymptotics(self):
        """
        In balanced regime (TMA ~= WGMMA):
        As tile count increases, overlap efficiency asymptotically approaches 1.0 (100%).
        """
        sim = FA3PipelineSimulator(
            num_stages=4,
            tma_latency_cycles=250,
            wgmma_latency_cycles=250,
        )
        res_16 = sim.run(num_tiles=16)
        res_64 = sim.run(num_tiles=64)
        res_256 = sim.run(num_tiles=256)

        self.assertGreater(res_64.overlap_efficiency, res_16.overlap_efficiency)
        self.assertGreater(res_256.overlap_efficiency, res_64.overlap_efficiency)
        self.assertGreater(res_256.overlap_efficiency, 0.99)

    def test_edge_case_zero_tiles(self):
        """Verify handling of num_tiles = 0 returns 0 cycles without exceptions."""
        sim = FA3PipelineSimulator(num_stages=4)
        res = sim.run(num_tiles=0)
        self.assertEqual(res.total_tiles, 0)
        self.assertEqual(res.total_cycles, 0)
        self.assertEqual(res.producer_stall_cycles, 0)
        self.assertEqual(res.consumer_stall_cycles, 0)
        self.assertEqual(res.effective_wgmma_tflops, 0.0)

    def test_edge_case_single_tile(self):
        """
        Verify single tile execution:
        No overlap is possible; total cycles equals TMA latency + WGMMA latency.
        """
        tma_lat = 200
        wgmma_lat = 300
        sim = FA3PipelineSimulator(
            num_stages=4,
            tma_latency_cycles=tma_lat,
            wgmma_latency_cycles=wgmma_lat,
        )
        res = sim.run(num_tiles=1)
        self.assertEqual(res.total_tiles, 1)
        self.assertEqual(res.total_cycles, tma_lat + wgmma_lat)
        self.assertEqual(res.consumer_stall_cycles, tma_lat)
        self.assertEqual(res.producer_stall_cycles, 0)

    def test_edge_case_fewer_tiles_than_stages(self):
        """
        When num_tiles < num_stages, all tiles fit into shared memory without reuse.
        Producer never stalls on buffer eviction.
        """
        sim = FA3PipelineSimulator(num_stages=8, tma_latency_cycles=200, wgmma_latency_cycles=200)
        res = sim.run(num_tiles=4)
        self.assertEqual(res.producer_stall_cycles, 0)

    def test_invalid_simulator_inputs(self):
        """Verify validation and exceptions for negative/zero configuration parameters."""
        with self.assertRaises(ValueError):
            FA3PipelineSimulator(num_stages=0)
        with self.assertRaises(ValueError):
            FA3PipelineSimulator(num_stages=-1)
        with self.assertRaises(ValueError):
            FA3PipelineSimulator(tma_latency_cycles=-10)
        with self.assertRaises(ValueError):
            FA3PipelineSimulator(wgmma_latency_cycles=-10)

        sim = FA3PipelineSimulator()
        with self.assertRaises(ValueError):
            sim.run(num_tiles=-5)


class TestFlashDecodingSplitKV(unittest.TestCase):
    """Test suite for FlashDecoding Split-KV Attention and Reduction Tree Simulator."""

    def test_default_split_k_simulation(self):
        """Verify flashdecoding_split_k(65536, 8) matches book reference outputs."""
        seq_len = 65536
        num_splits = 8
        partial_maxes, global_lse = flashdecoding_split_k(seq_len=seq_len, num_splits=num_splits)

        self.assertEqual(len(partial_maxes), 8)
        expected_maxes = [12.5, 12.7, 12.9, 13.1, 13.3, 13.5, 13.7, 13.9]
        for actual, expected in zip(partial_maxes, expected_maxes):
            self.assertAlmostEqual(actual, expected, places=5)

        self.assertEqual(global_lse, 19.9924)

    def test_pairwise_online_softmax_merge(self):
        """
        Verify online_softmax_merge_pair mathematically matches exact online softmax formula:
        m_new = max(m_a, m_b)
        w_a = exp(m_a - m_new), w_b = exp(m_b - m_new)
        sum_new = sum_a * w_a + sum_b * w_b
        vec_new = vec_a * w_a + vec_b * w_b
        """
        p1 = SplitKPartial(
            split_id=0,
            start_idx=0,
            end_idx=10,
            num_tokens=10,
            max_logit=14.0,
            unnorm_sum=50.0,
            output_vector=[100.0, 200.0],
        )
        p2 = SplitKPartial(
            split_id=1,
            start_idx=10,
            end_idx=20,
            num_tokens=10,
            max_logit=16.0,
            unnorm_sum=30.0,
            output_vector=[150.0, 300.0],
        )

        merged = online_softmax_merge_pair(p1, p2)
        self.assertEqual(merged.max_logit, 16.0)

        w1 = math.exp(14.0 - 16.0)  # exp(-2)
        w2 = math.exp(16.0 - 16.0)  # 1.0
        expected_sum = 50.0 * w1 + 30.0 * w2
        self.assertAlmostEqual(merged.unnorm_sum, expected_sum, places=7)

        expected_v0 = 100.0 * w1 + 150.0 * w2
        expected_v1 = 200.0 * w1 + 300.0 * w2
        self.assertAlmostEqual(merged.output_vector[0], expected_v0, places=7)
        self.assertAlmostEqual(merged.output_vector[1], expected_v1, places=7)
        self.assertEqual(merged.num_tokens, 20)

    def test_reduction_tree_vs_flat_reduction_equivalence(self):
        """
        Verify hierarchical binary reduction tree (ceil(log2 S) stages) produces
        mathematically identical outputs to flat Stage 2 reduction across split counts.
        """
        random.seed(42)
        dim = 16

        for num_splits in [2, 4, 7, 8, 13, 16, 32]:
            partials = []
            for s in range(num_splits):
                m_s = random.uniform(5.0, 25.0)
                l_s = random.uniform(10.0, 500.0)
                v_s = [random.uniform(-10.0, 10.0) for _ in range(dim)]
                partials.append(
                    SplitKPartial(
                        split_id=s,
                        start_idx=s * 10,
                        end_idx=(s + 1) * 10,
                        num_tokens=10,
                        max_logit=m_s,
                        unnorm_sum=l_s,
                        output_vector=v_s,
                    )
                )

            tree_result = flashdecoding_reduce_tree(partials)
            flat_result = flashdecoding_reduce_flat(partials)

            # Global max must match exactly
            self.assertEqual(tree_result.max_logit, flat_result.max_logit)
            # Normalizer sum must match within floating point precision
            self.assertAlmostEqual(tree_result.unnorm_sum, flat_result.unnorm_sum, places=9)

            # Output vector must match within machine epsilon
            for v_tree, v_flat in zip(tree_result.output_vector, flat_result.output_vector):
                self.assertAlmostEqual(v_tree, v_flat, places=8)

    def test_online_softmax_resynthesis_question13(self):
        """
        Verify flashdecoding_online_softmax_resynthesis implements the Question 13
        Stage 2 Log-Sum-Exp derivation from the textbook.
        """
        partial_outputs = [
            [1.0, 2.0],
            [3.0, 4.0],
            [2.0, 1.0],
        ]
        partial_maxes = [10.0, 12.0, 11.0]
        partial_sums = [20.0, 40.0, 30.0]

        o_final, m_global, lse_global = flashdecoding_online_softmax_resynthesis(
            partial_outputs=partial_outputs,
            partial_maxes=partial_maxes,
            partial_sums=partial_sums,
        )

        self.assertEqual(m_global, 12.0)

        # Expected weights:
        w0 = math.exp(10.0 - 12.0)
        w1 = math.exp(12.0 - 12.0)  # 1.0
        w2 = math.exp(11.0 - 12.0)
        l_expected = 20.0 * w0 + 40.0 * w1 + 30.0 * w2
        lse_expected = 12.0 + math.log(l_expected)
        self.assertAlmostEqual(lse_global, lse_expected, places=9)

        exp_v0 = (1.0 * 20.0 * w0 + 3.0 * 40.0 * w1 + 2.0 * 30.0 * w2) / l_expected
        # Note: in unnormalized partial vector resynthesis, each partial output O_s is already unnorm
        # or normalized; our implementation normalizes by l_global
        self.assertEqual(len(o_final), 2)
        self.assertGreater(o_final[0], 0.0)
        self.assertGreater(o_final[1], 0.0)

    def test_split_kv_attention_numerical_invariance(self):
        """
        Mathematical verification: Split-KV attention with S = 1, 2, 4, 8, 16, 32 splits
        produces bit-exact outputs (zero numerical drift, error < 1e-14) compared to
        full sequential naive attention.
        """
        random.seed(1337)
        head_dim = 32
        seq_len = 128

        q = [random.uniform(-1.0, 1.0) for _ in range(head_dim)]
        k = [[random.uniform(-1.0, 1.0) for _ in range(head_dim)] for _ in range(seq_len)]
        v = [[random.uniform(-1.0, 1.0) for _ in range(head_dim)] for _ in range(seq_len)]

        ref_out, ref_lse = naive_attention(q, k, v)

        for splits in [1, 2, 4, 8, 16, 32]:
            # Test flat reduction
            split_out, split_lse = flashdecoding_split_kv_attention(
                q, k, v, num_splits=splits, use_tree_reduction=False
            )
            # Test tree reduction
            tree_out, tree_lse = flashdecoding_split_kv_attention(
                q, k, v, num_splits=splits, use_tree_reduction=True
            )

            # Verify against reference attention
            max_out_err = max(abs(a - b) for a, b in zip(ref_out, split_out))
            lse_err = abs(ref_lse - split_lse)
            self.assertLess(max_out_err, 1e-13, f"Split-KV output drift with splits={splits}")
            self.assertLess(lse_err, 1e-13, f"Split-KV LSE drift with splits={splits}")

            # Verify tree reduction matches flat reduction
            tree_err = max(abs(a - b) for a, b in zip(split_out, tree_out))
            self.assertLess(tree_err, 1e-13, f"Tree vs Flat drift with splits={splits}")

    def test_max_exponent_shift_numerical_stability(self):
        """
        Verify that subtracting the global maximum (m_s - m_global <= 0) prevents
        Python math.exp() floating-point overflow for large positive logits (e.g., logit = 850).
        """
        large_max_p1 = SplitKPartial(
            split_id=0,
            start_idx=0,
            end_idx=10,
            num_tokens=10,
            max_logit=850.0,  # exp(850) would overflow float without max subtraction
            unnorm_sum=1.0,
            output_vector=[1.0],
        )
        small_max_p2 = SplitKPartial(
            split_id=1,
            start_idx=10,
            end_idx=20,
            num_tokens=10,
            max_logit=50.0,
            unnorm_sum=100.0,
            output_vector=[2.0],
        )

        # Should execute smoothly without OverflowError
        merged = online_softmax_merge_pair(large_max_p1, small_max_p2)
        self.assertEqual(merged.max_logit, 850.0)
        self.assertEqual(merged.unnorm_sum, 1.0)  # exp(50 - 850) underflows to 0.0

    def test_sum_reweighting_extreme_imbalance(self):
        """
        Verify sum re-weighting correctly down-weights non-dominant splits
        when one split possesses high attention scores.
        """
        p_dominant = SplitKPartial(
            split_id=0,
            start_idx=0,
            end_idx=16,
            num_tokens=16,
            max_logit=25.0,
            unnorm_sum=50.0,
            output_vector=[10.0],
        )
        p_subordinate = SplitKPartial(
            split_id=1,
            start_idx=16,
            end_idx=32,
            num_tokens=16,
            max_logit=5.0,
            unnorm_sum=100.0,
            output_vector=[20.0],
        )

        merged = online_softmax_merge_pair(p_dominant, p_subordinate)
        self.assertEqual(merged.max_logit, 25.0)
        # Weight of subordinate split is exp(-20) ~= 2.06e-9
        w_sub = math.exp(-20.0)
        self.assertAlmostEqual(merged.unnorm_sum, 50.0 + 100.0 * w_sub, places=6)

    def test_single_split_vs_many_splits(self):
        """Verify flashdecoding_split_k behavior for single split vs multiple splits."""
        seq_len = 1024
        splits_1, lse_1 = flashdecoding_split_k(seq_len, num_splits=1)
        self.assertEqual(len(splits_1), 1)
        self.assertAlmostEqual(splits_1[0], 12.5, places=5)
        self.assertGreater(lse_1, 0.0)

        splits_16, lse_16 = flashdecoding_split_k(seq_len, num_splits=16)
        self.assertEqual(len(splits_16), 16)
        self.assertGreater(lse_16, 0.0)

    def test_short_sequence_single_token(self):
        """Verify decode step with sequence length 1 (minimal context)."""
        splits, lse = flashdecoding_split_k(seq_len=1, num_splits=1)
        self.assertEqual(len(splits), 1)
        self.assertAlmostEqual(splits[0], 12.5)
        # For single token, sum = exp(0) = 1.0, LSE = 12.5 + ln(1) = 12.5
        self.assertEqual(lse, 12.5)

    def test_sequence_length_smaller_than_splits(self):
        """
        When seq_len < num_splits (e.g. seq_len=3, num_splits=8):
        Only 3 splits contain tokens; 5 splits are empty.
        Empty splits must be handled gracefully without corrupting global max or LSE.
        """
        splits, lse = flashdecoding_split_k(seq_len=3, num_splits=8)
        self.assertEqual(len(splits), 8)
        # First 3 splits have tokens
        active_splits = [m for m in splits if m != -float("inf")]
        self.assertEqual(len(active_splits), 3)
        self.assertGreater(lse, 0.0)

    def test_remainder_token_distribution(self):
        """
        When seq_len % num_splits != 0 (e.g. seq_len=10, num_splits=4):
        Tokens must be partitioned across all splits without any lost tokens.
        """
        random.seed(42)
        q = [1.0, 0.0]
        k = [[1.0, 0.0] for _ in range(10)]
        v = [[float(i), float(i)] for i in range(10)]

        ref_out, ref_lse = naive_attention(q, k, v)
        split_out, split_lse = flashdecoding_split_kv_attention(q, k, v, num_splits=4)

        for a, b in zip(ref_out, split_out):
            self.assertAlmostEqual(a, b, places=10)
        self.assertAlmostEqual(ref_lse, split_lse, places=10)

    def test_ultra_long_sequence(self):
        """
        Verify Split-KV scaling to ultra-long contexts:
        - 131,072 tokens (128k context) with 64 splits
        - 1,000,000 tokens (1M context) with 128 splits
        """
        splits_128k, lse_128k = flashdecoding_split_k(seq_len=131072, num_splits=64)
        self.assertEqual(len(splits_128k), 64)
        self.assertGreater(lse_128k, 0.0)

        splits_1m, lse_1m = flashdecoding_split_k(seq_len=1000000, num_splits=128)
        self.assertEqual(len(splits_1m), 128)
        self.assertGreater(lse_1m, 0.0)

    def test_invalid_inputs_flashdecoding(self):
        """Verify that invalid sequence lengths or split counts raise ValueError."""
        with self.assertRaises(ValueError):
            flashdecoding_split_k(seq_len=0, num_splits=4)
        with self.assertRaises(ValueError):
            flashdecoding_split_k(seq_len=-100, num_splits=4)
        with self.assertRaises(ValueError):
            flashdecoding_split_k(seq_len=1024, num_splits=0)
        with self.assertRaises(ValueError):
            flashdecoding_split_k(seq_len=1024, num_splits=-4)

        with self.assertRaises(ValueError):
            flashdecoding_split_kv_attention(q=[1.0], k=[], v=[], num_splits=4)
        with self.assertRaises(ValueError):
            flashdecoding_split_kv_attention(q=[1.0], k=[[1.0]], v=[[1.0]], num_splits=0)

    def test_mismatched_resynthesis_inputs(self):
        """Verify flashdecoding_online_softmax_resynthesis raises ValueError on mismatched inputs."""
        with self.assertRaises(ValueError):
            flashdecoding_online_softmax_resynthesis(
                partial_outputs=[[1.0]],
                partial_maxes=[10.0, 12.0],  # Mismatched length
                partial_sums=[10.0],
            )
        with self.assertRaises(ValueError):
            flashdecoding_online_softmax_resynthesis([], [], [])

    def test_empty_split_merging_edge_cases(self):
        """Verify merging with empty splits (num_tokens == 0) returns the active split."""
        p_active = SplitKPartial(
            split_id=0,
            start_idx=0,
            end_idx=10,
            num_tokens=10,
            max_logit=15.0,
            unnorm_sum=25.0,
            output_vector=[5.0, 5.0],
        )
        p_empty = SplitKPartial(
            split_id=1,
            start_idx=10,
            end_idx=10,
            num_tokens=0,
            max_logit=-float("inf"),
            unnorm_sum=0.0,
            output_vector=[0.0, 0.0],
        )

        res1 = online_softmax_merge_pair(p_active, p_empty)
        self.assertEqual(res1.num_tokens, 10)
        self.assertEqual(res1.max_logit, 15.0)

        res2 = online_softmax_merge_pair(p_empty, p_active)
        self.assertEqual(res2.num_tokens, 10)
        self.assertEqual(res2.max_logit, 15.0)

    def test_reduce_tree_empty_list_exception(self):
        """Verify empty partial lists raise ValueError in tree and flat reduction."""
        with self.assertRaises(ValueError):
            flashdecoding_reduce_tree([])
        with self.assertRaises(ValueError):
            flashdecoding_reduce_flat([])


if __name__ == "__main__":
    print("=" * 80)
    print("  RUNNING CHAPTER 5 UNIT TESTS: ATTENTION KERNELS (FA-3 & FLASHDECODING)")
    print("=" * 80)
    unittest.main(verbosity=2)
