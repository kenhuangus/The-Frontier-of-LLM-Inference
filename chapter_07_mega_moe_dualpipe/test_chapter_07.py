"""
Unit Test Suite for Chapter 7: Serving Mega-MoE at Scale (DeepSeek DualPipe & Expert Parallelism)
The Frontier of LLM Inference (2026 Edition)

Validates:
1. DeepSeek DualPipe Bidirectional 1F1B Schedule:
   - Pipeline bubble ratio formulas and speedup calculation
   - Chunk count variations (1, 2, 4, 8 chunks per microbatch)
   - Zero or odd chunk numbers (1, 3, 5 chunks, handling asymmetry)
   - Forward and backward overlapping stages (warmup, 1F1B steady state, cooldown)
   - Edge cases: P=1 (single stage zero bubble), P > M, invalid non-positive inputs
2. Mega-MoE Top-K Routing and Expert Load Balancer:
   - Auxiliary-loss-free routing across 256 routed experts (DeepSeek-V3 scale)
   - Top-k selection validity and uniqueness guarantees
   - Expert capacity factor limits (memory buffer bounding)
   - Token dropping and fallback to shared experts
   - Severe load skew detection and mitigation (all tokens routed to expert 0)
   - Shared expert unconditional dispatch
   - Edge cases: empty tokens, single token, invalid arguments
3. Full Chapter Integration:
   - Co-simulation of DeepSeek-V3 671B inference pipeline grid and MoE all-to-all dispatch
"""
import math
import unittest
from typing import Dict, List

from dualpipe_schedule import (
    calculate_dualpipe_bubble,
    generate_dualpipe_schedule,
    simulate_dualpipe_overlap
)
from expert_parallelism import (
    RoutingResult,
    calculate_expert_capacity,
    route_tokens_to_experts
)


class TestDualPipeBubbleCalculation(unittest.TestCase):
    """Tests for DeepSeek DualPipe bubble ratio formulas, reductions, and boundary cases."""

    def test_default_parameters(self):
        """Verify default pipeline simulation (8 microbatches, 4 stages, 2 chunks)."""
        res = calculate_dualpipe_bubble(num_microbatches=8, num_pipeline_stages=4, num_chunks=2)
        # Standard: (4 - 1) / 8 = 3/8 = 37.5%
        # DualPipe: (4 - 1) / (2 * 8) = 3/16 = 18.75%
        # Reduction: (37.5 - 18.75) / 37.5 = 50.0%
        self.assertEqual(res["Standard_1F1B_Bubble"], 37.5)
        self.assertEqual(res["DualPipe_Bubble"], 18.75)
        self.assertEqual(res["Bubble_Reduction_Pct"], 50.0)

    def test_chapter_example_grid(self):
        """Verify Chapter 7 reference grid: 16 microbatches, 8 stages."""
        res = calculate_dualpipe_bubble(num_microbatches=16, num_pipeline_stages=8, num_chunks=2)
        # Standard: 7 / 16 = 43.75%
        # DualPipe: 7 / 32 = 21.88%
        # Reduction: 50.0%
        self.assertEqual(res["Standard_1F1B_Bubble"], 43.75)
        self.assertEqual(res["DualPipe_Bubble"], 21.88)
        self.assertEqual(res["Bubble_Reduction_Pct"], 50.0)

    def test_chunk_count_variations(self):
        """Verify bubble ratios across chunk count variations (1, 2, 4, 8 chunks)."""
        stages = 8
        microbatches = 16

        # Chunk = 1 (Identical to Standard 1F1B)
        c1 = calculate_dualpipe_bubble(microbatches, stages, num_chunks=1)
        self.assertEqual(c1["Standard_1F1B_Bubble"], c1["DualPipe_Bubble"])
        self.assertEqual(c1["Bubble_Reduction_Pct"], 0.0)

        # Chunk = 2 (Standard DualPipe: 50% bubble reduction)
        c2 = calculate_dualpipe_bubble(microbatches, stages, num_chunks=2)
        self.assertEqual(c2["Bubble_Reduction_Pct"], 50.0)

        # Chunk = 4 (4 chunks per microbatch: 75% bubble reduction)
        c4 = calculate_dualpipe_bubble(microbatches, stages, num_chunks=4)
        expected_dual_c4 = round((7.0 / (4 * 16)) * 100.0, 2)
        self.assertEqual(c4["DualPipe_Bubble"], expected_dual_c4)
        self.assertEqual(c4["Bubble_Reduction_Pct"], 75.0)

        # Chunk = 8 (8 chunks per microbatch: 87.5% bubble reduction)
        c8 = calculate_dualpipe_bubble(microbatches, stages, num_chunks=8)
        self.assertEqual(c8["Bubble_Reduction_Pct"], 87.5)

    def test_odd_chunk_numbers(self):
        """Verify DualPipe handling of odd chunk numbers (1, 3, 5, 7 chunks)."""
        stages = 4
        microbatches = 12

        # 1 chunk: odd, 0% reduction
        res_1 = calculate_dualpipe_bubble(microbatches, stages, num_chunks=1)
        self.assertEqual(res_1["Bubble_Reduction_Pct"], 0.0)

        # 3 chunks: odd, reduction = (1 - 1/3) * 100 = 66.67%
        res_3 = calculate_dualpipe_bubble(microbatches, stages, num_chunks=3)
        self.assertEqual(res_3["Bubble_Reduction_Pct"], 66.67)

        # 5 chunks: odd, reduction = (1 - 1/5) * 100 = 80.0%
        res_5 = calculate_dualpipe_bubble(microbatches, stages, num_chunks=5)
        self.assertEqual(res_5["Bubble_Reduction_Pct"], 80.0)

        # 7 chunks: odd, reduction = (1 - 1/7) * 100 = 85.71%
        res_7 = calculate_dualpipe_bubble(microbatches, stages, num_chunks=7)
        self.assertEqual(res_7["Bubble_Reduction_Pct"], 85.71)

    def test_single_pipeline_stage_edge_case(self):
        """Verify single pipeline stage (P=1) produces zero bubble and no ZeroDivisionError."""
        res = calculate_dualpipe_bubble(num_microbatches=10, num_pipeline_stages=1, num_chunks=2)
        self.assertEqual(res["Standard_1F1B_Bubble"], 0.0)
        self.assertEqual(res["DualPipe_Bubble"], 0.0)
        self.assertEqual(res["Bubble_Reduction_Pct"], 0.0)

    def test_pipeline_stages_exceed_microbatches(self):
        """Verify P > M regime where pipeline bubble exceeds 100% idle time."""
        res = calculate_dualpipe_bubble(num_microbatches=4, num_pipeline_stages=16, num_chunks=2)
        # Standard: 15 / 4 = 375% bubble
        # DualPipe: 15 / 8 = 187.5% bubble
        self.assertEqual(res["Standard_1F1B_Bubble"], 375.0)
        self.assertEqual(res["DualPipe_Bubble"], 187.5)
        self.assertEqual(res["Bubble_Reduction_Pct"], 50.0)

    def test_large_pipeline_grid(self):
        """Verify large-scale pipeline grid (64 stages, 128 microbatches)."""
        res = calculate_dualpipe_bubble(num_microbatches=128, num_pipeline_stages=64, num_chunks=2)
        self.assertEqual(res["Standard_1F1B_Bubble"], 49.22)
        self.assertEqual(res["DualPipe_Bubble"], 24.61)
        self.assertEqual(res["Bubble_Reduction_Pct"], 50.0)

    def test_zero_and_negative_input_validation(self):
        """Ensure invalid microbatch, stage, or chunk counts raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_dualpipe_bubble(num_microbatches=0, num_pipeline_stages=4)
        with self.assertRaises(ValueError):
            calculate_dualpipe_bubble(num_microbatches=-8, num_pipeline_stages=4)
        with self.assertRaises(ValueError):
            calculate_dualpipe_bubble(num_microbatches=8, num_pipeline_stages=0)
        with self.assertRaises(ValueError):
            calculate_dualpipe_bubble(num_microbatches=8, num_pipeline_stages=-4)
        with self.assertRaises(ValueError):
            calculate_dualpipe_bubble(num_microbatches=8, num_pipeline_stages=4, num_chunks=0)
        with self.assertRaises(ValueError):
            calculate_dualpipe_bubble(num_microbatches=8, num_pipeline_stages=4, num_chunks=-2)


class TestDualPipeScheduleGeneration(unittest.TestCase):
    """Tests for forward and backward overlapping stage schedules in DualPipe."""

    def test_schedule_structure_and_counts(self):
        """Verify warmup, steady-state overlap, and cooldown step calculations."""
        sched = generate_dualpipe_schedule(num_microbatches=8, num_pipeline_stages=4, num_chunks=2)
        
        self.assertEqual(sched["Pipeline_Stages"], 4)
        self.assertEqual(sched["Microbatches"], 8)
        self.assertEqual(sched["Chunks_Per_Microbatch"], 2)
        self.assertFalse(sched["Is_Odd_Chunks"])
        
        # Warmup steps = P - 1 = 3
        # Cooldown steps = P - 1 = 3
        # Total chunks = 8 * 2 = 16
        # Steady steps = 16 - 3 = 13
        # Total steps = 3 + 13 + 3 = 19
        self.assertEqual(sched["Warmup_Steps"], 3)
        self.assertEqual(sched["Steady_Steps"], 13)
        self.assertEqual(sched["Cooldown_Steps"], 3)
        self.assertEqual(sched["Total_Steps"], 19)
        self.assertEqual(sched["Overlapped_Steps"], 13)
        
        expected_overlap_ratio = round((13 / 19) * 100.0, 2)
        self.assertEqual(sched["Overlap_Ratio_Pct"], expected_overlap_ratio)

    def test_forward_and_backward_stage_separation(self):
        """Verify phases strictly partition into forward warmup, overlapped steady state, and backward cooldown."""
        sched = generate_dualpipe_schedule(num_microbatches=6, num_pipeline_stages=4, num_chunks=2)
        timeline = sched["Stage_Schedules"][0]

        # 1. Warmup steps: only forward operations
        for step in timeline[:sched["Warmup_Steps"]]:
            self.assertEqual(step["phase"], "warmup")
            self.assertIsNotNone(step["forward_op"])
            self.assertIsNone(step["backward_op"])
            self.assertFalse(step["overlapped"])

        # 2. Steady steps: concurrent forward and backward
        steady_start = sched["Warmup_Steps"]
        steady_end = steady_start + sched["Steady_Steps"]
        for step in timeline[steady_start:steady_end]:
            self.assertEqual(step["phase"], "steady_state")
            self.assertIsNotNone(step["forward_op"])
            self.assertIsNotNone(step["backward_op"])
            self.assertTrue(step["overlapped"])

        # 3. Cooldown steps: only backward operations
        for step in timeline[steady_end:]:
            self.assertEqual(step["phase"], "cooldown")
            self.assertIsNone(step["forward_op"])
            self.assertIsNotNone(step["backward_op"])
            self.assertFalse(step["overlapped"])

    def test_odd_chunk_schedule_generation(self):
        """Verify schedule generation handles odd chunk counts (e.g. 3 chunks)."""
        sched = generate_dualpipe_schedule(num_microbatches=4, num_pipeline_stages=3, num_chunks=3)
        self.assertTrue(sched["Is_Odd_Chunks"])
        self.assertEqual(sched["Chunks_Per_Microbatch"], 3)
        
        # Total chunks = 4 * 3 = 12
        # Warmup = 2, Steady = 10, Cooldown = 2, Total = 14
        self.assertEqual(sched["Warmup_Steps"], 2)
        self.assertEqual(sched["Steady_Steps"], 10)
        self.assertEqual(sched["Cooldown_Steps"], 2)
        self.assertEqual(sched["Total_Steps"], 14)

    def test_single_stage_schedule(self):
        """Verify single pipeline stage schedule operates sequentially without pipeline bubbles."""
        sched = generate_dualpipe_schedule(num_microbatches=4, num_pipeline_stages=1, num_chunks=2)
        self.assertEqual(sched["Warmup_Steps"], 0)
        self.assertEqual(sched["Steady_Steps"], 0)
        self.assertEqual(sched["Cooldown_Steps"], 0)
        self.assertEqual(sched["Overlapped_Steps"], 0)
        # 4 microbatches * 2 chunks = 8 forward + 8 backward = 16 steps
        self.assertEqual(sched["Total_Steps"], 16)
        self.assertIn(0, sched["Stage_Schedules"])

    def test_simulate_dualpipe_overlap_composite_api(self):
        """Verify high-level simulate_dualpipe_overlap combines bubble metrics and schedule metadata."""
        res = simulate_dualpipe_overlap(num_microbatches=16, num_pipeline_stages=8, num_chunks=2)
        self.assertEqual(res["Microbatches"], 16)
        self.assertEqual(res["Pipeline_Stages"], 8)
        self.assertEqual(res["Standard_1F1B_Bubble"], 43.75)
        self.assertEqual(res["DualPipe_Bubble"], 21.88)
        self.assertEqual(res["Bubble_Reduction_Pct"], 50.0)
        self.assertEqual(res["Warmup_Steps"], 7)
        self.assertEqual(res["Cooldown_Steps"], 7)
        self.assertEqual(res["Steady_Steps"], 25)
        self.assertEqual(res["Total_Steps"], 39)
        self.assertEqual(res["Overlapped_Steps"], 25)
        self.assertAlmostEqual(res["Overlap_Ratio_Pct"], 64.10, places=2)


class TestExpertParallelismRouting(unittest.TestCase):
    """Tests for MoE token routing, top-k selection, and baseline load balancing."""

    def test_default_routing_deepseek_scale(self):
        """Verify DeepSeek-V3 scale routing (256 experts, top-8)."""
        tokens = [f"tok_{i}" for i in range(100)]
        res = route_tokens_to_experts(tokens, num_routed_experts=256, top_k=8, seed=42)
        
        self.assertEqual(len(res), 256)
        # Total assignments across all experts must equal 100 tokens * 8 = 800
        self.assertEqual(sum(res.values()), 800)
        self.assertEqual(res.dropped_tokens_count, 0)
        self.assertEqual(res.fallback_tokens_count, 0)

    def test_top_k_selection_uniqueness(self):
        """Verify each token routes to exactly top_k distinct experts without duplicate targets."""
        tokens = [f"tok_{i}" for i in range(50)]
        top_k = 6
        num_experts = 32
        res = route_tokens_to_experts(tokens, num_routed_experts=num_experts, top_k=top_k, seed=123)

        for tok, assigned_experts in res.token_assignments.items():
            self.assertEqual(len(assigned_experts), top_k)
            # Distinctness check: no duplicate expert IDs for the same token
            self.assertEqual(len(set(assigned_experts)), top_k)
            for exp in assigned_experts:
                self.assertTrue(0 <= exp < num_experts)

    def test_reproducibility_with_seed(self):
        """Verify identical seed yields bitwise identical routing assignments."""
        tokens = [f"tok_{i}" for i in range(40)]
        res1 = route_tokens_to_experts(tokens, num_routed_experts=16, top_k=4, seed=999)
        res2 = route_tokens_to_experts(tokens, num_routed_experts=16, top_k=4, seed=999)
        self.assertEqual(dict(res1), dict(res2))
        self.assertEqual(res1.token_assignments, res2.token_assignments)


class TestExpertCapacityFactorAndDropping(unittest.TestCase):
    """Tests for expert capacity factor bounding, buffer saturation, and token dropping."""

    def test_calculate_expert_capacity_formula(self):
        """Verify analytical formula: ceil((tokens * top_k / experts) * capacity_factor)."""
        # 1000 tokens, 10 experts, top_k=1, CF=1.0 -> 100
        cap_1 = calculate_expert_capacity(1000, 10, 1, 1.0)
        self.assertEqual(cap_1, 100)

        # 1000 tokens, 64 experts, top_k=4, CF=1.0 -> ceil(4000 / 64) = 63
        cap_2 = calculate_expert_capacity(1000, 64, 4, 1.0)
        self.assertEqual(cap_2, 63)

        # CF = 1.25 -> ceil(62.5 * 1.25) = ceil(78.125) = 79
        cap_3 = calculate_expert_capacity(1000, 64, 4, 1.25)
        self.assertEqual(cap_3, 79)

        # Minimum clamp to 1
        cap_min = calculate_expert_capacity(1, 100, 1, 0.1)
        self.assertEqual(cap_min, 1)

    def test_capacity_factor_strict_enforcement(self):
        """Verify no expert receives more tokens than the calculated capacity limit."""
        tokens = [f"tok_{i}" for i in range(500)]
        num_experts = 16
        top_k = 2
        cf = 1.1

        res = route_tokens_to_experts(
            tokens=tokens,
            num_routed_experts=num_experts,
            top_k=top_k,
            capacity_factor=cf,
            drop_tokens=True,
            seed=42
        )

        expected_cap = calculate_expert_capacity(500, num_experts, top_k, cf)
        self.assertEqual(res.capacity_limit, expected_cap)

        for exp, count in res.items():
            self.assertLessEqual(count, expected_cap)

    def test_token_dropping_accounting_conservation(self):
        """Verify total requested assignments = accepted assignments + dropped assignments."""
        tokens = [f"tok_{i}" for i in range(200)]
        top_k = 4
        total_requested = len(tokens) * top_k

        res = route_tokens_to_experts(
            tokens=tokens,
            num_routed_experts=8,
            top_k=top_k,
            capacity_factor=0.8,  # Constrained capacity guarantees dropping
            drop_tokens=True,
            seed=7
        )

        total_accepted = sum(res.values())
        self.assertEqual(total_accepted + res.dropped_tokens_count, total_requested)
        self.assertEqual(len(res.dropped_assignments), res.dropped_tokens_count)
        self.assertGreater(res.dropped_tokens_count, 0)

    def test_capacity_factor_headroom_reduces_dropping(self):
        """Verify increasing capacity factor strictly decreases dropped tokens."""
        tokens = [f"tok_{i}" for i in range(300)]
        kwargs = dict(num_routed_experts=16, top_k=2, drop_tokens=True, seed=42)

        res_cf08 = route_tokens_to_experts(tokens, capacity_factor=0.8, **kwargs)
        res_cf10 = route_tokens_to_experts(tokens, capacity_factor=1.0, **kwargs)
        res_cf15 = route_tokens_to_experts(tokens, capacity_factor=1.5, **kwargs)
        res_cf20 = route_tokens_to_experts(tokens, capacity_factor=2.0, **kwargs)

        self.assertGreaterEqual(res_cf08.dropped_tokens_count, res_cf10.dropped_tokens_count)
        self.assertGreaterEqual(res_cf10.dropped_tokens_count, res_cf15.dropped_tokens_count)
        self.assertEqual(res_cf20.dropped_tokens_count, 0)


class TestSevereLoadSkewAndFallback(unittest.TestCase):
    """Tests for severe load skew (all tokens routed to expert 0) and fallback mechanisms."""

    def test_severe_skew_unlimited_capacity(self):
        """Verify that with unlimited capacity, severe skew overwhelms expert 0 and starves others."""
        tokens = [f"tok_{i}" for i in range(1000)]
        num_experts = 10
        top_k = 1

        res = route_tokens_to_experts(
            tokens=tokens,
            num_routed_experts=num_experts,
            top_k=top_k,
            capacity_factor=None,
            forced_expert=0
        )

        self.assertEqual(res[0], 1000)
        for exp in range(1, num_experts):
            self.assertEqual(res[exp], 0)
        self.assertEqual(res.dropped_tokens_count, 0)
        
        # High coefficient of variation under severe skew
        cv = res.get_coefficient_of_variation()
        self.assertGreater(cv, 2.5)

    def test_severe_skew_with_capacity_limit_drops_overflow(self):
        """Verify capacity factor prevents buffer overflow on skewed expert by dropping excess tokens."""
        tokens = [f"tok_{i}" for i in range(1000)]
        num_experts = 10
        top_k = 1
        cf = 1.0  # Capacity per expert = ceil(1000 * 1 / 10 * 1.0) = 100

        res = route_tokens_to_experts(
            tokens=tokens,
            num_routed_experts=num_experts,
            top_k=top_k,
            capacity_factor=cf,
            drop_tokens=True,
            forced_expert=0
        )

        self.assertEqual(res.capacity_limit, 100)
        self.assertEqual(res[0], 100)  # Capped strictly at 100
        for exp in range(1, num_experts):
            self.assertEqual(res[exp], 0)

        # 900 tokens dropped
        self.assertEqual(res.dropped_tokens_count, 900)
        self.assertEqual(sum(res.values()) + res.dropped_tokens_count, 1000)

    def test_fallback_to_shared_expert_under_skew(self):
        """Verify overflow tokens redirect to DeepSeek-V3 shared expert instead of dropping."""
        tokens = [f"tok_{i}" for i in range(1000)]
        num_experts = 10
        top_k = 1
        cf = 1.0

        res = route_tokens_to_experts(
            tokens=tokens,
            num_routed_experts=num_experts,
            top_k=top_k,
            capacity_factor=cf,
            fallback_to_shared=True,
            num_shared_experts=1,
            forced_expert=0
        )

        self.assertEqual(res[0], 100)
        self.assertEqual(res.dropped_tokens_count, 0)
        self.assertEqual(res.fallback_tokens_count, 900)
        # Shared expert 0 receives baseline 1000 tokens + 900 fallback tokens = 1900
        self.assertEqual(res.shared_expert_counts[0], 1900)

    def test_shared_experts_unconditional_dispatch(self):
        """Verify shared experts receive all tokens unconditionally regardless of routing."""
        tokens = [f"tok_{i}" for i in range(250)]
        res = route_tokens_to_experts(
            tokens=tokens,
            num_routed_experts=32,
            top_k=2,
            num_shared_experts=2,
            seed=42
        )

        self.assertEqual(len(res.shared_expert_counts), 2)
        self.assertEqual(res.shared_expert_counts[0], 250)
        self.assertEqual(res.shared_expert_counts[1], 250)


class TestExpertParallelismEdgeCasesAndValidation(unittest.TestCase):
    """Tests for edge cases, invalid parameter validations, and dict contracts."""

    def test_empty_tokens_edge_case(self):
        """Verify empty token input handles cleanly without errors."""
        res = route_tokens_to_experts([], num_routed_experts=16, top_k=2, capacity_factor=1.0)
        self.assertEqual(len(res), 16)
        self.assertEqual(sum(res.values()), 0)
        self.assertEqual(res.dropped_tokens_count, 0)
        self.assertEqual(res.capacity_limit, 0)
        self.assertEqual(res.get_coefficient_of_variation(), 0.0)

    def test_single_token_edge_case(self):
        """Verify single token routes accurately."""
        res = route_tokens_to_experts(["tok_single"], num_routed_experts=8, top_k=3, seed=1)
        self.assertEqual(sum(res.values()), 3)
        self.assertIn("tok_single", res.token_assignments)
        self.assertEqual(len(res.token_assignments["tok_single"]), 3)

    def test_invalid_num_routed_experts(self):
        """Verify non-positive expert count raises ValueError."""
        with self.assertRaises(ValueError):
            route_tokens_to_experts(["t1"], num_routed_experts=0)
        with self.assertRaises(ValueError):
            route_tokens_to_experts(["t1"], num_routed_experts=-8)

    def test_invalid_top_k(self):
        """Verify non-positive top_k raises ValueError."""
        with self.assertRaises(ValueError):
            route_tokens_to_experts(["t1"], num_routed_experts=8, top_k=0)
        with self.assertRaises(ValueError):
            route_tokens_to_experts(["t1"], num_routed_experts=8, top_k=-2)

    def test_top_k_exceeds_num_routed_experts(self):
        """Verify top_k > num_routed_experts raises descriptive ValueError."""
        with self.assertRaises(ValueError):
            route_tokens_to_experts(["t1"], num_routed_experts=4, top_k=5)

    def test_boundary_top_k_equals_num_routed_experts(self):
        """Verify top_k == num_routed_experts routes every token to every expert."""
        tokens = [f"tok_{i}" for i in range(10)]
        res = route_tokens_to_experts(tokens, num_routed_experts=4, top_k=4)
        for exp_count in res.values():
            self.assertEqual(exp_count, 10)

    def test_invalid_capacity_factor(self):
        """Verify non-positive capacity factor raises ValueError."""
        with self.assertRaises(ValueError):
            route_tokens_to_experts(["t1"], num_routed_experts=8, capacity_factor=0.0)
        with self.assertRaises(ValueError):
            route_tokens_to_experts(["t1"], num_routed_experts=8, capacity_factor=-1.0)
        with self.assertRaises(ValueError):
            calculate_expert_capacity(100, 8, 2, capacity_factor=0.0)

    def test_invalid_forced_expert(self):
        """Verify out-of-range forced_expert raises ValueError."""
        with self.assertRaises(ValueError):
            route_tokens_to_experts(["t1"], num_routed_experts=8, forced_expert=-1)
        with self.assertRaises(ValueError):
            route_tokens_to_experts(["t1"], num_routed_experts=8, forced_expert=8)

    def test_invalid_num_shared_experts(self):
        """Verify negative shared experts count raises ValueError."""
        with self.assertRaises(ValueError):
            route_tokens_to_experts(["t1"], num_shared_experts=-1)

    def test_routing_result_dict_contract(self):
        """Verify RoutingResult behaves as a standard dictionary for backward compatibility."""
        tokens = ["t1", "t2"]
        res = route_tokens_to_experts(tokens, num_routed_experts=4, top_k=2)
        self.assertIsInstance(res, dict)
        self.assertEqual(len(res), 4)
        self.assertTrue(all(isinstance(k, int) for k in res.keys()))
        self.assertTrue(all(isinstance(v, int) for v in res.values()))
        # Dict lookup syntax
        self.assertIsInstance(res[0], int)
        self.assertEqual(res.get(999, -1), -1)


class TestDeepSeekV3MegaMoEIntegration(unittest.TestCase):
    """End-to-end integration tests combining DualPipe scheduling and Mega-MoE dispatch."""

    def test_deepseek_v3_production_fleet_simulation(self):
        """
        Simulate DeepSeek-V3 671B inference pipeline:
        - 256 routed experts + 1 shared expert
        - 8 pipeline stages
        - 16 microbatches
        - DualPipe bidirectional 2-chunk overlap
        """
        # 1. Pipeline schedule
        dualpipe = simulate_dualpipe_overlap(num_microbatches=16, num_pipeline_stages=8, num_chunks=2)
        self.assertEqual(dualpipe["Standard_1F1B_Bubble"], 43.75)
        self.assertEqual(dualpipe["DualPipe_Bubble"], 21.88)
        self.assertEqual(dualpipe["Bubble_Reduction_Pct"], 50.0)
        self.assertGreater(dualpipe["Overlap_Ratio_Pct"], 60.0)

        # 2. Mega-MoE token dispatch
        batch_tokens = [f"tok_{i}" for i in range(2048)]
        moe = route_tokens_to_experts(
            tokens=batch_tokens,
            num_routed_experts=256,
            top_k=8,
            capacity_factor=1.2,
            num_shared_experts=1,
            fallback_to_shared=True,
            seed=42
        )

        # Expected capacity per expert = ceil(2048 * 8 / 256 * 1.2) = ceil(64 * 1.2) = 77
        self.assertEqual(moe.capacity_limit, 77)
        self.assertEqual(len(moe), 256)
        # Shared expert receives all 2048 tokens + any fallback
        self.assertGreaterEqual(moe.shared_expert_counts[0], 2048)
        # Verify CV of well-balanced hash/router distribution is low
        self.assertLess(moe.get_coefficient_of_variation(), 0.3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
