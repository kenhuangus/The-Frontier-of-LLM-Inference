"""
Unit Test Suite for Chapter 1: Physics and Roofline Model of LLM Inference
The Frontier of LLM Inference (2026 Edition)

Validates:
1. KV Cache Sizing & Capacity Calculator (MHA, GQA, MQA, MLA, fleet cluster sizing)
2. Roofline Model Analyzer (Ridge points, MFU, compute vs memory bound transitions)
3. Discrete-Event Continuous Batching Serving Simulator (Chunked vs monolithic prefill, tail latency)
4. Inference Telemetry Tracker (TTFT, ITL, P50, P90, P95, P99, Jitter, Throughput)
"""
import math
import statistics
import unittest

from kv_cache_sizing import (
    calculate_kv_cache_bytes_per_token,
    calculate_mla_bytes_per_token,
    size_cluster_fleet
)
from roofline_analyzer import (
    analyze_roofline,
    classify_inference_regime
)
from serving_simulator import (
    ServingRequest,
    simulate_serving_policy
)
from telemetry_tracker import (
    InferenceTelemetryTracker
)


class TestKVCacheSizing(unittest.TestCase):
    """Tests for KV Cache memory footprint formulas and fleet sizing calculator."""

    def test_standard_mha_formula(self):
        """MHA with 32 layers, 32 KV heads, 128 head dim, FP16 (2 bytes)."""
        bytes_per_tok = calculate_kv_cache_bytes_per_token(
            num_layers=32,
            num_kv_heads=32,
            head_dim=128,
            precision_bytes=2
        )
        expected = 2 * 32 * 32 * 128 * 2  # 524,288 bytes (512 KB)
        self.assertEqual(bytes_per_tok, expected)
        self.assertEqual(bytes_per_tok, 524288)

    def test_standard_gqa_formula(self):
        """GQA with 80 layers, 8 KV heads, 128 head dim, FP16 (Llama-3-70B)."""
        bytes_per_tok = calculate_kv_cache_bytes_per_token(
            num_layers=80,
            num_kv_heads=8,
            head_dim=128,
            precision_bytes=2
        )
        expected = 2 * 80 * 8 * 128 * 2  # 327,680 bytes (320 KB)
        self.assertEqual(bytes_per_tok, expected)
        self.assertEqual(bytes_per_tok, 327680)

    def test_mqa_formula(self):
        """MQA with 1 KV head (multi-query attention)."""
        bytes_per_tok = calculate_kv_cache_bytes_per_token(
            num_layers=32,
            num_kv_heads=1,
            head_dim=128,
            precision_bytes=2
        )
        expected = 2 * 32 * 1 * 128 * 2  # 16,384 bytes (16 KB)
        self.assertEqual(bytes_per_tok, expected)

    def test_mha_vs_gqa_compression_ratio(self):
        """Verify GQA-8 reduces KV cache footprint by 4x compared to MHA-32."""
        mha = calculate_kv_cache_bytes_per_token(num_layers=32, num_kv_heads=32, head_dim=128)
        gqa = calculate_kv_cache_bytes_per_token(num_layers=32, num_kv_heads=8, head_dim=128)
        self.assertEqual(mha / gqa, 4.0)

    def test_deepseek_mla_formula(self):
        """DeepSeek Multi-Head Latent Attention (MLA) footprint with latent_dim=512, rope_dim=64."""
        bytes_per_tok = calculate_mla_bytes_per_token(
            num_layers=80,
            latent_dim=512,
            rope_dim=64,
            precision_bytes=2
        )
        expected = 80 * (512 + 64) * 2  # 92,160 bytes (90 KB)
        self.assertEqual(bytes_per_tok, expected)
        self.assertEqual(bytes_per_tok, 92160)

    def test_mla_vs_gqa_vram_savings(self):
        """Verify DeepSeek MLA achieves ~71.88% savings over standard GQA-8."""
        gqa = calculate_kv_cache_bytes_per_token(num_layers=80, num_kv_heads=8, head_dim=128)
        mla = calculate_mla_bytes_per_token(num_layers=80, latent_dim=512, rope_dim=64)
        savings = ((gqa - mla) / gqa) * 100.0
        self.assertAlmostEqual(savings, 71.875, places=2)

    def test_precision_scaling(self):
        """Verify FP8 (1 byte) halves the footprint compared to FP16 (2 bytes)."""
        fp16 = calculate_kv_cache_bytes_per_token(num_layers=40, num_kv_heads=8, head_dim=128, precision_bytes=2)
        fp8 = calculate_kv_cache_bytes_per_token(num_layers=40, num_kv_heads=8, head_dim=128, precision_bytes=1)
        self.assertEqual(fp16, fp8 * 2)

    def test_cluster_fleet_sizing_gqa_and_mla(self):
        """Verify cluster fleet requirements for 70B model with 32k context and 64 streams."""
        fleet_gqa = size_cluster_fleet(70.0, 32768, 64, use_mla=False, gpu_vram_gb=80.0)
        fleet_mla = size_cluster_fleet(70.0, 32768, 64, use_mla=True, gpu_vram_gb=80.0)

        # GQA: 140 GB weights, 640 GB KV cache, 936 GB total (with 20% headroom) -> 12 GPUs
        self.assertEqual(fleet_gqa["Weights_VRAM_GB"], 140.0)
        self.assertEqual(fleet_gqa["Total_KV_Cache_GB"], 640.0)
        self.assertEqual(fleet_gqa["Total_Required_VRAM_GB"], 936.0)
        self.assertEqual(fleet_gqa["Minimum_GPUs_80GB"], 12)

        # MLA: 140 GB weights, 180 GB KV cache, 384 GB total (with 20% headroom) -> 5 GPUs
        self.assertEqual(fleet_mla["Weights_VRAM_GB"], 140.0)
        self.assertEqual(fleet_mla["Total_KV_Cache_GB"], 180.0)
        self.assertEqual(fleet_mla["Total_Required_VRAM_GB"], 384.0)
        self.assertEqual(fleet_mla["Minimum_GPUs_80GB"], 5)

    def test_cluster_fleet_exact_boundary_ceil(self):
        """Ceil behavior: if total required VRAM exactly equals 80 GB, exactly 1 GPU is needed."""
        # weights = 40.0 * 2.0 = 80.0 GB, context=0 -> total required = 80.0 * 1.2 = 96.0 -> 2 GPUs
        # If we construct total_required = 80.0 exactly: weights = (80 / 1.2) / 2 = 33.333333333333336
        fleet = size_cluster_fleet(model_param_billions=(80.0 / 1.2) / 2.0, context_tokens=0, concurrency=0, gpu_vram_gb=80.0)
        self.assertEqual(fleet["Total_Required_VRAM_GB"], 80.0)
        self.assertEqual(fleet["Minimum_GPUs_80GB"], 1)

    def test_cluster_fleet_zero_parameters_and_concurrency(self):
        """Zero load: 0 weights, 0 tokens, 0 concurrency should require 0 GPUs."""
        fleet = size_cluster_fleet(0.0, 0, 0)
        self.assertEqual(fleet["Weights_VRAM_GB"], 0.0)
        self.assertEqual(fleet["Total_KV_Cache_GB"], 0.0)
        self.assertEqual(fleet["Total_Required_VRAM_GB"], 0.0)
        self.assertEqual(fleet["Minimum_GPUs_80GB"], 0)

    def test_cluster_fleet_extreme_concurrency(self):
        """Extreme batch size: 10,000 concurrency with 4k context."""
        fleet = size_cluster_fleet(70.0, 4096, 10000, use_mla=False, gpu_vram_gb=80.0)
        self.assertGreater(fleet["Total_KV_Cache_GB"], 10000.0)
        self.assertGreater(fleet["Minimum_GPUs_80GB"], 150)

    def test_negative_input_validation(self):
        """Ensure negative dimensions, parameters, or GPU sizes raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_kv_cache_bytes_per_token(-1, 8, 128)
        with self.assertRaises(ValueError):
            calculate_mla_bytes_per_token(80, -512, 64)
        with self.assertRaises(ValueError):
            size_cluster_fleet(-70.0, 1024, 8)
        with self.assertRaises(ValueError):
            size_cluster_fleet(70.0, -1024, 8)
        with self.assertRaises(ValueError):
            size_cluster_fleet(70.0, 1024, -8)
        with self.assertRaises(ValueError):
            size_cluster_fleet(70.0, 1024, 8, gpu_vram_gb=0)


class TestRooflineAnalyzer(unittest.TestCase):
    """Tests for Roofline operational intensity, ridge points, MFU, and bound transitions."""

    def test_nvidia_h100_decode_regime(self):
        """H100 SXM decode batch=1 (memory-bandwidth bound GEMV)."""
        res = analyze_roofline("NVIDIA H100 SXM (Decode B=1)", 1979.0, 3.35, 1.5)
        self.assertEqual(res["Hardware"], "NVIDIA H100 SXM (Decode B=1)")
        self.assertEqual(res["Ridge_Point_FLOP_per_Byte"], 590.75)
        self.assertEqual(res["Operational_Intensity"], 1.5)
        self.assertEqual(res["Attainable_TFLOPS"], 5.03)  # 1.5 * 3.35 = 5.025
        self.assertTrue(res["Is_Memory_Bound"])

        regime = classify_inference_regime("NVIDIA H100 SXM", 1979.0, res["Attainable_TFLOPS"], res["Is_Memory_Bound"])
        self.assertEqual(regime["Execution_Regime"], "Memory-Bandwidth Bound (GEMV decode)")
        self.assertAlmostEqual(regime["MFU_Percent"], 0.25, places=2)

    def test_nvidia_b200_prefill_regime(self):
        """B200 NVL with high operational intensity (compute-bound GEMM)."""
        ridge = 4500.0 / 8.0  # 562.5 FLOP/Byte
        res = analyze_roofline("NVIDIA B200 NVL (Prefill 4k)", 4500.0, 8.0, 800.0)
        self.assertEqual(res["Ridge_Point_FLOP_per_Byte"], 562.5)
        self.assertFalse(res["Is_Memory_Bound"])
        self.assertEqual(res["Attainable_TFLOPS"], 4500.0)

        regime = classify_inference_regime("NVIDIA B200 NVL", 4500.0, res["Attainable_TFLOPS"], res["Is_Memory_Bound"])
        self.assertEqual(regime["Execution_Regime"], "Compute Bound (GEMM prefill)")
        self.assertEqual(regime["MFU_Percent"], 100.0)

    def test_roofline_bound_transition_threshold(self):
        """Test exact transition between memory-bound and compute-bound around ridge point."""
        peak = 1000.0
        bw = 2.0
        ridge_point = peak / bw  # 500.0 FLOP/Byte

        # Just below ridge point -> Memory bound
        below = analyze_roofline("TestGPU", peak, bw, 499.9)
        self.assertTrue(below["Is_Memory_Bound"])
        self.assertLess(below["Attainable_TFLOPS"], peak)
        regime_below = classify_inference_regime("TestGPU", peak, below["Attainable_TFLOPS"], below["Is_Memory_Bound"])
        self.assertIn("Memory-Bandwidth Bound", regime_below["Execution_Regime"])

        # Exactly at ridge point -> Transitions to Compute bound ceiling
        at_ridge = analyze_roofline("TestGPU", peak, bw, 500.0)
        self.assertFalse(at_ridge["Is_Memory_Bound"])
        self.assertEqual(at_ridge["Attainable_TFLOPS"], peak)
        regime_at = classify_inference_regime("TestGPU", peak, at_ridge["Attainable_TFLOPS"], at_ridge["Is_Memory_Bound"])
        self.assertIn("Compute Bound", regime_at["Execution_Regime"])
        self.assertEqual(regime_at["MFU_Percent"], 100.0)

        # Just above ridge point -> Compute bound
        above = analyze_roofline("TestGPU", peak, bw, 500.1)
        self.assertFalse(above["Is_Memory_Bound"])
        self.assertEqual(above["Attainable_TFLOPS"], peak)

    def test_roofline_zero_operational_intensity(self):
        """Zero intensity -> 0 attainable TFLOPS and 0% MFU."""
        res = analyze_roofline("ZeroGPU", 1000.0, 2.0, 0.0)
        self.assertEqual(res["Attainable_TFLOPS"], 0.0)
        self.assertTrue(res["Is_Memory_Bound"])
        regime = classify_inference_regime("ZeroGPU", 1000.0, res["Attainable_TFLOPS"], res["Is_Memory_Bound"])
        self.assertEqual(regime["MFU_Percent"], 0.0)

    def test_roofline_invalid_inputs(self):
        """Test validation for non-positive hardware parameters."""
        with self.assertRaises(ValueError):
            analyze_roofline("GPU", -100.0, 2.0, 10.0)
        with self.assertRaises(ValueError):
            analyze_roofline("GPU", 100.0, 0.0, 10.0)
        with self.assertRaises(ValueError):
            analyze_roofline("GPU", 100.0, 2.0, -1.0)
        with self.assertRaises(ValueError):
            classify_inference_regime("GPU", 0.0, 10.0, False)
        with self.assertRaises(ValueError):
            classify_inference_regime("GPU", 100.0, -5.0, False)


class TestServingSimulator(unittest.TestCase):
    """Tests for discrete-event continuous batching simulator and chunked prefill isolation."""

    def test_serving_request_init(self):
        """Verify ServingRequest state tracking."""
        req = ServingRequest("req-1", prompt_len=1024, gen_len=64, arrival_ms=0.0)
        self.assertEqual(req.remaining_prefill, 1024)
        self.assertEqual(req.remaining_decode, 64)
        self.assertEqual(req.itls_ms, [])

        with self.assertRaises(ValueError):
            ServingRequest("bad", prompt_len=-1, gen_len=10)
        with self.assertRaises(ValueError):
            ServingRequest("bad", prompt_len=10, gen_len=-1)

    def test_monolithic_vs_chunked_tail_latency(self):
        """Verify chunked prefill isolates tail latency compared to monolithic prefill."""
        monolithic = simulate_serving_policy(chunk_size=4096, num_requests=20)
        chunked = simulate_serving_policy(chunk_size=512, num_requests=20)

        # P50 should remain close to base decode step cost (~15ms)
        self.assertEqual(monolithic["P50_ITL_ms"], 15.0)
        self.assertEqual(chunked["P50_ITL_ms"], 15.0)

        # Chunked prefill strictly bounds maximum chunk size and therefore P99 tail latency
        self.assertLess(chunked["P99_ITL_ms"], monolithic["P99_ITL_ms"])
        # Monolithic tail spike is ~44ms, chunked is ~22.68ms
        self.assertAlmostEqual(chunked["P99_ITL_ms"], 22.68, delta=1.0)
        self.assertAlmostEqual(monolithic["P99_ITL_ms"], 44.04, delta=2.0)

    def test_zero_requests_edge_case(self):
        """Simulate with zero requests: must return cleanly with 0.0 latencies without crashing."""
        res = simulate_serving_policy(chunk_size=512, num_requests=0)
        self.assertEqual(res["Chunk_Size"], 512)
        self.assertEqual(res["P50_ITL_ms"], 0.0)
        self.assertEqual(res["P99_ITL_ms"], 0.0)

    def test_single_request_edge_case(self):
        """Simulate with exactly 1 request."""
        custom = [ServingRequest("single-req", prompt_len=100, gen_len=5, arrival_ms=0.0)]
        res = simulate_serving_policy(chunk_size=512, custom_requests=custom)
        self.assertGreater(res["P50_ITL_ms"], 0.0)
        self.assertGreater(res["P99_ITL_ms"], 0.0)

    def test_boundary_token_lengths(self):
        """Test boundary lengths: 0 prompt tokens and 0 generation tokens."""
        # Case A: Prompt len = 0, Gen len = 10 (immediate decode)
        req_zero_prompt = [ServingRequest("req-zp", prompt_len=0, gen_len=10)]
        res_a = simulate_serving_policy(chunk_size=512, custom_requests=req_zero_prompt)
        self.assertEqual(res_a["P50_ITL_ms"], 15.0)

        # Case B: Prompt len = 500, Gen len = 0 (prefill only, zero decode tokens)
        req_zero_gen = [ServingRequest("req-zg", prompt_len=500, gen_len=0)]
        res_b = simulate_serving_policy(chunk_size=512, custom_requests=req_zero_gen)
        self.assertEqual(res_b["P50_ITL_ms"], 0.0)
        self.assertEqual(res_b["P99_ITL_ms"], 0.0)

    def test_extreme_chunk_sizes(self):
        """Test very small chunk (1 token) and huge chunk (100k tokens)."""
        custom = [ServingRequest("req-1", prompt_len=100, gen_len=5)]
        res_micro = simulate_serving_policy(chunk_size=1, custom_requests=custom)
        self.assertGreater(res_micro["P50_ITL_ms"], 0.0)

        res_huge = simulate_serving_policy(chunk_size=100000, custom_requests=custom)
        self.assertGreater(res_huge["P50_ITL_ms"], 0.0)

    def test_invalid_simulation_parameters(self):
        """Test validation for invalid chunk size and negative request counts."""
        with self.assertRaises(ValueError):
            simulate_serving_policy(chunk_size=0)
        with self.assertRaises(ValueError):
            simulate_serving_policy(chunk_size=-512)
        with self.assertRaises(ValueError):
            simulate_serving_policy(num_requests=-1)


class TestTelemetryTracker(unittest.TestCase):
    """Tests for InferenceTelemetryTracker high-precision telemetry, jitter, and percentiles."""

    def test_zero_tokens_error(self):
        """Tracker must raise ValueError if no tokens were recorded."""
        tracker = InferenceTelemetryTracker("req-empty")
        with self.assertRaises(ValueError):
            tracker.compute_summary()
        with self.assertRaises(ValueError):
            tracker.compute_jitter_and_throughput()

    def test_single_token_edge_case(self):
        """Tracker with exactly 1 token has 0 ITL jitter and 0 ITL tail latencies."""
        t0 = 100.0
        t1 = 100.05  # 50ms TTFT
        tracker = InferenceTelemetryTracker("req-single", t_submit=t0)
        tracker.record_token(timestamp=t1)

        summary = tracker.compute_summary()
        self.assertEqual(summary["Total_Tokens"], 1)
        self.assertEqual(summary["TTFT_ms"], 50.0)
        self.assertEqual(summary["P50_ITL_ms"], 0.0)
        self.assertEqual(summary["P90_ITL_ms"], 0.0)
        self.assertEqual(summary["P95_ITL_ms"], 0.0)
        self.assertEqual(summary["P99_ITL_ms"], 0.0)

        metrics = tracker.compute_jitter_and_throughput()
        self.assertEqual(metrics["ITL_Jitter_Std_ms"], 0.0)
        self.assertEqual(metrics["Generation_Tokens_Per_Sec"], 0.0)

    def test_deterministic_percentiles_p50_p90_p95_p99(self):
        """Inject 101 tokens with linearly increasing ITLs from 10ms to 110ms."""
        t_submit = 0.0
        t_first = 0.040  # 40ms TTFT
        tracker = InferenceTelemetryTracker("req-100", t_submit=t_submit)
        tracker.record_token(timestamp=t_first)

        # 100 inter-token intervals: 10.0 ms, 11.0 ms, ..., 109.0 ms
        curr = t_first
        for i in range(100):
            interval_sec = (10.0 + i) / 1000.0
            curr += interval_sec
            tracker.record_token(timestamp=curr)

        summary = tracker.compute_summary()
        self.assertEqual(summary["Total_Tokens"], 101)
        self.assertEqual(summary["TTFT_ms"], 40.0)

        # ITLs are [10.0, 11.0, ..., 109.0]. Length = 100.
        # P50 index = 50 -> 60.0 ms
        # P90 index = min(int(100 * 0.90), 99) = 90 -> 100.0 ms
        # P95 index = min(int(100 * 0.95), 99) = 95 -> 105.0 ms
        # P99 index = min(int(100 * 0.99), 99) = 99 -> 109.0 ms
        self.assertEqual(summary["P50_ITL_ms"], 60.0)
        self.assertEqual(summary["P90_ITL_ms"], 100.0)
        self.assertEqual(summary["P95_ITL_ms"], 105.0)
        self.assertEqual(summary["P99_ITL_ms"], 109.0)

        # Monotonicity check
        self.assertLessEqual(summary["P50_ITL_ms"], summary["P90_ITL_ms"])
        self.assertLessEqual(summary["P90_ITL_ms"], summary["P95_ITL_ms"])
        self.assertLessEqual(summary["P95_ITL_ms"], summary["P99_ITL_ms"])

    def test_jitter_std_and_throughput(self):
        """Verify zero jitter for strictly constant ITL and accurate throughput calculation."""
        t_submit = 1.0
        tracker = InferenceTelemetryTracker("req-const", t_submit=t_submit)
        
        # 51 tokens at strictly 20ms intervals (0.020 sec)
        # Total generated decode tokens = 50 tokens over exactly 1.000 second -> 50.0 tokens/sec
        t_first = 1.050  # 50ms TTFT
        tracker.record_token(timestamp=t_first)
        for i in range(1, 51):
            tracker.record_token(timestamp=t_first + i * 0.020)

        metrics = tracker.compute_jitter_and_throughput()
        self.assertEqual(metrics["Total_Tokens"], 51)
        self.assertEqual(metrics["TTFT_ms"], 50.0)
        self.assertEqual(metrics["P50_ITL_ms"], 20.0)
        self.assertEqual(metrics["P90_ITL_ms"], 20.0)
        self.assertEqual(metrics["P99_ITL_ms"], 20.0)
        self.assertEqual(metrics["ITL_Jitter_Std_ms"], 0.0)  # Zero jitter
        self.assertEqual(metrics["Generation_Tokens_Per_Sec"], 50.0)

    def test_known_jitter_standard_deviation(self):
        """Verify ITL jitter std matches standard statistical calculation."""
        tracker = InferenceTelemetryTracker("req-jitter", t_submit=0.0)
        tracker.record_token(timestamp=0.1)  # token 0
        # Intervals in seconds: [0.010, 0.020, 0.030, 0.040] -> ITLs: [10, 20, 30, 40] ms
        timestamps = [0.10, 0.11, 0.13, 0.16, 0.20]
        for t in timestamps[1:]:
            tracker.record_token(timestamp=t)

        metrics = tracker.compute_jitter_and_throughput()
        expected_stdev = statistics.stdev([10.0, 20.0, 30.0, 40.0])  # ~12.91 ms
        self.assertAlmostEqual(metrics["ITL_Jitter_Std_ms"], round(expected_stdev, 2), places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
