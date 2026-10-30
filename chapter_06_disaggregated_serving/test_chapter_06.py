"""
Unit Test Suite for Chapter 6: Disaggregated Serving Architectures (DistServe, Mooncake, vLLM V1)
The Frontier of LLM Inference

Validates:
1. Mooncake 4-Tier RDMA KV Cache Transfer Engine:
   - Line-rate bandwidth calculations and base NIC latency over 100-1600 Gbps fabrics
   - Small 4KB page transfers (latency-dominated) vs 1GB KV cache dumps (bandwidth-dominated)
   - RoCEv2 MTU chunking, packetization, and framing overhead (1024, 2048, 4096 MTUs)
   - Effective bandwidth goodput and line-rate efficiency asymptotic scaling
   - Edge cases: zero payload, negative inputs, invalid MTU, zero bandwidth

2. Disaggregated Prefill-Decode (PD) Serving Simulator:
   - Coupled (monolithic collocated) vs disaggregated serving comparisons
   - Queue decoupling: tail TTFT reduction and decode ITL / TPOT jitter isolation
   - Queue empty conditions (0 requests gracefully handled without division errors)
   - Heavy burst loads (simultaneous arrival, queuing delay, decode protection)
   - TTFT and ITL metric validation and percentile monotonicity (P50 <= P99)
   - Worker node failure tolerance (partial failure graceful degradation, total failure error handling)
   - Worker node saturation and backpressure queue capacity bounding (request dropping)
   - ServingRequest lifecycle and property calculations

3. End-to-End CLI Integration:
   - Clean execution of Chapter 6 scripts in standalone mode
"""
import math
import os
import subprocess
import sys
import unittest
from typing import Dict, Union

# Ensure the chapter directory is on the module search path
CHAPTER_DIR = os.path.dirname(os.path.abspath(__file__))
if CHAPTER_DIR not in sys.path:
    sys.path.insert(0, CHAPTER_DIR)

from disaggregated_pd_simulator import (
    DisaggregatedServingSimulator,
    ServingRequest,
    simulate_cluster_serving
)
from rdma_kv_transfer import (
    calculate_effective_bandwidth_gbps,
    calculate_rdma_transfer_time_ms,
    simulate_chunked_rdma_transfer
)


class TestRDMAKVTransferBasic(unittest.TestCase):
    """Tests for basic RDMA transfer formulas, bandwidth scaling, and parameter validation."""

    def test_default_deepseek_mla_transfer(self):
        """Verify baseline DeepSeek MLA 4000-token prompt KV transfer over 400Gbps RoCEv2."""
        # 4000 tokens * 576 latent dim * 2 bytes = 4,608,000 bytes (~4.39 MB)
        kv_bytes = 4000 * 576 * 2
        transfer_ms = calculate_rdma_transfer_time_ms(
            kv_cache_bytes=kv_bytes,
            network_bandwidth_gbps=400.0,
            base_latency_us=5.0
        )
        # Expected: (4608000 / (400e9 / 8)) + 5e-6 = 0.00009216 + 0.000005 = 0.00009716 s = 0.09716 ms
        self.assertAlmostEqual(transfer_ms, 0.09716, places=4)
        self.assertLess(transfer_ms, 0.1)  # Sub-millisecond handoff

    def test_varying_network_bandwidths(self):
        """Verify inverse scaling of transfer time across NIC speeds: 100G, 200G, 400G, 800G, 1600G."""
        kv_bytes = 50 * 1024 * 1024  # 50 MB
        speeds = [100.0, 200.0, 400.0, 800.0, 1600.0]
        prev_time = float("inf")

        for speed in speeds:
            t = calculate_rdma_transfer_time_ms(kv_bytes, network_bandwidth_gbps=speed, base_latency_us=5.0)
            self.assertGreater(t, 0.0)
            self.assertLess(t, prev_time, f"Higher bandwidth {speed} Gbps should yield lower transfer time")
            prev_time = t

        # Doubling bandwidth from 400G to 800G should roughly halve serialization time
        t_400 = calculate_rdma_transfer_time_ms(kv_bytes, network_bandwidth_gbps=400.0, base_latency_us=0.0)
        t_800 = calculate_rdma_transfer_time_ms(kv_bytes, network_bandwidth_gbps=800.0, base_latency_us=0.0)
        self.assertAlmostEqual(t_400 / t_800, 2.0, places=3)

    def test_zero_payload_edge_case(self):
        """Zero bytes KV cache should return 0.0 ms without executing division."""
        t_zero = calculate_rdma_transfer_time_ms(0, network_bandwidth_gbps=400.0, base_latency_us=5.0)
        self.assertEqual(t_zero, 0.0)

        bw_zero = calculate_effective_bandwidth_gbps(0, transfer_time_ms=5.0)
        self.assertEqual(bw_zero, 0.0)

        sim_zero = simulate_chunked_rdma_transfer(0, network_bandwidth_gbps=400.0)
        self.assertEqual(sim_zero["transfer_time_ms"], 0.0)
        self.assertEqual(sim_zero["num_packets"], 0)
        self.assertEqual(sim_zero["effective_bandwidth_gbps"], 0.0)

    def test_base_latency_sensitivity(self):
        """Verify that base NIC latency directly offsets total transfer time."""
        kv_bytes = 1024 * 1024  # 1 MB
        t_lat0 = calculate_rdma_transfer_time_ms(kv_bytes, network_bandwidth_gbps=400.0, base_latency_us=0.0)
        t_lat10 = calculate_rdma_transfer_time_ms(kv_bytes, network_bandwidth_gbps=400.0, base_latency_us=10.0)
        # Difference should be exactly 10 microseconds = 0.010 ms
        diff_ms = t_lat10 - t_lat0
        self.assertAlmostEqual(diff_ms, 0.010, places=5)

    def test_input_validation_errors(self):
        """Verify that invalid inputs raise appropriate ValueErrors."""
        with self.assertRaises(ValueError):
            calculate_rdma_transfer_time_ms(-100)  # negative bytes

        with self.assertRaises(ValueError):
            calculate_rdma_transfer_time_ms(1024, network_bandwidth_gbps=0.0)  # zero bandwidth

        with self.assertRaises(ValueError):
            calculate_rdma_transfer_time_ms(1024, network_bandwidth_gbps=-400.0)  # negative bandwidth

        with self.assertRaises(ValueError):
            calculate_rdma_transfer_time_ms(1024, base_latency_us=-1.0)  # negative latency

        with self.assertRaises(ValueError):
            calculate_rdma_transfer_time_ms(1024, mtu_bytes=0)  # zero MTU

        with self.assertRaises(ValueError):
            calculate_rdma_transfer_time_ms(1024, header_overhead_bytes=-1)  # negative overhead


class TestRDMATransferSizeScaling(unittest.TestCase):
    """Tests for payload size scaling from 4KB pages to 1GB dumps."""

    def test_small_4kb_block_latency_dominated(self):
        """Small 4KB page transfer is dominated by NIC base latency (~5us)."""
        kv_bytes = 4096  # 4KB
        sim = simulate_chunked_rdma_transfer(
            kv_cache_bytes=kv_bytes,
            network_bandwidth_gbps=400.0,
            base_latency_us=5.0,
            mtu_bytes=4096
        )
        # Serialization time: 4096 / (50 GB/s) = 8.192e-8 s = 0.00008 ms = 0.08192 us
        # Total latency is heavily dominated by 5 us base latency
        self.assertEqual(sim["num_packets"], 1)
        self.assertAlmostEqual(sim["transfer_time_ms"], 0.00508, places=4)
        # Efficiency is very low (< 5%) due to base latency overhead
        self.assertLess(sim["line_rate_efficiency"], 0.05)
        self.assertLess(sim["effective_bandwidth_gbps"], 20.0)

    def test_medium_prompt_transfer(self):
        """Medium 4.6 MB prompt transfer achieves high bandwidth utilization."""
        kv_bytes = 4608000
        sim = simulate_chunked_rdma_transfer(
            kv_cache_bytes=kv_bytes,
            network_bandwidth_gbps=400.0,
            base_latency_us=5.0,
            mtu_bytes=4096
        )
        self.assertLess(sim["transfer_time_ms"], 0.15)  # < 0.15 ms
        # High line rate efficiency (> 90%)
        self.assertGreater(sim["line_rate_efficiency"], 0.90)
        self.assertGreater(sim["effective_bandwidth_gbps"], 360.0)

    def test_large_1gb_dump_bandwidth_dominated(self):
        """Large 1GB KV dump (1,073,741,824 bytes) reaches near-peak line rate (>98%)."""
        kv_bytes = 1024 * 1024 * 1024  # 1 GB
        sim = simulate_chunked_rdma_transfer(
            kv_cache_bytes=kv_bytes,
            network_bandwidth_gbps=400.0,
            base_latency_us=5.0,
            mtu_bytes=4096,
            header_overhead_bytes=64
        )
        # Serialization time over 400 Gbps: ~21.5 ms
        self.assertGreater(sim["transfer_time_ms"], 20.0)
        self.assertLess(sim["transfer_time_ms"], 23.0)
        # Base latency (5 us) is completely negligible (< 0.05% of total time)
        base_share = (5.0 * 1e-3) / sim["transfer_time_ms"]
        self.assertLess(base_share, 0.001)
        # Achieved efficiency accounts for 64B per 4096B packet overhead: 4096 / (4096 + 64) = ~98.46%
        self.assertGreater(sim["line_rate_efficiency"], 0.97)

    def test_monotone_transfer_time_scaling(self):
        """Transfer time must strictly increase as payload scales."""
        sizes = [4096, 65536, 1048576, 16777216, 67108864, 1073741824]
        times = [
            calculate_rdma_transfer_time_ms(sz, network_bandwidth_gbps=400.0, base_latency_us=5.0)
            for sz in sizes
        ]
        for i in range(len(times) - 1):
            self.assertLess(times[i], times[i + 1], "Larger payloads must take longer to transfer")

    def test_effective_bandwidth_asymptote(self):
        """Effective bandwidth must monotonically increase toward line-rate as payload increases."""
        sizes = [4096, 65536, 1048576, 16777216, 67108864, 1073741824]
        eff_bws = []
        for sz in sizes:
            t = calculate_rdma_transfer_time_ms(sz, network_bandwidth_gbps=400.0, base_latency_us=5.0)
            bw = calculate_effective_bandwidth_gbps(sz, t)
            eff_bws.append(bw)

        for i in range(len(eff_bws) - 1):
            self.assertLessEqual(eff_bws[i], eff_bws[i + 1], "Effective goodput should increase with payload")
        self.assertGreater(eff_bws[-1], 390.0)  # Near 400 Gbps line rate


class TestRDMAMTUChunking(unittest.TestCase):
    """Tests for packetization, RoCEv2 MTU sizing, and framing overhead."""

    def test_packet_count_exact_and_non_exact_multiples(self):
        """Verify packet counting with ceiling logic."""
        mtu = 4096
        # Exact single packet
        sim1 = simulate_chunked_rdma_transfer(4096, mtu_bytes=mtu)
        self.assertEqual(sim1["num_packets"], 1)

        # 1 byte over -> 2 packets
        sim2 = simulate_chunked_rdma_transfer(4097, mtu_bytes=mtu)
        self.assertEqual(sim2["num_packets"], 2)

        # Exact 10 packets
        sim10 = simulate_chunked_rdma_transfer(4096 * 10, mtu_bytes=mtu)
        self.assertEqual(sim10["num_packets"], 10)

    def test_mtu_size_framing_efficiency(self):
        """Larger MTU (4096) yields lower framing overhead than smaller MTU (1024)."""
        payload = 1024 * 1024  # 1 MB
        sim_1k = simulate_chunked_rdma_transfer(payload, mtu_bytes=1024, header_overhead_bytes=64)
        sim_2k = simulate_chunked_rdma_transfer(payload, mtu_bytes=2048, header_overhead_bytes=64)
        sim_4k = simulate_chunked_rdma_transfer(payload, mtu_bytes=4096, header_overhead_bytes=64)

        self.assertEqual(sim_1k["num_packets"], 1024)
        self.assertEqual(sim_2k["num_packets"], 512)
        self.assertEqual(sim_4k["num_packets"], 256)

        # Wire bytes should be higher for 1024 MTU due to 4x packet headers
        self.assertGreater(sim_1k["wire_bytes"], sim_2k["wire_bytes"])
        self.assertGreater(sim_2k["wire_bytes"], sim_4k["wire_bytes"])

        # Transfer time with 4K MTU should be strictly lower than 1K MTU
        self.assertLess(sim_4k["transfer_time_ms"], sim_1k["transfer_time_ms"])

    def test_chunked_vs_unchunked_overhead_comparison(self):
        """Packetized MTU transfer takes strictly longer than ideal unchunked transfer due to header framing."""
        payload = 10 * 1024 * 1024  # 10 MB
        unchunked_t = calculate_rdma_transfer_time_ms(payload, network_bandwidth_gbps=400.0, mtu_bytes=None)
        chunked_t = calculate_rdma_transfer_time_ms(payload, network_bandwidth_gbps=400.0, mtu_bytes=4096, header_overhead_bytes=64)
        self.assertGreater(chunked_t, unchunked_t)
        # Extra time corresponds to 64 bytes per 4096 bytes: ~1.56% extra serialization time
        ratio = chunked_t / unchunked_t
        self.assertAlmostEqual(ratio, (4096 + 64) / 4096, delta=0.01)

    def test_simulate_chunked_invalid_inputs(self):
        """simulate_chunked_rdma_transfer raises ValueError on negative bytes or invalid MTU."""
        with self.assertRaises(ValueError):
            simulate_chunked_rdma_transfer(-1024)
        with self.assertRaises(ValueError):
            simulate_chunked_rdma_transfer(4096, mtu_bytes=0)

    def test_effective_bandwidth_boundary_values(self):
        """calculate_effective_bandwidth_gbps handles boundary zero and negative values."""
        self.assertEqual(calculate_effective_bandwidth_gbps(0, 10.0), 0.0)
        self.assertEqual(calculate_effective_bandwidth_gbps(1000, 0.0), 0.0)
        self.assertEqual(calculate_effective_bandwidth_gbps(-100, 10.0), 0.0)
        self.assertEqual(calculate_effective_bandwidth_gbps(1000, -5.0), 0.0)


class TestDisaggregatedPDSimulatorBasic(unittest.TestCase):
    """Tests for discrete-event simulator core logic, queue decoupling, and output structure."""

    def test_simulation_dictionary_keys(self):
        """Verify that simulation outputs all required performance and health keys."""
        res = simulate_cluster_serving(is_disaggregated=True, num_requests=10)
        expected_keys = [
            "Architecture",
            "P50_TTFT_ms",
            "P99_TTFT_ms",
            "P50_TPOT_ms",
            "P99_TPOT_ms",
            "P50_ITL_ms",
            "P99_ITL_ms",
            "Hardware_Efficiency",
            "Requests_Completed",
            "Requests_Dropped"
        ]
        for k in expected_keys:
            self.assertIn(k, res, f"Result must contain key {k}")

        self.assertEqual(res["P50_TPOT_ms"], res["P50_ITL_ms"])
        self.assertEqual(res["P99_TPOT_ms"], res["P99_ITL_ms"])
        self.assertEqual(res["Requests_Completed"], 10)
        self.assertEqual(res["Requests_Dropped"], 0)

    def test_coupled_vs_disaggregated_comparison(self):
        """Verify the central thesis: disaggregated serving reduces tail TTFT and eliminates decode jitter."""
        mono = simulate_cluster_serving(is_disaggregated=False, num_requests=100, random_seed=42)
        disagg = simulate_cluster_serving(is_disaggregated=True, num_requests=100, random_seed=42)

        # TTFT tail latency should be significantly lower in disaggregated mode
        self.assertLess(
            disagg["P99_TTFT_ms"],
            mono["P99_TTFT_ms"],
            "Disaggregated P99 TTFT must be strictly better than collocated"
        )

        # TPOT/ITL tail jitter should be drastically reduced (no prefill stalls during decode)
        self.assertLess(
            disagg["P99_TPOT_ms"],
            mono["P99_TPOT_ms"],
            "Disaggregated P99 TPOT must eliminate prefill interference spikes"
        )
        self.assertLess(disagg["P99_TPOT_ms"], 30.0, "Disaggregated decode ITL should remain near base 18-25ms")

        # Hardware efficiency should be superior in disaggregated pools
        self.assertGreater(
            disagg["Hardware_Efficiency"],
            mono["Hardware_Efficiency"],
            "Disaggregated serving should deliver higher hardware efficiency"
        )
        self.assertEqual(disagg["Hardware_Efficiency"], 0.88)
        self.assertEqual(mono["Hardware_Efficiency"], 0.58)

    def test_percentile_monotonicity(self):
        """P50 must always be less than or equal to P99 for both TTFT and ITL."""
        for is_disagg in [False, True]:
            res = simulate_cluster_serving(is_disaggregated=is_disagg, num_requests=50, random_seed=123)
            self.assertLessEqual(
                res["P50_TTFT_ms"],
                res["P99_TTFT_ms"],
                "P50 TTFT must be <= P99 TTFT"
            )
            self.assertLessEqual(
                res["P50_TPOT_ms"],
                res["P99_TPOT_ms"],
                "P50 TPOT must be <= P99 TPOT"
            )

    def test_queue_empty_condition(self):
        """0 requests should cleanly return zeroed metrics without division by zero or index errors."""
        for is_disagg in [False, True]:
            res = simulate_cluster_serving(is_disaggregated=is_disagg, num_requests=0)
            self.assertEqual(res["P50_TTFT_ms"], 0.0)
            self.assertEqual(res["P99_TTFT_ms"], 0.0)
            self.assertEqual(res["P50_TPOT_ms"], 0.0)
            self.assertEqual(res["P99_TPOT_ms"], 0.0)
            self.assertEqual(res["P50_ITL_ms"], 0.0)
            self.assertEqual(res["P99_ITL_ms"], 0.0)
            self.assertEqual(res["Hardware_Efficiency"], 0.0)
            self.assertEqual(res["Requests_Completed"], 0)
            self.assertEqual(res["Requests_Dropped"], 0)

    def test_single_request_handling(self):
        """A single request should execute successfully with 1 TTFT sample and gen_len ITL samples."""
        res = simulate_cluster_serving(is_disaggregated=True, num_requests=1)
        self.assertEqual(res["Requests_Completed"], 1)
        # Single request has exactly 1 TTFT value, so P50 == P99
        self.assertEqual(res["P50_TTFT_ms"], res["P99_TTFT_ms"])
        # ITL has multiple token generation iterations (gen_len tokens), so P50 <= P99
        self.assertLessEqual(res["P50_TPOT_ms"], res["P99_TPOT_ms"])
        self.assertGreater(res["P50_TTFT_ms"], 0.0)
        self.assertGreater(res["P50_TPOT_ms"], 0.0)


class TestDisaggregatedPDBurstAndCongestion(unittest.TestCase):
    """Tests for heavy burst arrival loads, queue backlog growth, and decode isolation."""

    def test_heavy_burst_loads_surge_tail_latency(self):
        """Burst arrival (all requests arriving at t=0) causes queue backlog surge."""
        steady = simulate_cluster_serving(is_disaggregated=False, num_requests=50, burst_arrival=False, random_seed=7)
        burst = simulate_cluster_serving(is_disaggregated=False, num_requests=50, burst_arrival=True, random_seed=7)

        # Queue buildup in burst mode dramatically increases P99 TTFT
        self.assertGreater(
            burst["P99_TTFT_ms"],
            steady["P99_TTFT_ms"],
            "Burst load should significantly increase P99 TTFT due to queuing delay"
        )

    def test_burst_decode_jitter_isolation_in_disaggregated_mode(self):
        """In disaggregated mode, even under heavy burst, decode ITL remains protected."""
        steady_disagg = simulate_cluster_serving(is_disaggregated=True, num_requests=50, burst_arrival=False, random_seed=42)
        burst_disagg = simulate_cluster_serving(is_disaggregated=True, num_requests=50, burst_arrival=True, random_seed=42)

        # While TTFT increases due to prefill queue delay, decode TPOT remains bounded
        self.assertLess(
            burst_disagg["P99_TPOT_ms"],
            28.0,
            "Disaggregated decode ITL should remain isolated from prefill burst stalls"
        )
        self.assertAlmostEqual(burst_disagg["P50_TPOT_ms"], steady_disagg["P50_TPOT_ms"], delta=3.0)


class TestDisaggregatedPDWorkerFaultAndSaturation(unittest.TestCase):
    """Tests for worker node failures, reduced cluster capacity, and queue saturation."""

    def test_partial_prefill_worker_failure_degrades_ttft(self):
        """When 1 of 2 prefill workers fails, prefill capacity is halved, increasing queue delay."""
        healthy = simulate_cluster_serving(
            is_disaggregated=True,
            num_requests=50,
            num_prefill_workers=2,
            failed_prefill_workers=0,
            random_seed=42
        )
        degraded = simulate_cluster_serving(
            is_disaggregated=True,
            num_requests=50,
            num_prefill_workers=2,
            failed_prefill_workers=1,
            random_seed=42
        )
        self.assertEqual(healthy["Requests_Completed"], 50)
        self.assertEqual(degraded["Requests_Completed"], 50)
        # Halved prefill throughput causes higher queuing delay
        self.assertGreater(
            degraded["P99_TTFT_ms"],
            healthy["P99_TTFT_ms"],
            "Partial prefill node outage must increase tail TTFT"
        )
        # But decode ITL remains unaffected
        self.assertAlmostEqual(degraded["P50_TPOT_ms"], healthy["P50_TPOT_ms"], delta=2.0)

    def test_partial_decode_worker_failure(self):
        """When 1 of 2 decode workers fails in disaggregated mode, all requests still complete."""
        healthy = simulate_cluster_serving(
            is_disaggregated=True,
            num_requests=30,
            num_decode_workers=2,
            failed_decode_workers=0,
            random_seed=42
        )
        degraded = simulate_cluster_serving(
            is_disaggregated=True,
            num_requests=30,
            num_decode_workers=2,
            failed_decode_workers=1,
            random_seed=42
        )
        self.assertEqual(healthy["Requests_Completed"], 30)
        self.assertEqual(degraded["Requests_Completed"], 30)

    def test_rdma_transfer_latency_sensitivity(self):
        """Increasing RDMA KV handoff latency shifts disaggregated TTFT by that exact duration."""
        res_fast_rdma = simulate_cluster_serving(
            is_disaggregated=True,
            num_requests=20,
            rdma_transfer_latency_ms=0.1,
            random_seed=42
        )
        res_slow_rdma = simulate_cluster_serving(
            is_disaggregated=True,
            num_requests=20,
            rdma_transfer_latency_ms=10.0,
            random_seed=42
        )
        # Shift in P50 and P99 TTFT should be roughly the 9.9 ms difference
        diff_p50 = res_slow_rdma["P50_TTFT_ms"] - res_fast_rdma["P50_TTFT_ms"]
        self.assertAlmostEqual(diff_p50, 9.9, delta=0.5)

    def test_random_seed_reproducibility(self):
        """Identical random seeds yield deterministic simulation output."""
        res1 = simulate_cluster_serving(is_disaggregated=True, num_requests=30, random_seed=999)
        res2 = simulate_cluster_serving(is_disaggregated=True, num_requests=30, random_seed=999)
        self.assertEqual(res1, res2)

    def test_total_worker_failure_raises_runtime_error(self):
        """If all workers in any required pool fail, simulation must raise RuntimeError."""
        # Total collocated worker failure
        with self.assertRaises(RuntimeError):
            simulate_cluster_serving(
                is_disaggregated=False,
                num_collocated_workers=4,
                failed_collocated_workers=4
            )

        # Total prefill worker failure
        with self.assertRaises(RuntimeError):
            simulate_cluster_serving(
                is_disaggregated=True,
                num_prefill_workers=2,
                failed_prefill_workers=2
            )

        # Total decode worker failure
        with self.assertRaises(RuntimeError):
            simulate_cluster_serving(
                is_disaggregated=True,
                num_decode_workers=2,
                failed_decode_workers=2
            )

    def test_queue_saturation_backpressure_drops_excess_requests(self):
        """Setting a bounded max_queue_size drops requests that arrive when queue is full."""
        num_reqs = 100
        # Restrict queue size to 5 requests per worker
        res = simulate_cluster_serving(
            is_disaggregated=False,
            num_requests=num_reqs,
            burst_arrival=True,
            max_queue_size=5,
            num_collocated_workers=4
        )
        self.assertGreater(res["Requests_Dropped"], 0, "Burst requests should overflow bounded queue")
        self.assertEqual(res["Requests_Completed"] + res["Requests_Dropped"], num_reqs)
        # Total completed cannot exceed max_queue_size * num_workers (plus active batch)
        self.assertLessEqual(res["Requests_Completed"], 5 * 4 + 4)

    def test_invalid_parameters_raise_value_error(self):
        """Verify invalid worker configurations and negative request counts raise ValueError."""
        with self.assertRaises(ValueError):
            simulate_cluster_serving(is_disaggregated=True, num_requests=-1)

        with self.assertRaises(ValueError):
            simulate_cluster_serving(is_disaggregated=True, num_prefill_workers=0)

        with self.assertRaises(ValueError):
            simulate_cluster_serving(is_disaggregated=True, num_decode_workers=-2)

        with self.assertRaises(ValueError):
            simulate_cluster_serving(is_disaggregated=False, num_collocated_workers=0)

        with self.assertRaises(ValueError):
            simulate_cluster_serving(is_disaggregated=True, failed_prefill_workers=-1)

        with self.assertRaises(ValueError):
            simulate_cluster_serving(is_disaggregated=True, max_queue_size=0)


class TestServingRequestDataclass(unittest.TestCase):
    """Tests for ServingRequest state tracking, TTFT, and ITL computation."""

    def test_request_lifecycle(self):
        """Verify TTFT and ITL computation properties on request."""
        req = ServingRequest(
            req_id="test-1",
            prompt_len=2048,
            gen_len=64,
            arrival_ms=10.0,
            remaining_prefill=2048,
            remaining_decode=64
        )
        self.assertEqual(req.ttft_ms, 0.0)
        self.assertEqual(req.avg_itl_ms, 0.0)

        # Set prefill end
        req.prefill_end_ms = 45.0
        self.assertEqual(req.ttft_ms, 35.0)

        # Set decode start (after RDMA transfer)
        req.decode_start_ms = 45.1
        self.assertEqual(req.ttft_ms, 35.1)

        # Add ITL steps
        req.itls_ms = [18.2, 19.1, 18.5, 20.0]
        self.assertAlmostEqual(req.avg_itl_ms, 18.95, places=2)


class TestChapterIntegration(unittest.TestCase):
    """Integration test executing Chapter 6 scripts directly to verify standalone CLI execution."""

    def test_disaggregated_pd_simulator_cli_execution(self):
        """Execute disaggregated_pd_simulator.py as a subprocess; expect returncode 0."""
        script_path = os.path.join(CHAPTER_DIR, "disaggregated_pd_simulator.py")
        res = subprocess.run([sys.executable, script_path], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(res.returncode, 0, f"Script failed with stderr:\n{res.stderr}")
        self.assertIn("CHAPTER 6: MONOLITHIC VS DISAGGREGATED SERVING", res.stdout)
        self.assertIn("Collocated Mode", res.stdout)
        self.assertIn("Disaggregated Mode", res.stdout)
        self.assertIn("Disaggregation isolates tail latency", res.stdout)

    def test_rdma_kv_transfer_cli_execution(self):
        """Execute rdma_kv_transfer.py as a subprocess; expect returncode 0."""
        script_path = os.path.join(CHAPTER_DIR, "rdma_kv_transfer.py")
        res = subprocess.run([sys.executable, script_path], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(res.returncode, 0, f"Script failed with stderr:\n{res.stderr}")
        self.assertIn("KV Cache Payload", res.stdout)
        self.assertIn("RDMA Transfer Time over 400Gbps RoCEv2", res.stdout)


if __name__ == "__main__":
    unittest.main()
