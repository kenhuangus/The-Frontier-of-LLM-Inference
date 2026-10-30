"""
Unit Test Suite for Chapter 2: KV Cache Optimization & Multi-Head Latent Attention (MLA)
The Frontier of LLM Inference (2026 Edition)

Validates:
1. RadixAttention Prefix Caching Engine:
   - Zero-length token sequences
   - Exact prefix match & block retrieval
   - Partial prefix match & dynamic node splitting
   - Deeply nested token tree structures
   - Reference counting & safe memory pinning
   - Eviction under tight capacity limit (LRU policy)
   - Shared prefix memory recycling
2. Multi-Head Latent Attention (MLA) Simulation:
   - Low-rank KV projection equations (W_DKV, W_UK)
   - Decoupled RoPE dimension handling (W_KR, k_t^R)
   - Matrix absorption query transformation (q_t W^UK)
   - Attention score calculation with absorbed content and decoupled RoPE
   - VRAM savings percentage vs standard MHA (98.24%) and GQA (71.88%)
   - Autoregressive decode FLOPs scaling & cache reset
3. KV Cache Architecture Benchmark:
   - Per-token byte calculation across MHA, GQA, and MLA architectures
   - Multi-context VRAM sizing across 4k, 32k, 128k, and 1M tokens
   - Memory-bound decode latency calculation under GPU HBM bandwidth
   - Autoregressive generation throughput (tokens/second)
"""
import os
import sys
import math
import unittest
from typing import List

# Ensure chapter directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from radix_prefix_cache import (
    RadixTreeNode,
    RadixAttentionEngine
)
from mla_simulation import (
    MLAModelConfig,
    MLALayerSimulation
)
from kv_cache_benchmark import (
    ArchConfig,
    compute_token_kv_bytes,
    compute_context_vram_gb,
    calculate_decode_latency_ms,
    calculate_generation_throughput,
    run_benchmark
)


class TestRadixPrefixCache(unittest.TestCase):
    """Tests for SGLang RadixAttention deterministic prefix cache engine."""

    def setUp(self):
        self.engine = RadixAttentionEngine(total_gpu_blocks=100, block_size=16)

    def test_zero_length_token_sequence(self):
        """Zero-length token sequences should return empty blocks and not crash."""
        matched_blocks, traversed = self.engine.match_prefix([])
        self.assertEqual(matched_blocks, [])
        self.assertEqual(len(traversed), 1)
        self.assertEqual(traversed[0], self.engine.root)
        self.assertEqual(self.engine.stats["cache_misses"], 1)

        allocated = self.engine.insert_sequence([])
        self.assertEqual(allocated, [])
        self.assertEqual(len(self.engine.root.children), 0)

    def test_exact_prefix_match(self):
        """Exact matching of previously inserted sequence."""
        tokens = list(range(100, 148))  # 48 tokens -> 3 blocks (16 each)
        allocated = self.engine.insert_sequence(tokens)
        self.assertEqual(len(allocated), 3)

        matched_blocks, traversed = self.engine.match_prefix(tokens)
        self.assertEqual(matched_blocks, allocated)
        self.assertEqual(self.engine.stats["cache_hits"], 1)
        self.assertEqual(len(traversed), 2)  # [root, child]

    def test_partial_prefix_match_splits_node(self):
        """Partial prefix match should split node and return matched prefix blocks."""
        # Insert 32 tokens (2 blocks)
        seq1 = list(range(100, 132))
        alloc1 = self.engine.insert_sequence(seq1)
        self.assertEqual(len(alloc1), 2)

        # Query with sequence sharing first 16 tokens but diverging at token 16
        seq2 = list(range(100, 116)) + [999, 1000]
        matched_blocks, traversed = self.engine.match_prefix(seq2)

        # First 16 tokens match -> 1 block
        self.assertEqual(len(matched_blocks), 1)
        self.assertEqual(matched_blocks, alloc1[:1])
        self.assertEqual(self.engine.stats["cache_hits"], 1)

        # Verify tree split: root -> split_node (16 tokens) -> child (remaining 16 tokens)
        split_node = traversed[-1]
        self.assertEqual(len(split_node.token_ids), 16)
        self.assertEqual(split_node.token_ids, list(range(100, 116)))
        self.assertIn(116, split_node.children)

    def test_insert_with_partial_overlap_branching(self):
        """Inserting two sequences with shared prefix creates branching radix tree."""
        sys_prompt = list(range(100, 300))  # 200 tokens (13 blocks)
        req1 = sys_prompt + list(range(1000, 1050))  # 250 tokens (16 blocks)
        req2 = sys_prompt + list(range(2000, 2080))  # 280 tokens

        alloc1 = self.engine.insert_sequence(req1)
        self.assertEqual(len(alloc1), 16)

        m2, traversed2 = self.engine.match_prefix(req2)
        self.assertEqual(len(m2), 13)  # ceil(200 / 16) = 13 blocks matched
        self.assertEqual(self.engine.stats["cache_hits"], 1)

        alloc2 = self.engine.insert_sequence(req2)
        # 80 new tokens -> ceil(80 / 16) = 5 new blocks allocated
        self.assertEqual(len(alloc2), 5)

        # Verify both branches are retrievable
        m1_check, _ = self.engine.match_prefix(req1)
        self.assertGreaterEqual(len(m1_check), 16)

        m2_check, _ = self.engine.match_prefix(req2)
        self.assertGreaterEqual(len(m2_check), 18)

    def test_deeply_nested_token_trees(self):
        """Inserting sequences with successively deeper prefixes builds multi-level hierarchy."""
        sequences = [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 9, 10],
            [1, 2, 3, 4, 11, 12],
            [1, 2, 13, 14],
            [1, 15, 16],
            [20, 21, 22]  # Disjoint root branch
        ]
        for seq in sequences:
            self.engine.insert_sequence(seq)

        for seq in sequences:
            matched, _ = self.engine.match_prefix(seq)
            self.assertGreater(len(matched), 0, f"Failed to match seq: {seq}")

        # Total cached tokens verification across compressed radix nodes:
        # [1] (1), [2] (1), [3, 4] (2), [5, 6] (2), [7, 8] (2), [9, 10] (2),
        # [11, 12] (2), [13, 14] (2), [15, 16] (2), [20, 21, 22] (3) = 19 unique tokens
        total_tokens = self.engine.get_total_cached_tokens()
        self.assertEqual(total_tokens, 19)

    def test_reference_counting_and_release(self):
        """Reference counting properly increments on insert and decrements on release."""
        seq = [10, 20, 30, 40]
        self.engine.insert_sequence(seq)
        node = self.engine.root.children[10]
        self.assertEqual(node.ref_count, 1)

        self.engine.inc_ref_count(node)
        self.assertEqual(node.ref_count, 2)

        self.engine.dec_ref_count(node)
        self.assertEqual(node.ref_count, 1)

        self.engine.dec_ref_count(node)
        self.assertEqual(node.ref_count, 0)

        # Does not decrement below zero
        self.engine.dec_ref_count(node)
        self.assertEqual(node.ref_count, 0)

        # Release sequence helper
        self.engine.inc_ref_count(node)
        released = self.engine.release_sequence(seq)
        self.assertEqual(released, 1)
        self.assertEqual(node.ref_count, 0)

    def test_eviction_under_tight_capacity_limit(self):
        """Cache correctly evicts unreferenced LRU nodes when GPU blocks run out."""
        # 4 total blocks, each sequence of 16 tokens takes 1 block
        tight_engine = RadixAttentionEngine(total_gpu_blocks=4, block_size=16)

        seqs = [
            [10 + i] * 16 for i in range(4)
        ]
        for seq in seqs:
            tight_engine.insert_sequence(seq)

        self.assertEqual(len(tight_engine.free_blocks), 0)

        # Inserting a 5th sequence when all nodes have ref_count=1 must raise OOM
        seq5 = [99] * 16
        with self.assertRaises(RuntimeError):
            tight_engine.insert_sequence(seq5)

        # Release the first sequence (ref_count -> 0)
        tight_engine.release_sequence(seqs[0])

        # Inserting seq5 now succeeds by evicting seqs[0]
        alloc5 = tight_engine.insert_sequence(seq5)
        self.assertEqual(len(alloc5), 1)
        self.assertEqual(tight_engine.stats["evicted_nodes"], 1)
        self.assertEqual(len(tight_engine.free_blocks), 0)

    def test_shared_prefix_memory_recycling(self):
        """Evicting a leaf preserves shared prefix node as long as other children exist."""
        shared_prefix = [1, 2, 3, 4]
        seq_a = shared_prefix + [10, 11]
        seq_b = shared_prefix + [20, 21]

        self.engine.insert_sequence(seq_a)
        self.engine.insert_sequence(seq_b)

        # Release seq_a
        self.engine.release_sequence(seq_a)

        # Evict LRU
        evicted = self.engine._evict_lru_node()
        self.assertTrue(evicted)
        self.assertEqual(self.engine.stats["evicted_nodes"], 1)

        # seq_b must still be 100% intact and matchable
        matched_b, _ = self.engine.match_prefix(seq_b)
        self.assertGreater(len(matched_b), 0)

        # Now release seq_b as well
        self.engine.release_sequence(seq_b)

        # Both remaining leaf (seq_b suffix) and then the shared prefix can be evicted
        evicted2 = self.engine._evict_lru_node()
        self.assertTrue(evicted2)

        evicted3 = self.engine._evict_lru_node()
        self.assertTrue(evicted3)

    def test_lru_timestamp_eviction_order(self):
        """Nodes are evicted strictly in least-recently-accessed order."""
        engine = RadixAttentionEngine(total_gpu_blocks=10, block_size=16)
        seq1 = [100] * 16
        seq2 = [200] * 16
        engine.insert_sequence(seq1)
        engine.insert_sequence(seq2)

        # Release both
        engine.release_sequence(seq1)
        engine.release_sequence(seq2)

        node1 = engine.root.children[100]
        node2 = engine.root.children[200]

        # Set explicit timestamps: node1 older than node2
        node1.last_access_time = 1000.0
        node2.last_access_time = 2000.0

        # First eviction must pick node1
        engine._evict_lru_node()
        self.assertNotIn(100, engine.root.children)
        self.assertIn(200, engine.root.children)

        # Second eviction must pick node2
        engine._evict_lru_node()
        self.assertNotIn(200, engine.root.children)


class TestMLASimulation(unittest.TestCase):
    """Tests for DeepSeek Multi-Head Latent Attention (MLA) mechanics and VRAM savings."""

    def setUp(self):
        self.cfg = MLAModelConfig()
        self.mla = MLALayerSimulation(self.cfg)

    def test_model_config_defaults(self):
        """Verify DeepSeek-V4-Pro architectural baseline configuration."""
        self.assertEqual(self.cfg.d_model, 7168)
        self.assertEqual(self.cfg.n_heads, 128)
        self.assertEqual(self.cfg.d_head, 128)
        self.assertEqual(self.cfg.n_layers, 60)
        self.assertEqual(self.cfg.d_c, 512)
        self.assertEqual(self.cfg.d_h_r, 64)
        self.assertEqual(self.cfg.bytes_per_elem, 2.0)

    def test_token_kv_bytes_mla(self):
        """MLA caches only (d_c + d_h_r) * bytes per token per layer."""
        expected_bytes = (512 + 64) * 2  # 1,152 bytes
        self.assertEqual(self.mla.compute_token_kv_bytes(), expected_bytes)
        self.assertEqual(self.mla.compute_token_kv_bytes(), 1152)

    def test_token_kv_bytes_standard_mha(self):
        """Standard MHA caches 2 * n_heads * d_head * bytes per token per layer."""
        expected_mha = 2 * 128 * 128 * 2  # 65,536 bytes (64 KB)
        self.assertEqual(self.mla.compute_mha_token_kv_bytes(), expected_mha)
        self.assertEqual(self.mla.compute_mha_token_kv_bytes(), 65536)

    def test_vram_savings_vs_mha(self):
        """MLA achieves ~98.24% reduction in KV cache memory compared to standard MHA."""
        savings = self.mla.compute_vram_savings_vs_mha()
        expected = ((65536 - 1152) / 65536) * 100.0  # 98.2421875%
        self.assertAlmostEqual(savings, expected, places=4)
        self.assertAlmostEqual(savings, 98.24, places=2)

    def test_vram_savings_vs_gqa(self):
        """MLA achieves ~71.88% reduction in KV cache memory compared to GQA-8."""
        gqa_bytes = self.mla.compute_gqa_token_kv_bytes(n_kv_heads=8)
        expected_gqa = 2 * 8 * 128 * 2  # 4,096 bytes
        self.assertEqual(gqa_bytes, expected_gqa)

        savings = self.mla.compute_vram_savings_vs_gqa(n_kv_heads=8)
        expected_savings = ((4096 - 1152) / 4096) * 100.0  # 71.875%
        self.assertAlmostEqual(savings, expected_savings, places=3)
        self.assertAlmostEqual(savings, 71.875, places=2)

    def test_low_rank_projection_equation(self):
        """Validate low-rank projection equation c_t^{KV} = W^{DKV} h_t."""
        d_model = self.cfg.d_model
        d_c = self.cfg.d_c
        hidden_state = [1.0] * d_model
        # Simple projection matrix with constant 0.001
        W_DKV = [[0.001] * d_model for _ in range(d_c)]
        c_kv = self.mla.project_latent_kv(hidden_state, W_DKV)

        self.assertEqual(len(c_kv), d_c)
        expected_val = round(0.001 * d_model, 4)  # 7.168
        self.assertAlmostEqual(c_kv[0], expected_val, places=3)

    def test_decoupled_rope_dimension_handling(self):
        """Validate decoupled RoPE projection k_t^R = W^{KR} h_t."""
        d_model = self.cfg.d_model
        d_h_r = self.cfg.d_h_r
        hidden_state = [1.0] * d_model
        W_KR = [[0.002] * d_model for _ in range(d_h_r)]
        k_rope = self.mla.project_rope_key(hidden_state, W_KR)

        self.assertEqual(len(k_rope), d_h_r)
        expected_val = round(0.002 * d_model, 4)  # 14.336
        self.assertAlmostEqual(k_rope[0], expected_val, places=3)

    def test_matrix_absorption_equation(self):
        """Validate query absorption q_{t, absorbed} = q_t W^{UK} to avoid decompressing c_t^{KV}."""
        d_head = self.cfg.d_head
        d_c = self.cfg.d_c
        query_head = [1.0] * d_head
        W_UK = [[0.01] * d_c for _ in range(d_head)]

        q_absorbed = self.mla.absorb_query_matrix(query_head, W_UK)
        self.assertEqual(len(q_absorbed), d_c)
        expected_val = round(0.01 * d_head, 4)  # 1.28
        self.assertAlmostEqual(q_absorbed[0], expected_val, places=3)

    def test_absorbed_attention_score_computation(self):
        """Validate attention score combining absorbed content and decoupled RoPE keys."""
        d_c = self.cfg.d_c
        d_h_r = self.cfg.d_h_r
        q_abs = [0.1] * d_c
        c_kv = [0.2] * d_c
        q_rope = [0.05] * d_h_r
        k_rope = [0.05] * d_h_r

        score = self.mla.compute_absorbed_attention_score(q_abs, q_rope, c_kv, k_rope)
        content = 0.1 * 0.2 * d_c  # 0.02 * 512 = 10.24
        position = 0.05 * 0.05 * d_h_r  # 0.0025 * 64 = 0.16
        scale = math.sqrt(128 + 64)
        expected = (content + position) / scale
        self.assertAlmostEqual(score, expected, places=4)

    def test_decode_step_flops_scaling_and_reset(self):
        """Autoregressive decode step FLOPs scale linearly with cached history."""
        self.mla.reset()
        self.assertEqual(len(self.mla.cached_latent_kv), 0)

        # Step 0: num_cached = 0 -> 2 * 1 * (512 + 64) * 128 = 147,456 FLOPs
        flops_0 = self.mla.decode_step(0)
        self.assertEqual(flops_0, 147456.0)
        self.assertEqual(len(self.mla.cached_latent_kv), 1)

        # Step 1: num_cached = 1 -> 2 * 2 * 576 * 128 = 294,912 FLOPs
        flops_1 = self.mla.decode_step(1)
        self.assertEqual(flops_1, 294912.0)
        self.assertEqual(len(self.mla.cached_latent_kv), 2)

        self.mla.reset()
        self.assertEqual(len(self.mla.cached_latent_kv), 0)
        self.assertEqual(len(self.mla.cached_rope_keys), 0)

    def test_dimension_validation_errors(self):
        """Projection methods must validate dimension mismatches."""
        with self.assertRaises(ValueError):
            self.mla.project_latent_kv([1.0] * 10, [[1.0] * 7168 for _ in range(512)])
        with self.assertRaises(ValueError):
            self.mla.project_rope_key([1.0] * 10, [[1.0] * 7168 for _ in range(64)])
        with self.assertRaises(ValueError):
            self.mla.absorb_query_matrix([1.0] * 10, [[1.0] * 512 for _ in range(128)])


class TestKVCacheBenchmark(unittest.TestCase):
    """Tests for multi-architecture KV cache memory, latency, and throughput benchmark."""

    def setUp(self):
        self.mha_model = ArchConfig("Llama-1 65B (MHA)", n_layers=80, n_heads=64, n_kv_heads=64, d_head=128)
        self.gqa_model = ArchConfig("Llama-3 70B (GQA-8)", n_layers=80, n_heads=64, n_kv_heads=8, d_head=128)
        self.gqa_405b = ArchConfig("Llama-3 405B (GQA-8)", n_layers=126, n_heads=128, n_kv_heads=8, d_head=128)
        self.mla_model = ArchConfig("DeepSeek-V3/V4 (MLA)", n_layers=60, n_heads=128, n_kv_heads=128, d_head=128, is_mla=True, d_c=512, d_h_r=64)

    def test_token_kv_bytes_calculations(self):
        """Verify token KV cache byte formulas across all architectures."""
        # Llama-1 65B: 2 * 80 * 64 * 128 * 2 = 2,621,440 B (2560 KB)
        mha_bytes = compute_token_kv_bytes(self.mha_model)
        self.assertEqual(mha_bytes, 2621440)

        # Llama-3 70B: 2 * 80 * 8 * 128 * 2 = 327,680 B (320 KB)
        gqa_bytes = compute_token_kv_bytes(self.gqa_model)
        self.assertEqual(gqa_bytes, 327680)

        # Llama-3 405B: 2 * 126 * 8 * 128 * 2 = 516,096 B (504 KB)
        gqa_405b_bytes = compute_token_kv_bytes(self.gqa_405b)
        self.assertEqual(gqa_405b_bytes, 516096)

        # DeepSeek MLA: (512 + 64) * 60 * 2 = 69,120 B (67.5 KB)
        mla_bytes = compute_token_kv_bytes(self.mla_model)
        self.assertEqual(mla_bytes, 69120)

    def test_context_vram_gb_scaling(self):
        """Verify context scaling in GiB across 4k, 32k, 128k, and 1M tokens."""
        # Llama-1 65B at 4k: (2621440 * 4096) / (1024^3) = 10.0 GiB
        vram_4k = compute_context_vram_gb(self.mha_model, 4096)
        self.assertAlmostEqual(vram_4k, 10.0, places=2)

        # Llama-3 70B at 32k: (327680 * 32768) / (1024^3) = 10.0 GiB
        vram_32k = compute_context_vram_gb(self.gqa_model, 32768)
        self.assertAlmostEqual(vram_32k, 10.0, places=2)

        # DeepSeek MLA at 128k: (69120 * 131072) / (1024^3) = 8.4375 GiB
        vram_128k = compute_context_vram_gb(self.mla_model, 131072)
        self.assertAlmostEqual(vram_128k, 8.44, places=2)

        # DeepSeek MLA at 1M: (69120 * 1048576) / (1024^3) = 67.5 GiB
        vram_1m = compute_context_vram_gb(self.mla_model, 1048576)
        self.assertAlmostEqual(vram_1m, 67.50, places=2)

    def test_decode_latency_and_throughput(self):
        """Verify decode latency and throughput calculations under H100 SXM5 bandwidth (3.35 TB/s)."""
        ctx = 131072  # 128k context
        bw = 3.35     # 3.35 TB/s

        # Llama-1 65B (MHA): total bytes = 2621440 * 131072 = 343,597,383,680 B (~343.6 GB)
        # Latency = 343.6 GB / 3.35 TB/s = ~102.57 ms
        lat_mha = calculate_decode_latency_ms(self.mha_model, context_len=ctx, memory_bandwidth_tb_s=bw)
        thr_mha = calculate_generation_throughput(self.mha_model, context_len=ctx, memory_bandwidth_tb_s=bw)
        self.assertAlmostEqual(lat_mha, 102.57, delta=0.5)
        self.assertAlmostEqual(thr_mha, 9.7, delta=0.5)

        # DeepSeek MLA: total bytes = 69120 * 131072 = 9,059,696,640 B (~9.06 GB)
        # Latency = 9.06 GB / 3.35 TB/s = ~2.70 ms
        lat_mla = calculate_decode_latency_ms(self.mla_model, context_len=ctx, memory_bandwidth_tb_s=bw)
        thr_mla = calculate_generation_throughput(self.mla_model, context_len=ctx, memory_bandwidth_tb_s=bw)
        self.assertAlmostEqual(lat_mla, 2.70, delta=0.2)
        self.assertAlmostEqual(thr_mla, 369.8, delta=2.0)

        # Speedup ratio: MLA provides ~38x throughput improvement over MHA at 128k context
        speedup = thr_mla / thr_mha
        self.assertAlmostEqual(speedup, 38.0, delta=1.0)

    def test_batch_size_scaling(self):
        """Latency scales linearly with batch size while generation throughput remains constant."""
        lat_b1 = calculate_decode_latency_ms(self.mla_model, context_len=4096, batch_size=1)
        lat_b4 = calculate_decode_latency_ms(self.mla_model, context_len=4096, batch_size=4)
        self.assertAlmostEqual(lat_b4, lat_b1 * 4, places=4)

        thr_b1 = calculate_generation_throughput(self.mla_model, context_len=4096, batch_size=1)
        thr_b4 = calculate_generation_throughput(self.mla_model, context_len=4096, batch_size=4)
        self.assertAlmostEqual(thr_b1, thr_b4, places=4)

    def test_benchmark_runner_executes(self):
        """run_benchmark() runs and completes without exception."""
        try:
            run_benchmark()
        except Exception as e:
            self.fail(f"run_benchmark() raised exception: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
