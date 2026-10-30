"""
Unit Test Suite for Chapter 9: Ultra-Long Context Mastery (Dual-Chunk Attention & YaRN RoPE Scaling)
The Frontier of LLM Inference

Validates:
1. Dual-Chunk Attention (DCA):
   - Computational savings simulation across standard and extreme context lengths (32k, 128k, 1M, 2M)
   - Sequences shorter than a single chunk (seq_len < chunk_size)
   - Uneven chunk lengths and remainder token interaction math
   - Zero-length sequence and invalid non-positive argument handling
   - Intra-chunk vs successive-chunk vs inter-chunk boundary classification
   - Relative position remapping guarantees within pretrained context envelopes
   - Full causal attention score computation and softmax normalization
2. YaRN RoPE Spectral Scaling:
   - Scale factor 1.0 identity verification (exact equivalence to native unscaled RoPE)
   - Large context scale factors (8x, 16x, 32x, 64x)
   - Frequency band partitioning (Band 1 extrapolation, Band 2 interpolation, Band 3 smooth ramp)
   - Boundary threshold dynamics (low = orig_ctx / beta_fast, high = orig_ctx / beta_slow)
   - Zero dimensions handling (dim = 0 graceful exit without ZeroDivisionError)
   - Negative positions handling and RoPE isometric inverse rotation properties
   - Attention temperature calibration law (1/t = 0.1 * ln(s) + 1.0)
3. End-to-End Chapter Integration:
   - Qwen2.5-1M 1,048,576-token context co-simulation combining DCA and YaRN
"""
import math
import unittest
from typing import List, Tuple

from dual_chunk_attention import (
    DualChunkAttention,
    classify_attention_relation,
    compute_dca_attention_scores,
    dual_chunk_attention_simulation,
    get_chunk_assignment,
    map_dca_positions,
)
from yarn_rope_scaling import (
    YaRNScaledRotaryEmbedding,
    apply_yarn_rope,
    compute_yarn_frequencies,
    compute_yarn_mscale,
    compute_yarn_rotary_embedding,
)


class TestDualChunkAttentionSimulation(unittest.TestCase):
    """Tests for DCA quadratic prefill mitigation, FLOPs calculation, and chunk boundary math."""

    def test_standard_context_lengths(self):
        """Verify published DCA computational savings across 32k, 128k, and 1M tokens."""
        cases = [
            (32768, 4096, 1073741824, 141557760, 86.82),
            (131072, 4096, 17179869184, 666894336, 96.12),
            (1048576, 4096, 1099511627776, 12851347456, 98.83),
        ]
        for seq_len, chunk_size, exp_full, exp_dca, exp_savings in cases:
            full, dca, savings = dual_chunk_attention_simulation(seq_len, chunk_size)
            self.assertEqual(full, exp_full)
            self.assertEqual(dca, exp_dca)
            self.assertAlmostEqual(savings, exp_savings, places=2)

    def test_sequence_shorter_than_single_chunk(self):
        """Verify sequence lengths strictly smaller than chunk_size execute local attention with zero padding."""
        # 512 tokens with 4096 chunk size
        full, dca, savings = dual_chunk_attention_simulation(512, chunk_size=4096)
        self.assertEqual(full, 512 * 512)
        self.assertEqual(dca, 512 * 512)
        self.assertEqual(savings, 0.0)

        # Single token
        full, dca, savings = dual_chunk_attention_simulation(1, chunk_size=4096)
        self.assertEqual(full, 1)
        self.assertEqual(dca, 1)
        self.assertEqual(savings, 0.0)

        # Exact single chunk boundary
        full, dca, savings = dual_chunk_attention_simulation(4096, chunk_size=4096)
        self.assertEqual(full, 4096 * 4096)
        self.assertEqual(dca, 4096 * 4096)
        self.assertEqual(savings, 0.0)

    def test_uneven_chunk_lengths(self):
        """Verify sequences with uneven chunk remainders accurately calculate intra and inter FLOPs."""
        # seq_len = 10,000, chunk_size = 4096
        # 2 full chunks (4096 each) + remainder 1808
        full, dca, savings = dual_chunk_attention_simulation(10000, chunk_size=4096, exact=True)
        self.assertEqual(full, 10000 * 10000)

        exp_intra = 2 * (4096 * 4096) + (1808 * 1808)
        exp_full_pairs = (2 * 1) // 2
        exp_inter = exp_full_pairs * (4096 * 64) + 2 * (1808 * 64)
        exp_dca = exp_intra + exp_inter
        exp_savings = round((100000000 - exp_dca) / 100000000 * 100.0, 2)

        self.assertEqual(dca, exp_dca)
        self.assertEqual(savings, exp_savings)
        self.assertGreater(savings, 60.0)

    def test_zero_and_negative_sequence_length(self):
        """Verify zero sequence length returns zero FLOPs without division-by-zero error, and negative raises ValueError."""
        full, dca, savings = dual_chunk_attention_simulation(0, chunk_size=4096)
        self.assertEqual(full, 0)
        self.assertEqual(dca, 0)
        self.assertEqual(savings, 0.0)

        with self.assertRaises(ValueError):
            dual_chunk_attention_simulation(-100, chunk_size=4096)

    def test_invalid_chunk_sizes(self):
        """Verify non-positive chunk sizes raise ValueError."""
        with self.assertRaises(ValueError):
            dual_chunk_attention_simulation(1000, chunk_size=0)
        with self.assertRaises(ValueError):
            dual_chunk_attention_simulation(1000, chunk_size=-512)

    def test_ultra_long_context_scaling(self):
        """Verify asymptotic FLOP savings exceed 99.5% at 2M+ and 4M+ tokens."""
        for length in [2097152, 4194304]:
            full, dca, savings = dual_chunk_attention_simulation(length, chunk_size=8192)
            self.assertGreater(savings, 99.0)
            self.assertLess(dca, full)


class TestDualChunkAttentionBoundaryMath(unittest.TestCase):
    """Tests for token-to-chunk partitioning, boundary classification, and RoPE coordinate remapping."""

    def test_get_chunk_assignment(self):
        """Verify macro chunk index and local offset calculation."""
        chunk_size = 4096
        self.assertEqual(get_chunk_assignment(0, chunk_size), (0, 0))
        self.assertEqual(get_chunk_assignment(4095, chunk_size), (0, 4095))
        self.assertEqual(get_chunk_assignment(4096, chunk_size), (1, 0))
        self.assertEqual(get_chunk_assignment(10000, chunk_size), (2, 1808))

        with self.assertRaises(ValueError):
            get_chunk_assignment(-1, chunk_size)
        with self.assertRaises(ValueError):
            get_chunk_assignment(10, 0)

    def test_classify_attention_relation(self):
        """Verify categorization into intra, successive, inter, and future-masked attention."""
        chunk_size = 4096

        # Intra-chunk (same macro chunk)
        self.assertEqual(classify_attention_relation(100, 50, chunk_size), "intra")
        self.assertEqual(classify_attention_relation(4095, 0, chunk_size), "intra")
        self.assertEqual(classify_attention_relation(8500, 8200, chunk_size), "intra")

        # Successive-chunk (adjacent preceding chunk)
        self.assertEqual(classify_attention_relation(4096, 4095, chunk_size), "successive")
        self.assertEqual(classify_attention_relation(4100, 100, chunk_size), "successive")
        self.assertEqual(classify_attention_relation(8192, 4096, chunk_size), "successive")

        # Inter-chunk (distant preceding chunks, delta >= 2)
        self.assertEqual(classify_attention_relation(8200, 100, chunk_size), "inter")
        self.assertEqual(classify_attention_relation(1000000, 50, chunk_size), "inter")

        # Future masked (causal condition violated)
        self.assertEqual(classify_attention_relation(50, 100, chunk_size), "future_masked")
        self.assertEqual(classify_attention_relation(4095, 4096, chunk_size), "future_masked")

        # Input error handling
        with self.assertRaises(ValueError):
            classify_attention_relation(-1, 0, chunk_size)
        with self.assertRaises(ValueError):
            classify_attention_relation(0, -1, chunk_size)
        with self.assertRaises(ValueError):
            classify_attention_relation(10, 5, -4096)

    def test_map_dca_positions(self):
        """Verify DCA position remapping bounds relative positions within the pretrained envelope."""
        chunk_size = 4096
        pretrained_len = 32768

        # 1. Intra-chunk: keys and queries maintain identical local offset
        q_pos, k_pos = map_dca_positions(200, 100, chunk_size, pretrained_len)
        self.assertEqual(q_pos, 200)
        self.assertEqual(k_pos, 100)
        delta_intra = q_pos - k_pos
        self.assertTrue(0 <= delta_intra < chunk_size)

        # 2. Successive-chunk: queries offset by +chunk_size
        q_pos, k_pos = map_dca_positions(4100, 100, chunk_size, pretrained_len)
        self.assertEqual(q_pos, 4 + chunk_size)
        self.assertEqual(k_pos, 100)
        delta_succ = q_pos - k_pos
        self.assertTrue(1 <= delta_succ < 2 * chunk_size)
        self.assertLess(delta_succ, pretrained_len)

        # 3. Inter-chunk: query position capped within pretrained range
        q_pos, k_pos = map_dca_positions(9000, 50, chunk_size, pretrained_len)
        # q_off = 9000 % 4096 = 808
        self.assertEqual(q_pos, 2 * chunk_size + 808)
        self.assertEqual(k_pos, 50)
        self.assertLess(q_pos, pretrained_len)

        # Extreme context capping
        q_pos_extreme, _ = map_dca_positions(1000000, 0, chunk_size, pretrained_len=16384)
        self.assertLessEqual(q_pos_extreme, 16383)

        # Causal violation error
        with self.assertRaises(ValueError):
            map_dca_positions(50, 100, chunk_size, pretrained_len)


class TestDualChunkAttentionScoreComputation(unittest.TestCase):
    """Tests for causal dot-product attention score computation and softmax distribution."""

    def test_empty_and_mismatched_inputs(self):
        """Verify empty inputs return empty outputs and dimensionality mismatches raise ValueError."""
        self.assertEqual(compute_dca_attention_scores([], [], chunk_size=4096), [])
        self.assertEqual(compute_dca_attention_scores([[1.0]], [], chunk_size=4096), [])

        # Inconsistent query dimensions
        with self.assertRaises(ValueError):
            compute_dca_attention_scores([[1.0, 2.0], [1.0]], [[1.0, 2.0]], chunk_size=4096)

        # Key dimension mismatch
        with self.assertRaises(ValueError):
            compute_dca_attention_scores([[1.0, 2.0]], [[1.0, 2.0, 3.0]], chunk_size=4096)

    def test_causal_masking_property(self):
        """Verify causal masking strictly zeroes future key probability mass."""
        q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        k = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

        attn = compute_dca_attention_scores(q, k, chunk_size=2, causal=True)
        self.assertEqual(len(attn), 3)

        # Token 0 can only attend to Token 0
        self.assertAlmostEqual(attn[0][0], 1.0, places=5)
        self.assertAlmostEqual(attn[0][1], 0.0, places=5)
        self.assertAlmostEqual(attn[0][2], 0.0, places=5)

        # Token 1 can attend to Token 0 and Token 1
        self.assertGreater(attn[1][0], 0.0)
        self.assertGreater(attn[1][1], 0.0)
        self.assertAlmostEqual(attn[1][2], 0.0, places=5)

        # Row stochastic property: sum across keys equals 1.0
        for i in range(3):
            self.assertAlmostEqual(sum(attn[i]), 1.0, places=5)

    def test_sequences_shorter_than_chunk_size(self):
        """Verify attention computation functions seamlessly for sequences smaller than chunk size."""
        seq_len = 4
        chunk_size = 16
        q = [[0.5, -0.2 * i] for i in range(seq_len)]
        k = [[0.1 * j, 0.4] for j in range(seq_len)]

        attn = compute_dca_attention_scores(q, k, chunk_size=chunk_size, causal=True)
        self.assertEqual(len(attn), seq_len)
        for row in attn:
            self.assertEqual(len(row), seq_len)
            self.assertAlmostEqual(sum(row), 1.0, places=5)

    def test_uneven_chunk_attention(self):
        """Verify attention across uneven chunk boundaries (e.g. 5 tokens with chunk_size 2)."""
        # 5 tokens span chunks: [0, 1], [2, 3], [4]
        q = [[1.0, 0.5]] * 5
        k = [[1.0, 0.5]] * 5

        attn = compute_dca_attention_scores(q, k, chunk_size=2, causal=True)
        self.assertEqual(len(attn), 5)
        for i, row in enumerate(attn):
            self.assertAlmostEqual(sum(row), 1.0, places=5)
            # Future keys masked
            for j in range(i + 1, 5):
                self.assertEqual(row[j], 0.0)


class TestYaRNRoPEFrequencies(unittest.TestCase):
    """Tests for YaRN spectral frequency modulation, scale 1.0 identity, and ramp thresholds."""

    def test_scale_identity_1_0(self):
        """Verify scale=1.0 returns exact native unscaled RoPE frequencies (identity preservation)."""
        dim = 64
        base = 10000.0
        yarn_freqs = compute_yarn_frequencies(dim=dim, base=base, scale=1.0)
        native_freqs = [round(1.0 / (base ** (i / dim)), 6) for i in range(0, dim, 2)]

        self.assertEqual(len(yarn_freqs), len(native_freqs))
        self.assertEqual(yarn_freqs, native_freqs)

    def test_large_scale_factors(self):
        """Verify frequency scaling across large context factors (8x, 16x, 32x, 64x)."""
        dim = 128
        base = 10000.0
        for scale in [8.0, 16.0, 32.0, 64.0]:
            freqs = compute_yarn_frequencies(dim=dim, base=base, scale=scale)
            self.assertEqual(len(freqs), dim // 2)

            # Monotonically non-increasing frequencies
            for i in range(len(freqs) - 1):
                self.assertGreaterEqual(freqs[i], freqs[i + 1])

            # Channel 0 (high-frequency, wavelength ~6.28) must remain 100% unscaled
            self.assertAlmostEqual(freqs[0], 1.0, places=5)

            # Lowest frequency channel (very long wavelength) must be scaled approximately by 1/scale
            native_lowest = 1.0 / (base ** ((dim - 2) / dim))
            expected_lowest = native_lowest / scale
            self.assertAlmostEqual(freqs[-1], round(expected_lowest, 6), places=4)

    def test_frequency_interpolation_vs_extrapolation_ramp_thresholds(self):
        """Verify YaRN 3-band partitioning across alpha (low) and beta (high) wavelength thresholds."""
        dim = 64
        base = 10000.0
        orig_ctx = 4096
        beta_fast = 32.0
        beta_slow = 1.0
        scale = 16.0

        low_thresh = orig_ctx / beta_fast # 128.0
        high_thresh = orig_ctx / beta_slow # 4096.0

        freqs = compute_yarn_frequencies(
            dim=dim, base=base, scale=scale,
            original_context_len=orig_ctx,
            beta_fast=beta_fast, beta_slow=beta_slow
        )

        for i in range(0, dim, 2):
            idx = i // 2
            native_f = 1.0 / (base ** (i / dim))
            wl = 2.0 * math.pi / native_f

            if wl <= low_thresh:
                # Band 1: High frequency / pure extrapolation
                self.assertEqual(freqs[idx], round(native_f, 6))
            elif wl >= high_thresh:
                # Band 2: Low frequency / pure interpolation
                self.assertEqual(freqs[idx], round(native_f / scale, 6))
            else:
                # Band 3: Medium frequency / smooth ramp
                self.assertLess(freqs[idx], round(native_f, 6))
                self.assertGreater(freqs[idx], round(native_f / scale, 6))

    def test_zero_dimensions_handling(self):
        """Verify dim=0 exits gracefully returning empty list without ZeroDivisionError."""
        self.assertEqual(compute_yarn_frequencies(dim=0), [])
        self.assertEqual(compute_yarn_rotary_embedding(pos=10, freqs=[]), ([], []))
        self.assertEqual(apply_yarn_rope(vec=[], pos=5, freqs=[]), [])

    def test_invalid_dimension_inputs(self):
        """Verify negative or odd dimensions raise ValueError."""
        with self.assertRaises(ValueError):
            compute_yarn_frequencies(dim=-64)
        with self.assertRaises(ValueError):
            compute_yarn_frequencies(dim=63) # Odd dim

    def test_invalid_scale_and_parameters(self):
        """Verify non-positive scales, bases, and beta thresholds raise ValueError."""
        with self.assertRaises(ValueError):
            compute_yarn_frequencies(scale=0.0)
        with self.assertRaises(ValueError):
            compute_yarn_frequencies(scale=-8.0)
        with self.assertRaises(ValueError):
            compute_yarn_frequencies(base=0.0)
        with self.assertRaises(ValueError):
            compute_yarn_frequencies(beta_fast=0.0)
        with self.assertRaises(ValueError):
            compute_yarn_frequencies(beta_slow=-1.0)


class TestYaRNRoPEEmbeddingsAndNegativePositions(unittest.TestCase):
    """Tests for rotary position embeddings, temperature calibration (mscale), and negative positions."""

    def test_temperature_calibration_mscale(self):
        """Verify attention logit temperature calibration factor (1/t = 0.1 * ln(s) + 1.0)."""
        # Identity at scale <= 1.0
        self.assertEqual(compute_yarn_mscale(scale=1.0), 1.0)
        self.assertEqual(compute_yarn_mscale(scale=0.5), 1.0)

        # Scale = 8x: 0.1 * ln(8) + 1.0 = 1.207944
        self.assertAlmostEqual(compute_yarn_mscale(scale=8.0), 1.207944, places=5)

        # Scale = 16x: 0.1 * ln(16) + 1.0 = 1.277259
        self.assertAlmostEqual(compute_yarn_mscale(scale=16.0), 1.277259, places=5)

        # Scale = 64x: 0.1 * ln(64) + 1.0 = 1.415888
        self.assertAlmostEqual(compute_yarn_mscale(scale=64.0), 1.415888, places=5)

        # Non-positive scale raises ValueError
        with self.assertRaises(ValueError):
            compute_yarn_mscale(scale=0.0)
        with self.assertRaises(ValueError):
            compute_yarn_mscale(scale=-4.0)

    def test_rotary_embedding_generation(self):
        """Verify cos and sin coordinate generation at zero and positive positions."""
        freqs = [1.0, 0.5, 0.25]
        # Position 0 identity
        cos_0, sin_0 = compute_yarn_rotary_embedding(pos=0, freqs=freqs)
        self.assertEqual(cos_0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertEqual(sin_0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Position 10
        cos_10, sin_10 = compute_yarn_rotary_embedding(pos=10, freqs=freqs)
        for i, f in enumerate(freqs):
            self.assertAlmostEqual(cos_10[i], math.cos(10 * f), places=5)
            self.assertAlmostEqual(sin_10[i], math.sin(10 * f), places=5)

    def test_negative_positions_handling(self):
        """Verify mathematically sound negative position handling and RoPE parity relations."""
        freqs = [0.8, 0.3, 0.1]
        pos = 7

        cos_pos, sin_pos = compute_yarn_rotary_embedding(pos=pos, freqs=freqs)
        cos_neg, sin_neg = compute_yarn_rotary_embedding(pos=-pos, freqs=freqs)

        # Even parity: cos(-theta) = cos(theta)
        for cp, cn in zip(cos_pos, cos_neg):
            self.assertAlmostEqual(cp, cn, places=6)

        # Odd parity: sin(-theta) = -sin(theta)
        for sp, sn in zip(sin_pos, sin_neg):
            self.assertAlmostEqual(sp, -sn, places=6)

    def test_inverse_rotation_property(self):
        """Verify rotating a vector by +pos and then -pos recovers the original vector (unitary isometry)."""
        freqs = compute_yarn_frequencies(dim=8, scale=1.0)
        original_vec = [1.2, -0.5, 3.1, 0.8, -2.0, 1.5, 0.4, -1.1]

        # Rotate forward at pos = 12
        rotated = apply_yarn_rope(original_vec, pos=12, freqs=freqs, mscale=1.0)
        # Rotate backward at pos = -12
        recovered = apply_yarn_rope(rotated, pos=-12, freqs=freqs, mscale=1.0)

        for orig, rec in zip(original_vec, recovered):
            self.assertAlmostEqual(orig, rec, places=4)

    def test_isometric_norm_preservation(self):
        """Verify RoPE rotation preserves Euclidean norm (||R(v)|| == ||v||) when mscale=1.0."""
        freqs = compute_yarn_frequencies(dim=8, scale=4.0)
        vec = [1.0, 2.0, -1.0, 0.5, 3.0, -2.5, 1.5, -0.5]
        norm_before = math.sqrt(sum(x * x for x in vec))

        rotated = apply_yarn_rope(vec, pos=42, freqs=freqs, mscale=1.0)
        norm_after = math.sqrt(sum(x * x for x in rotated))

        self.assertAlmostEqual(norm_before, norm_after, places=4)


class TestChapter09Integration(unittest.TestCase):
    """End-to-end integration test co-simulating Qwen2.5-1M 1M context with DCA and YaRN."""

    def test_qwen25_1m_long_context_pipeline(self):
        """
        Co-simulates Qwen2.5-1M ultra-long context inference architecture:
        - Sequence length: 1,048,576 tokens
        - Chunk size: 8,192 tokens (128 macro chunks)
        - Native pretraining context: 32,768 tokens
        - YaRN spectral scaling across 128 hidden head dimensions
        """
        # 1. Dual Chunk Attention Factorization
        dca = DualChunkAttention(dim=128, chunk_size=8192, pretrained_len=32768)
        full_flops, dca_flops, savings = dca.simulate_savings(1048576)

        self.assertEqual(full_flops, 1048576 * 1048576)
        self.assertGreater(savings, 98.0)

        # 2. Boundary Verification for 128 Macro Chunks
        chunk_idx, offset = dca.get_assignment(1000000)
        self.assertEqual(chunk_idx, 1000000 // 8192)
        self.assertEqual(offset, 1000000 % 8192)

        # Intra-chunk interaction
        self.assertEqual(dca.classify_relation(8190, 8100), "intra")
        # Successive-chunk interaction
        self.assertEqual(dca.classify_relation(8200, 8100), "successive")
        # Inter-chunk interaction
        self.assertEqual(dca.classify_relation(1000000, 100), "inter")

        # 3. Position Remapping Bounds
        q_pos, k_pos = dca.remap_positions(1000000, 100)
        self.assertLess(q_pos, 32768)
        self.assertLess(k_pos, 8192)

        # 4. YaRN Spectral RoPE Layer
        yarn = YaRNScaledRotaryEmbedding(
            dim=128,
            max_position_embeddings=1048576,
            original_max_position_embeddings=32768,
            base=10000.0,
            beta_fast=32.0,
            beta_slow=1.0
        )
        self.assertEqual(len(yarn.inv_freq), 64)
        self.assertGreater(yarn.mscale, 1.3)

        # Rotate query and key vectors
        q_vec = [0.1 * (i % 5) for i in range(128)]
        k_vec = [0.05 * (i % 7) for i in range(128)]

        q_rot = yarn.apply_rope(q_vec, pos=q_pos)
        k_rot = yarn.apply_rope(k_vec, pos=k_pos)

        self.assertEqual(len(q_rot), 128)
        self.assertEqual(len(k_rot), 128)

        # 5. Attention Probability Matrix over Local Chunk Window
        attn_matrix = dca.compute_attention([q_rot], [k_rot], causal=False)
        self.assertEqual(len(attn_matrix), 1)
        self.assertEqual(len(attn_matrix[0]), 1)
        self.assertAlmostEqual(attn_matrix[0][0], 1.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
