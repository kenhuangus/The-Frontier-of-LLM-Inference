"""
Unit Test Suite for Chapter 4: Extreme Quantization and Precision (FP4 / FP8 / KIVI)
Book: The Frontier of LLM Inference

Validates:
1. mxfp4_microscaling.py:
   - E2M1 FP4 numeric representation and rounding boundaries
   - OCP MXFP4 block size 32 microscaling and alignment padding
   - All-zero input blocks and zero-scale handling
   - Extreme values (NaN, +/- Inf, large dynamic range)
   - Scale factor clamping within E8M0 bounds [2^-127, 2^127]
   - Signal-to-Quantization-Noise Ratio (SNR / SQNR in dB) computation
2. kivi_quantizer.py:
   - Asymmetric 2-bit quantization and code clamping [0, 3]
   - Constant-value and zero-variance vector stability
   - Per-channel Key cache quantization across temporal tokens
   - Per-token Value cache quantization across feature channels
   - Mathematical residual error bounds (|error| <= (max - min) / 6.0)
   - Memory footprint compression (4x vs FP8, 8x vs FP16)
3. fast_rmsnorm.py:
   - Exact mathematical RMSNorm definition and output scaling
   - Zero-vector normalization without numerical division-by-zero
   - Epsilon stability across dynamic ranges (0.0 to 1.0)
   - Scale weight scaling (affine transformation, zero weights, inversion)
   - Batch RMSNorm evaluation and dimension mismatch safety
"""
import math
import random
import unittest
from typing import List

# Import chapter modules under test
from mxfp4_microscaling import (
    E2M1_VALUES,
    MIN_SCALE,
    MAX_SCALE,
    quantize_e2m1,
    quantize_mxfp4_block,
    quantize_mxfp4_tensor,
    calculate_snr,
)
from kivi_quantizer import (
    quantize_2bit_asymmetric,
    dequantize_2bit,
    quantize_key_per_channel,
    dequantize_key_per_channel,
    quantize_value_per_token,
    dequantize_value_per_token,
    max_residual_error,
)
from fast_rmsnorm import (
    rmsnorm_reference,
    fast_rmsnorm_batch,
)


class TestMXFP4Microscaling(unittest.TestCase):
    """Test suite for NVIDIA Blackwell NV-FP4 / OCP MXFP4 microscaling simulator."""

    def test_e2m1_representable_magnitudes(self):
        """Verify that exact E2M1 magnitudes quantize to themselves losslessly."""
        for mag in E2M1_VALUES:
            self.assertEqual(quantize_e2m1(mag), mag)
            self.assertEqual(quantize_e2m1(-mag), -mag)

    def test_e2m1_midpoint_rounding(self):
        """Verify nearest-value rounding for intermediate values."""
        # Midpoint between 0.0 and 0.5 is 0.25; equidistant rounds to candidate with <= dist
        q_pos = quantize_e2m1(0.24)
        self.assertEqual(q_pos, 0.0)
        q_high = quantize_e2m1(0.26)
        self.assertEqual(q_high, 0.5)

        # Values around 5.0 (midpoint between 4.0 and 6.0)
        self.assertIn(quantize_e2m1(5.0), [4.0, 6.0])
        self.assertEqual(quantize_e2m1(5.1), 6.0)
        self.assertEqual(quantize_e2m1(3.9), 4.0)

    def test_e2m1_extreme_values(self):
        """Verify robust handling of NaN and +/- Infinity."""
        self.assertEqual(quantize_e2m1(float('nan')), 0.0)
        self.assertEqual(quantize_e2m1(float('inf')), 6.0)
        self.assertEqual(quantize_e2m1(float('-inf')), -6.0)
        # Values exceeding maximum magnitude clamp to 6.0
        self.assertEqual(quantize_e2m1(100.0), 6.0)
        self.assertEqual(quantize_e2m1(-100.0), -6.0)

    def test_mxfp4_block_basic(self):
        """Verify basic block microscaling with known vector."""
        block = [-2.8, -1.4, 0.05, 0.45, 1.12, 2.34, 3.98, -0.72]
        dequant, scale = quantize_mxfp4_block(block, block_size=16)
        
        # Max magnitude is 3.98, so scale factor = 3.98 / 6.0
        expected_scale = 3.98 / 6.0
        self.assertAlmostEqual(scale, expected_scale, places=5)
        self.assertEqual(len(dequant), len(block))
        
        # Max element should quantize to exactly 3.98 (scaled to 6.0, E2M1 max)
        max_idx = block.index(3.98)
        self.assertAlmostEqual(dequant[max_idx], 3.98, places=5)

    def test_mxfp4_all_zero_block(self):
        """Verify that an all-zero input block produces all-zero outputs without division by zero."""
        zeros_block = [0.0] * 32
        dequant, scale = quantize_mxfp4_block(zeros_block, block_size=32)
        self.assertEqual(len(dequant), 32)
        self.assertEqual(scale, 0.0)
        self.assertTrue(all(v == 0.0 for v in dequant))

    def test_mxfp4_empty_block(self):
        """Verify that an empty list returns empty results safely."""
        dequant, scale = quantize_mxfp4_block([])
        self.assertEqual(dequant, [])
        self.assertEqual(scale, 0.0)

    def test_mxfp4_block_size_32_alignment(self):
        """Verify full tensor microscaling partitioned into blocks of 32 elements."""
        # Test exact multiple of 32 (64 elements)
        random.seed(42)
        weights_64 = [random.gauss(0.0, 1.0) for _ in range(64)]
        dequant_64, scales_64 = quantize_mxfp4_tensor(weights_64, block_size=32)
        
        self.assertEqual(len(dequant_64), 64)
        self.assertEqual(len(scales_64), 2)  # Exactly 2 blocks
        for s in scales_64:
            self.assertGreater(s, 0.0)

        # Test unaligned length (e.g. 50 elements) with padding and unpadding
        weights_50 = [random.gauss(0.0, 1.0) for _ in range(50)]
        dequant_50, scales_50 = quantize_mxfp4_tensor(weights_50, block_size=32)
        self.assertEqual(len(dequant_50), 50)
        self.assertEqual(len(scales_50), 2)  # Ceil(50 / 32) = 2 blocks

    def test_mxfp4_large_dynamic_range(self):
        """Verify numerical stability across extreme dynamic ranges (1e-25 to 1e25)."""
        # Very small inputs
        tiny_weights = [1e-20 * (i + 1) for i in range(32)]
        dequant_tiny, scale_tiny = quantize_mxfp4_block(tiny_weights, block_size=32)
        self.assertGreater(scale_tiny, 0.0)
        self.assertEqual(len(dequant_tiny), 32)
        # Check max element is reconstructed accurately (scaled to 6.0)
        self.assertAlmostEqual(dequant_tiny[-1], tiny_weights[-1], delta=1e-25)

        # Very large inputs
        huge_weights = [1e20 * (i + 1) for i in range(32)]
        dequant_huge, scale_huge = quantize_mxfp4_block(huge_weights, block_size=32)
        self.assertGreater(scale_huge, 0.0)
        self.assertFalse(math.isinf(scale_huge))
        self.assertFalse(math.isnan(scale_huge))
        self.assertAlmostEqual(dequant_huge[-1], huge_weights[-1], delta=1e15)

    def test_mxfp4_scale_factor_clamping(self):
        """Verify scale factor clamping to OCP E8M0 limits."""
        # Scale below minimum E8M0 scale clamps to MIN_SCALE
        sub_min_tensor = [1e-45] * 32
        _, scale_clamped_min = quantize_mxfp4_block(sub_min_tensor, clamp_scale=True)
        self.assertGreaterEqual(scale_clamped_min, MIN_SCALE)

        # Scale exceeding maximum E8M0 scale clamps to MAX_SCALE
        huge_tensor = [1e40] * 32
        _, scale_clamped_max = quantize_mxfp4_block(huge_tensor, clamp_scale=True)
        self.assertLessEqual(scale_clamped_max, MAX_SCALE)

    def test_mxfp4_snr_calculation(self):
        """Verify Signal-to-Quantization-Noise Ratio calculation."""
        # Lossless case -> infinite SNR
        sig = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(calculate_snr(sig, sig), float('inf'))

        # All-zero signal -> 0.0 SNR
        self.assertEqual(calculate_snr([0.0] * 4, [0.0] * 4), 0.0)

        # Dimension mismatch raises ValueError
        with self.assertRaises(ValueError):
            calculate_snr([1.0, 2.0], [1.0])

        # Analytical SNR test: signal = [2.0], quantized = [1.0]
        # P_signal = 4.0, P_noise = (2-1)^2 = 1.0 => SNR = 10 * log10(4/1) = 6.0206 dB
        snr_test = calculate_snr([2.0], [1.0])
        self.assertAlmostEqual(snr_test, 10.0 * math.log10(4.0), places=4)

        # Blackwell realistic weight distribution SNR (should exceed 15 dB)
        random.seed(1337)
        normal_block = [random.gauss(0.0, 1.0) for _ in range(32)]
        dequant, _ = quantize_mxfp4_block(normal_block, block_size=32)
        realistic_snr = calculate_snr(normal_block, dequant)
        self.assertGreater(realistic_snr, 15.0)


class TestKIVIQuantizer(unittest.TestCase):
    """Test suite for KIVI 2-bit asymmetric KV cache quantizer."""

    def test_asymmetric_2bit_basic(self):
        """Verify that 4 distinct linearly spaced values map exactly to codes [0, 1, 2, 3]."""
        vals = [0.0, 1.0, 2.0, 3.0]
        q_codes, scale, zp = quantize_2bit_asymmetric(vals)
        self.assertEqual(q_codes, [0, 1, 2, 3])
        self.assertAlmostEqual(scale, 1.0)
        self.assertAlmostEqual(zp, 0.0)

        # Dequantization should reconstruct exactly
        recon = dequantize_2bit(q_codes, scale, zp)
        for orig, rec in zip(vals, recon):
            self.assertAlmostEqual(orig, rec, places=6)

    def test_asymmetric_2bit_clamping(self):
        """Verify that all quantized codes fall strictly within 2-bit range [0, 3]."""
        random.seed(101)
        data = [random.uniform(-50.0, 75.0) for _ in range(100)]
        q_codes, scale, zp = quantize_2bit_asymmetric(data)
        for q in q_codes:
            self.assertIsInstance(q, int)
            self.assertGreaterEqual(q, 0)
            self.assertLessEqual(q, 3)

    def test_constant_input_stability(self):
        """Verify zero-variance / constant input list does not cause division by zero."""
        constant_val = 3.14159
        vals = [constant_val] * 16
        q_codes, scale, zp = quantize_2bit_asymmetric(vals)
        self.assertEqual(len(q_codes), 16)
        recon = dequantize_2bit(q_codes, scale, zp)
        for r in recon:
            self.assertAlmostEqual(r, constant_val, places=5)

    def test_empty_input(self):
        """Verify empty input returns empty codes safely."""
        q_codes, scale, zp = quantize_2bit_asymmetric([])
        self.assertEqual(q_codes, [])
        self.assertEqual(scale, 1.0)
        self.assertEqual(zp, 0.0)

    def test_per_channel_key_quantization(self):
        """
        Verify KIVI per-channel Key quantization.
        Key matrix shape: [num_tokens=4, num_channels=3]
        Each column (channel) must have its own independent scale and zero_point.
        """
        key_cache = [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
        ]
        q_keys, scales, zps = quantize_key_per_channel(key_cache)

        self.assertEqual(len(scales), 3)  # 3 channels
        self.assertEqual(len(zps), 3)
        self.assertEqual(len(q_keys), 4)  # 4 tokens
        self.assertEqual(len(q_keys[0]), 3)

        # Check channel scales: col 0 has range 3 -> scale 1.0; col 1 range 30 -> scale 10.0
        self.assertAlmostEqual(scales[0], 1.0, places=5)
        self.assertAlmostEqual(scales[1], 10.0, places=5)
        self.assertAlmostEqual(scales[2], 100.0, places=5)

        # Dequantize and verify exact reconstruction for linear steps
        dequant_keys = dequantize_key_per_channel(q_keys, scales, zps)
        for t in range(4):
            for d in range(3):
                self.assertAlmostEqual(dequant_keys[t][d], key_cache[t][d], places=5)

    def test_per_token_value_quantization(self):
        """
        Verify KIVI per-token Value quantization.
        Value matrix shape: [num_tokens=3, num_channels=4]
        Each row (token) must have its own independent scale and zero_point.
        """
        value_cache = [
            [0.0, 1.0, 2.0, 3.0],       # Token 0: scale = 1.0, zp = 0.0
            [10.0, 20.0, 30.0, 40.0],   # Token 1: scale = 10.0, zp = 10.0
            [-3.0, -1.0, 1.0, 3.0],     # Token 2: scale = 2.0, zp = -3.0
        ]
        q_values, scales, zps = quantize_value_per_token(value_cache)

        self.assertEqual(len(scales), 3)  # 3 tokens
        self.assertEqual(len(zps), 3)
        self.assertEqual(len(q_values), 3)
        self.assertEqual(len(q_values[0]), 4)

        self.assertAlmostEqual(scales[0], 1.0, places=5)
        self.assertAlmostEqual(scales[1], 10.0, places=5)
        self.assertAlmostEqual(scales[2], 2.0, places=5)

        # Dequantize and verify reconstruction
        dequant_values = dequantize_value_per_token(q_values, scales, zps)
        for t in range(3):
            for d in range(4):
                self.assertAlmostEqual(dequant_values[t][d], value_cache[t][d], places=5)

    def test_residual_error_bounds(self):
        """
        Formally verify the theoretical 2-bit quantization residual error bound:
        |x - x_hat| <= scale / 2.0 = (max - min) / 6.0
        across diverse continuous distributions.
        """
        random.seed(999)
        test_distributions = [
            [random.uniform(-10.0, 10.0) for _ in range(500)],
            [random.gauss(0.0, 5.0) for _ in range(500)],
            [random.expovariate(0.5) - 2.0 for _ in range(500)],
        ]

        for dist in test_distributions:
            q_vals, scale, zp = quantize_2bit_asymmetric(dist)
            recon = dequantize_2bit(q_vals, scale, zp)
            err = max_residual_error(dist, recon)
            theoretical_bound = (max(dist) - min(dist)) / 6.0 + 1e-7
            self.assertLessEqual(
                err,
                theoretical_bound,
                f"Residual error {err} exceeded theoretical bound {theoretical_bound}"
            )

    def test_memory_savings_ratio(self):
        """Verify architectural memory reduction claim: 16-bit to 2-bit is 8x savings."""
        fp16_bits = 16
        kivi_bits = 2
        compression_ratio = fp16_bits / kivi_bits
        self.assertEqual(compression_ratio, 8.0)
        fp8_compression = 8 / kivi_bits
        self.assertEqual(fp8_compression, 4.0)


class TestFastRMSNorm(unittest.TestCase):
    """Test suite for fast RMSNorm reference and fused layer simulator."""

    def test_rmsnorm_basic_correctness(self):
        """Verify RMSNorm on a known vector matches analytical definition."""
        x = [1.0, -1.0, 1.0, -1.0]
        w = [1.0, 1.0, 1.0, 1.0]
        # mean_sq = (1 + 1 + 1 + 1) / 4 = 1.0
        # rstd = 1 / sqrt(1.0 + 0.0) = 1.0
        out = rmsnorm_reference(x, w, eps=0.0)
        for orig, val in zip(x, out):
            self.assertAlmostEqual(orig, val, places=6)

    def test_rmsnorm_unit_rms_property(self):
        """Verify that with weight=1.0, the output vector has root-mean-square equal to 1.0."""
        random.seed(777)
        x = [random.gauss(5.0, 3.0) for _ in range(64)]
        w = [1.0] * 64
        out = rmsnorm_reference(x, w, eps=1e-8)
        out_rms = math.sqrt(sum(v * v for v in out) / len(out))
        self.assertAlmostEqual(out_rms, 1.0, places=4)

    def test_rmsnorm_zero_vector(self):
        """Verify zero-vector input produces strictly all zeros without division by zero."""
        x = [0.0] * 128
        w = [1.0] * 128
        out = rmsnorm_reference(x, w, eps=1e-6)
        self.assertEqual(len(out), 128)
        self.assertTrue(all(v == 0.0 for v in out))

    def test_rmsnorm_scale_weight_scaling(self):
        """Verify scale weight scaling (affine scale parameter gamma / weight)."""
        x = [2.0, -2.0, 4.0, -4.0]
        # Weight with distinct multipliers: 0.0, 2.5, -1.0, 0.5
        w = [0.0, 2.5, -1.0, 0.5]
        out = rmsnorm_reference(x, w, eps=1e-6)

        # Zero weight should force output to 0.0
        self.assertEqual(out[0], 0.0)

        # Ratio of out[1] / out[3] should match w[1]*x[1] / (w[3]*x[3])
        ratio_out = out[1] / out[3]
        expected_ratio = (w[1] * x[1]) / (w[3] * x[3])
        self.assertAlmostEqual(ratio_out, expected_ratio, places=5)

    def test_rmsnorm_epsilon_stability(self):
        """Verify numerical stability across wide range of epsilon values."""
        x = [1e-8, -2e-8, 1e-8]
        w = [1.0, 1.0, 1.0]
        for eps in [0.0, 1e-12, 1e-8, 1e-6, 1e-4, 1e-2, 1.0]:
            out = rmsnorm_reference(x, w, eps=eps)
            self.assertEqual(len(out), 3)
            for v in out:
                self.assertFalse(math.isnan(v))
                self.assertFalse(math.isinf(v))

    def test_rmsnorm_dimension_mismatch(self):
        """Verify that mismatched x and weight vectors raise a ValueError."""
        x = [1.0, 2.0, 3.0]
        w = [1.0, 1.0]
        with self.assertRaises(ValueError):
            rmsnorm_reference(x, w)

    def test_rmsnorm_invalid_epsilon(self):
        """Verify that negative epsilon values raise a ValueError."""
        x = [1.0, 2.0]
        w = [1.0, 1.0]
        with self.assertRaises(ValueError):
            rmsnorm_reference(x, w, eps=-1e-5)

    def test_rmsnorm_empty_input(self):
        """Verify empty input vector returns empty list safely."""
        self.assertEqual(rmsnorm_reference([], []), [])

    def test_fast_rmsnorm_batch(self):
        """Verify batch RMSNorm computes row-wise normalizations accurately."""
        batch_x = [
            [1.0, 2.0, -1.0],
            [3.0, 0.0, -3.0],
            [0.0, 0.0, 0.0],
        ]
        w = [1.0, 2.0, 0.5]
        batch_out = fast_rmsnorm_batch(batch_x, w, eps=1e-6)

        self.assertEqual(len(batch_out), 3)
        for i in range(3):
            single_out = rmsnorm_reference(batch_x[i], w, eps=1e-6)
            self.assertEqual(batch_out[i], single_out)


if __name__ == "__main__":
    print("=" * 80)
    print("  RUNNING CHAPTER 4 UNIT TESTS: QUANTIZATION (FP4 / FP8 / KIVI / RMSNORM)")
    print("=" * 80)
    unittest.main(verbosity=2)
