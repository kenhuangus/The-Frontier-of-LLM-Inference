"""
Comprehensive Unit Test Suite for Chapter 10: Production SRE & XGrammar
Covers:
  - cluster_sizing_tco.py:
      * Baseline H100 and B200 cluster sizing calculations
      * Zero QPS edge case handling (0 nodes, 0 GPUs, $0 CapEx, $0 cost/M tokens)
      * Monotonic high SLA targets (P90, P95, P99, P99.9 queueing headroom)
      * Amortized cost per million tokens (CapEx and TCO basis across depreciation windows)
      * Node scaling and architectural comparison across H100 vs B200 configurations
      * Input validation (negative QPS, negative tokens, unknown GPU type, SLA bounds)
  - xgrammar_fsm_evaluator.py:
      * 32-bit GPU packed bitmask generation and single-cycle token lookup
      * Backward-compatible token ID sets for EXPECT_KEY, EXPECT_COLON, EXPECT_VALUE
      * Empty JSON objects `{}` and arrays `[]`
      * Nested objects and arrays of arbitrary depth
      * Escaped characters in strings and diverse JSON data types (numbers, booleans, null)
      * Invalid syntax rejection (trailing commas, mismatched brackets, missing colons, EOF)
      * Direct token ID step-by-step decoding and bitmask consistency verification
"""
import math
import os
import sys
import unittest
from typing import List, Dict, Any

# Ensure chapter directory is in sys.path
CHAPTER_DIR = os.path.dirname(os.path.abspath(__file__))
if CHAPTER_DIR not in sys.path:
    sys.path.insert(0, CHAPTER_DIR)

from cluster_sizing_tco import (
    calculate_cluster_tco,
    GPU_SPECS,
)
from xgrammar_fsm_evaluator import (
    JSONGrammarBitmaskEvaluator,
    JSONFSM,
    validate_json_syntax,
)


class TestClusterSizingTCO(unittest.TestCase):
    """Test suite for GPU fleet capacity planning, SRE SLA headroom, and TCO modeling."""

    def test_baseline_h100_sizing(self):
        """Verify baseline H100 sizing at 50 QPS (2k prompt, 200 gen)."""
        res = calculate_cluster_tco(50.0, 2000, 200, "H100")
        
        # Total tokens/sec: 50 * (2000 * 0.05 + 200) = 15,000 tok/s
        self.assertEqual(res["Total_Tokens_Per_Sec"], 15000.0)
        # 15000 / 150 = 100 GPUs -> ceil(100 / 8) = 13 nodes -> 104 GPUs
        self.assertEqual(res["Required_Nodes"], 13)
        self.assertEqual(res["Total_GPUs"], 104)
        # 104 * $30,000 = $3,120,000 CapEx
        self.assertEqual(res["Hardware_CapEx_USD"], 3120000)
        # Cost per million tokens should be positive and within realistic datacenter range ($2 - $5)
        self.assertGreater(res["Cost_Per_Million_Tokens_USD"], 2.0)
        self.assertLess(res["Cost_Per_Million_Tokens_USD"], 6.0)

    def test_baseline_b200_sizing_and_efficiency(self):
        """Verify B200 sizing at 50 QPS demonstrates superior architectural efficiency."""
        res_h100 = calculate_cluster_tco(50.0, 2000, 200, "H100")
        res_b200 = calculate_cluster_tco(50.0, 2000, 200, "B200")

        # B200 decode throughput: 350 tok/s vs 150 tok/s on H100
        # 15000 / 350 = 42.86 -> 43 GPUs -> ceil(43 / 8) = 6 nodes -> 48 GPUs
        self.assertEqual(res_b200["Required_Nodes"], 6)
        self.assertEqual(res_b200["Total_GPUs"], 48)
        self.assertEqual(res_b200["Hardware_CapEx_USD"], 1920000)

        # B200 must require fewer nodes, fewer GPUs, and lower total CapEx than H100
        self.assertLess(res_b200["Required_Nodes"], res_h100["Required_Nodes"])
        self.assertLess(res_b200["Total_GPUs"], res_h100["Total_GPUs"])
        self.assertLess(res_b200["Hardware_CapEx_USD"], res_h100["Hardware_CapEx_USD"])
        self.assertLess(res_b200["Cost_Per_Million_Tokens_USD"], res_h100["Cost_Per_Million_Tokens_USD"])

    def test_scaling_across_qps_workloads(self):
        """Verify monotonic node and GPU scaling across increasing QPS targets."""
        workloads = [10.0, 50.0, 100.0, 250.0, 500.0]
        prev_h100_gpus = 0
        prev_b200_gpus = 0

        for qps in workloads:
            h100 = calculate_cluster_tco(qps, 1000, 100, "H100")
            b200 = calculate_cluster_tco(qps, 1000, 100, "B200")

            # Nodes must be multiples of 8 GPUs
            self.assertEqual(h100["Total_GPUs"], h100["Required_Nodes"] * 8)
            self.assertEqual(b200["Total_GPUs"], b200["Required_Nodes"] * 8)

            # Monotonic increase in required capacity
            self.assertGreaterEqual(h100["Total_GPUs"], prev_h100_gpus)
            self.assertGreaterEqual(b200["Total_GPUs"], prev_b200_gpus)
            self.assertLessEqual(b200["Total_GPUs"], h100["Total_GPUs"])

            prev_h100_gpus = h100["Total_GPUs"]
            prev_b200_gpus = b200["Total_GPUs"]

    def test_zero_qps_handling(self):
        """Verify zero QPS safely yields 0 nodes, 0 GPUs, $0 CapEx, and $0 unit cost."""
        res = calculate_cluster_tco(0.0, 2000, 200, "H100")
        self.assertEqual(res["Target_QPS"], 0.0)
        self.assertEqual(res["Total_Tokens_Per_Sec"], 0.0)
        self.assertEqual(res["Required_Nodes"], 0)
        self.assertEqual(res["Total_GPUs"], 0)
        self.assertEqual(res["Hardware_CapEx_USD"], 0)
        self.assertEqual(res["Total_TCO_USD"], 0.0)
        self.assertEqual(res["Cost_Per_Million_Tokens_USD"], 0.0)
        self.assertEqual(res["CapEx_Per_Million_Tokens_USD"], 0.0)

        # Test integer zero as well
        res_int = calculate_cluster_tco(0, 0, 0, "B200")
        self.assertEqual(res_int["Required_Nodes"], 0)
        self.assertEqual(res_int["Total_GPUs"], 0)

    def test_high_sla_targets_headroom(self):
        """Verify higher SLA percentile targets allocate strictly more capacity headroom."""
        base = calculate_cluster_tco(100.0, 1500, 150, "H100", sla_target=0.0)
        p90 = calculate_cluster_tco(100.0, 1500, 150, "H100", sla_target=0.90)
        p95 = calculate_cluster_tco(100.0, 1500, 150, "H100", sla_target=0.95)
        p99 = calculate_cluster_tco(100.0, 1500, 150, "H100", sla_target=0.99)
        p999 = calculate_cluster_tco(100.0, 1500, 150, "H100", sla_target=0.999)

        # Headroom multipliers must scale monotonically with SLA target
        self.assertEqual(base["Headroom_Multiplier"], 1.0)
        self.assertGreater(p90["Headroom_Multiplier"], base["Headroom_Multiplier"])
        self.assertGreater(p95["Headroom_Multiplier"], p90["Headroom_Multiplier"])
        self.assertGreater(p99["Headroom_Multiplier"], p95["Headroom_Multiplier"])
        self.assertGreater(p999["Headroom_Multiplier"], p99["Headroom_Multiplier"])

        # High SLA targets require >= nodes and CapEx
        self.assertGreaterEqual(p99["Total_GPUs"], base["Total_GPUs"])
        self.assertGreaterEqual(p99["Hardware_CapEx_USD"], base["Hardware_CapEx_USD"])
        self.assertGreaterEqual(p999["Total_GPUs"], p99["Total_GPUs"])

    def test_cost_per_million_tokens_amortization(self):
        """Verify cost per million tokens sensitivity across depreciation windows and OpEx."""
        # Baseline 3-year amortization with 20% annual OpEx
        res_3yr = calculate_cluster_tco(100.0, 1000, 100, "H100", amortization_years=3.0, annual_opex_ratio=0.20)
        # 1-year aggressive depreciation
        res_1yr = calculate_cluster_tco(100.0, 1000, 100, "H100", amortization_years=1.0, annual_opex_ratio=0.20)
        # 5-year extended lifecycle
        res_5yr = calculate_cluster_tco(100.0, 1000, 100, "H100", amortization_years=5.0, annual_opex_ratio=0.20)

        # Longer depreciation amortizes CapEx over more tokens, lowering cost per million tokens
        self.assertGreater(res_1yr["Cost_Per_Million_Tokens_USD"], res_3yr["Cost_Per_Million_Tokens_USD"])
        self.assertGreater(res_3yr["Cost_Per_Million_Tokens_USD"], res_5yr["Cost_Per_Million_Tokens_USD"])

        # CapEx per million tokens is strictly 1/3 for 3-year vs 1-year
        self.assertAlmostEqual(res_1yr["CapEx_Per_Million_Tokens_USD"] / 3.0, res_3yr["CapEx_Per_Million_Tokens_USD"], delta=0.05)

        # OpEx ratio variation: higher OpEx increases TCO cost but preserves CapEx cost
        res_low_opex = calculate_cluster_tco(100.0, 1000, 100, "H100", annual_opex_ratio=0.10)
        res_high_opex = calculate_cluster_tco(100.0, 1000, 100, "H100", annual_opex_ratio=0.30)
        self.assertGreater(res_high_opex["Total_TCO_USD"], res_low_opex["Total_TCO_USD"])
        self.assertEqual(res_high_opex["Hardware_CapEx_USD"], res_low_opex["Hardware_CapEx_USD"])

    def test_input_validation_errors(self):
        """Verify comprehensive validation of parameters, raising ValueError for invalid inputs."""
        # Negative QPS
        with self.assertRaises(ValueError):
            calculate_cluster_tco(-1.0, 1000, 100, "H100")

        # Negative prompt or generation tokens
        with self.assertRaises(ValueError):
            calculate_cluster_tco(10.0, -500, 100, "H100")
        with self.assertRaises(ValueError):
            calculate_cluster_tco(10.0, 1000, -50, "H100")

        # Unsupported GPU architecture
        with self.assertRaises(ValueError):
            calculate_cluster_tco(10.0, 1000, 100, "RTX_4090")

        # Invalid SLA targets (must be in [0.0, 1.0))
        with self.assertRaises(ValueError):
            calculate_cluster_tco(10.0, 1000, 100, "H100", sla_target=-0.1)
        with self.assertRaises(ValueError):
            calculate_cluster_tco(10.0, 1000, 100, "H100", sla_target=1.0)
        with self.assertRaises(ValueError):
            calculate_cluster_tco(10.0, 1000, 100, "H100", sla_target=1.5)

        # Invalid amortization years or OpEx ratio
        with self.assertRaises(ValueError):
            calculate_cluster_tco(10.0, 1000, 100, "H100", amortization_years=0.0)
        with self.assertRaises(ValueError):
            calculate_cluster_tco(10.0, 1000, 100, "H100", amortization_years=-2.0)
        with self.assertRaises(ValueError):
            calculate_cluster_tco(10.0, 1000, 100, "H100", annual_opex_ratio=-0.1)


class TestXGrammarBitmaskEvaluator(unittest.TestCase):
    """Test suite for 32-bit packed vocabulary bitmasks and GPU token lookup."""

    def setUp(self):
        self.evaluator = JSONGrammarBitmaskEvaluator(vocab_size=32000)

    def test_vocab_size_and_num_words(self):
        """Verify 32-bit word packing math for standard and custom vocabulary sizes."""
        # 32000 / 32 = 1000 words exactly
        self.assertEqual(self.evaluator.num_words, 1000)

        # 32001 tokens requires 1001 words
        eval_32001 = JSONGrammarBitmaskEvaluator(vocab_size=32001)
        self.assertEqual(eval_32001.num_words, 1001)

        # DeepSeek / LLaMA-3 128k vocabulary
        eval_128k = JSONGrammarBitmaskEvaluator(vocab_size=128256)
        self.assertEqual(eval_128k.num_words, 4008)

        # Invalid non-positive vocab size
        with self.assertRaises(ValueError):
            JSONGrammarBitmaskEvaluator(vocab_size=0)
        with self.assertRaises(ValueError):
            JSONGrammarBitmaskEvaluator(vocab_size=-100)

    def test_bitmask_generation_and_lookup(self):
        """Verify token bitmask packing and single-cycle bitwise evaluation."""
        tokens = [0, 15, 31, 32, 63, 64, 1024, 31999]
        bitmask = self.evaluator.generate_bitmask(tokens)
        self.assertEqual(len(bitmask), self.evaluator.num_words)

        # All specified tokens must be allowed
        for tok in tokens:
            self.assertTrue(self.evaluator.is_token_allowed(bitmask, tok))

        # Unspecified adjacent tokens must be disallowed
        disallowed = [1, 14, 16, 30, 33, 62, 65, 1023, 1025, 31998]
        for tok in disallowed:
            self.assertFalse(self.evaluator.is_token_allowed(bitmask, tok))

    def test_empty_tokens_bitmask(self):
        """Verify generating a bitmask with no valid tokens yields all-zero words."""
        bitmask = self.evaluator.generate_bitmask([])
        self.assertEqual(len(bitmask), self.evaluator.num_words)
        self.assertTrue(all(w == 0 for w in bitmask))

        # No token should be allowed
        for tok in [0, 34, 58, 123, 31999]:
            self.assertFalse(self.evaluator.is_token_allowed(bitmask, tok))

    def test_out_of_bounds_tokens(self):
        """Verify out-of-bounds token handling in bitmask generation and lookup."""
        # Generating with out-of-range token raises ValueError
        with self.assertRaises(ValueError):
            self.evaluator.generate_bitmask([-1])
        with self.assertRaises(ValueError):
            self.evaluator.generate_bitmask([32000])

        # Lookup with out-of-range token gracefully returns False
        bitmask = self.evaluator.generate_bitmask([34, 58])
        self.assertFalse(self.evaluator.is_token_allowed(bitmask, -1))
        self.assertFalse(self.evaluator.is_token_allowed(bitmask, 32000))
        self.assertFalse(self.evaluator.is_token_allowed(bitmask, 999999))

        # Bitmask length mismatch raises ValueError
        with self.assertRaises(ValueError):
            self.evaluator.is_token_allowed([0] * 50, 10)

    def test_backward_compatible_state_tokens(self):
        """Verify exact preservation of baseline Chapter 10 token sets."""
        self.assertEqual(self.evaluator.get_valid_tokens_for_state("EXPECT_KEY"), [34, 32, 10])
        self.assertEqual(self.evaluator.get_valid_tokens_for_state("EXPECT_COLON"), [58, 32])
        self.assertEqual(self.evaluator.get_valid_tokens_for_state("EXPECT_VALUE"), [34, 48, 49, 123, 91])
        # Unknown state fallback
        self.assertEqual(self.evaluator.get_valid_tokens_for_state("UNKNOWN_STATE"), list(range(100)))

    def test_get_bitmask_for_state(self):
        """Verify direct state-to-bitmask conversion."""
        mask = self.evaluator.get_bitmask_for_state("EXPECT_KEY")
        self.assertTrue(self.evaluator.is_token_allowed(mask, 34))  # '"'
        self.assertTrue(self.evaluator.is_token_allowed(mask, 32))  # ' '
        self.assertTrue(self.evaluator.is_token_allowed(mask, 10))  # '\n'
        self.assertFalse(self.evaluator.is_token_allowed(mask, 58)) # ':'
        self.assertFalse(self.evaluator.is_token_allowed(mask, 123))# '{'


class TestJSONFSMGrammar(unittest.TestCase):
    """Test suite for stateful JSON Finite State Machine parser and grammar constraints."""

    def setUp(self):
        self.fsm = JSONFSM()

    def test_empty_json_object(self):
        """Verify empty JSON object `{}` and whitespace variations."""
        self.assertTrue(self.fsm.feed_sequence("{}"))
        self.assertTrue(self.fsm.is_accepted())
        self.assertEqual(self.fsm.nesting_depth, 0)
        self.assertEqual(self.fsm.state, "TERMINAL")

        # With internal whitespace
        self.assertTrue(validate_json_syntax("{   }"))
        self.assertTrue(validate_json_syntax("{\n\t}"))

    def test_empty_json_array(self):
        """Verify empty JSON array `[]` and whitespace variations."""
        self.assertTrue(self.fsm.feed_sequence("[]"))
        self.assertTrue(self.fsm.is_accepted())
        self.assertEqual(self.fsm.nesting_depth, 0)
        self.assertEqual(self.fsm.state, "TERMINAL")

        # With internal whitespace
        self.assertTrue(validate_json_syntax("[   ]"))
        self.assertTrue(validate_json_syntax("[\n\t]"))

    def test_primitive_value_types(self):
        """Verify JSON objects containing all standard primitive types."""
        cases = [
            '{"name": "production_cluster"}',
            '{"node_count": 64}',
            '{"negative_metric": -42}',
            '{"ratio": 3.14159}',
            '{"scientific": 1.25e+4}',
            '{"enabled": true}',
            '{"offline": false}',
            '{"null_entry": null}',
            '{"escaped_str": "Quote \\"here\\", backslash \\\\"}',
            '{"multiple_keys": "val1", "num": 100, "flag": true}',
        ]
        for json_str in cases:
            self.assertTrue(validate_json_syntax(json_str), f"Failed to validate: {json_str}")

    def test_nested_objects_and_arrays(self):
        """Verify deep arbitrary nesting of objects within arrays and vice versa."""
        nested_cases = [
            '{"deep": {"level2": {"level3": {"value": 100}}}}',
            '{"array_of_numbers": [1, 2, 3, 4, 5]}',
            '{"matrix": [[1, 0], [0, 1]]}',
            '{"services": [{"name": "sglang", "port": 8000}, {"name": "vllm", "port": 8001}]}',
            '[{"id": 1, "tags": ["sre", "gpu"]}, {"id": 2, "tags": ["network"]}]',
            '{"empty_nested": {"obj": {}, "arr": []}}',
            '[[[], []], {}]',
        ]
        for json_str in nested_cases:
            self.assertTrue(validate_json_syntax(json_str), f"Failed nested JSON: {json_str}")

    def test_invalid_syntax_rejection(self):
        """Verify strict rejection of malformed or invalid JSON grammar."""
        invalid_cases = [
            ('{: "missing_key"}', "Colon right after opening brace"),
            ('{"key": }', "Missing value after colon"),
            ('{"key" "missing_colon"}', "Missing colon between key and value"),
            ('{"a": 1, }', "Trailing comma in object"),
            ('[1, 2, ]', "Trailing comma in array"),
            ('{"a": [1, 2}', "Mismatched closing brace for array"),
            ('[{"a": 1]]', "Mismatched closing bracket for object"),
            ('{"unclosed_string: 42}', "Unclosed string key"),
            ('{"unclosed_brace": 1', "Incomplete JSON object missing '}'"),
            ('[1, 2, 3', "Incomplete JSON array missing ']'"),
            ('{} extra_tokens', "Trailing tokens after terminal JSON"),
            ('', "Empty string input"),
            ('   ', "Whitespace only input"),
            ('42', "Bare primitive without container"),
            ('true', "Bare boolean without container"),
            ('{"a": 1} {"b": 2}', "Multiple concatenated root objects"),
        ]
        for json_str, desc in invalid_cases:
            self.assertFalse(validate_json_syntax(json_str), f"Should have rejected ({desc}): {json_str}")

    def test_token_id_step_evaluation(self):
        """Verify step() consuming discrete integer token IDs (ASCII ordinals)."""
        # Test sequence: {"id": 1}
        token_ids = [
            ord('{'), ord('"'), ord('i'), ord('d'), ord('"'),
            ord(':'), ord(' '), ord('1'), ord('}')
        ]
        self.fsm.reset()
        for tok in token_ids:
            # can_accept should return True before stepping
            self.assertTrue(self.fsm.can_accept(tok))
            self.fsm.step(tok)

        self.assertTrue(self.fsm.is_accepted())
        self.assertEqual(self.fsm.state, "TERMINAL")

    def test_disallowed_token_id_rejection(self):
        """Verify step() raises ValueError when presented with disallowed token ID."""
        self.fsm.reset()
        # In START state, colon (58) is forbidden
        with self.assertRaises(ValueError):
            self.fsm.step(ord(':'))

        # In START state, validly consume '{'
        self.fsm.step(ord('{'))
        # In EXPECT_KEY_OR_CLOSE, comma (44) is forbidden
        with self.assertRaises(ValueError):
            self.fsm.step(ord(','))

        # In EXPECT_KEY_OR_CLOSE, consume '"' to enter IN_KEY
        self.fsm.step(ord('"'))
        self.fsm.step(ord('k'))
        self.fsm.step(ord('"'))
        # Now in EXPECT_COLON, '{' (123) is forbidden
        with self.assertRaises(ValueError):
            self.fsm.step(ord('{'))

    def test_bitmask_and_allowed_tokens_consistency(self):
        """Verify that every token in get_allowed_tokens() is bit-1 in get_current_bitmask()."""
        # Test across multiple states during parsing of '{"gpu": "H100"}'
        test_str = '{"gpu": "H100"}'
        self.fsm.reset()

        for ch in test_str:
            allowed = self.fsm.get_allowed_tokens()
            bitmask = self.fsm.get_current_bitmask()

            # Verify every allowed token evaluates to True under active bitmask
            for tok_id in allowed:
                self.assertTrue(
                    self.fsm.evaluator.is_token_allowed(bitmask, tok_id),
                    f"Token {tok_id} in get_allowed_tokens() was masked out by bitmask in state {self.fsm.state}"
                )

            self.fsm.step_char(ch)

        self.assertTrue(self.fsm.is_accepted())


if __name__ == "__main__":
    unittest.main()
