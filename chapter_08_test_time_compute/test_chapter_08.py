"""
Unit Test Suite for Chapter 8: Test-Time Reasoning Compute Dynamics
Covers DynamicBudgetController and ReasoningStreamDemuxer.
"""
import math
import os
import sys
import unittest

# Ensure current directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from dynamic_budget_controller import DynamicBudgetController, ReasoningBudgetController
from reasoning_stream_demuxer import ReasoningStreamDemuxer


class TestDynamicBudgetController(unittest.TestCase):
    """Test suite for Dynamic reasoning compute budget controller and prompt classifier."""

    def test_class_alias_compatibility(self):
        """Verify ReasoningBudgetController alias points to DynamicBudgetController."""
        self.assertIs(ReasoningBudgetController, DynamicBudgetController)

    def test_model_effort_budgets_discrete_tiers(self):
        """Verify discrete effort tier allocations for DeepSeek-V4-Pro, GLM-5.3, and Kimi K3."""
        # DeepSeek-V4-Pro
        self.assertEqual(DynamicBudgetController.allocate_thinking_budget("deepseek-v4-pro", 0.1), 1024)
        self.assertEqual(DynamicBudgetController.allocate_thinking_budget("deepseek-v4-pro", 0.5), 8192)
        self.assertEqual(DynamicBudgetController.allocate_thinking_budget("deepseek-v4-pro", 0.9), 32768)

        # GLM-5.3
        self.assertEqual(DynamicBudgetController.allocate_thinking_budget("glm-5.3", 0.2), 2048)
        self.assertEqual(DynamicBudgetController.allocate_thinking_budget("glm-5.3", 0.6), 16384)
        self.assertEqual(DynamicBudgetController.allocate_thinking_budget("glm-5.3", 0.8), 65536)

        # Kimi K3 (ensuring low, high, max keys all exist without KeyError)
        self.assertEqual(DynamicBudgetController.allocate_thinking_budget("kimi-k3", 0.15), 2048)
        self.assertEqual(DynamicBudgetController.allocate_thinking_budget("kimi-k3", 0.5), 16384)
        self.assertEqual(DynamicBudgetController.allocate_thinking_budget("kimi-k3", 0.95), 65536)

        # Fallback unknown model
        self.assertEqual(DynamicBudgetController.allocate_thinking_budget("frontier-unknown", 0.1), 1024)
        self.assertEqual(DynamicBudgetController.allocate_thinking_budget("frontier-unknown", 0.4), 4096)
        self.assertEqual(DynamicBudgetController.allocate_thinking_budget("frontier-unknown", 0.85), 16384)

    def test_boundary_and_out_of_range_complexity_clamping(self):
        """Verify boundary thresholds (0.0, 0.3, 0.7, 1.0) and negative/excess score clamping."""
        # Exact boundaries
        self.assertEqual(DynamicBudgetController.allocate_thinking_budget("deepseek-v4-pro", 0.0), 1024)
        self.assertEqual(DynamicBudgetController.allocate_thinking_budget("deepseek-v4-pro", 0.3), 8192)
        self.assertEqual(DynamicBudgetController.allocate_thinking_budget("deepseek-v4-pro", 0.7), 32768)
        self.assertEqual(DynamicBudgetController.allocate_thinking_budget("deepseek-v4-pro", 1.0), 32768)

        # Out-of-range clamping
        self.assertEqual(DynamicBudgetController.allocate_thinking_budget("deepseek-v4-pro", -1.0), 1024)
        self.assertEqual(DynamicBudgetController.allocate_thinking_budget("deepseek-v4-pro", 2.5), 32768)

    def test_empty_and_whitespace_prompt_classification(self):
        """Verify edge cases for empty strings, whitespace, and None prompts."""
        for empty_val in ["", "   ", "\t\n\r  ", None]:
            score = DynamicBudgetController.estimate_complexity(empty_val)
            cls_label = DynamicBudgetController.classify_prompt(empty_val)
            self.assertEqual(score, 0.0, f"Expected 0.0 for {empty_val!r}, got {score}")
            self.assertEqual(cls_label, "simple", f"Expected 'simple' for {empty_val!r}, got {cls_label}")

    def test_simple_prompt_classification(self):
        """Verify classification of factual lookups and short queries as 'simple'."""
        simple_prompts = [
            "Hello, how are you?",
            "What is the capital of France?",
            "Define KV Cache.",
            "Who is the author of Hamlet?",
            "Translate 'inference' to Spanish.",
            "What is 2 + 2?",
        ]
        for prompt in simple_prompts:
            score = DynamicBudgetController.estimate_complexity(prompt)
            cls_label = DynamicBudgetController.classify_prompt(prompt)
            self.assertLess(score, 0.3, f"Prompt '{prompt}' should have complexity < 0.3, got {score}")
            self.assertEqual(cls_label, "simple", f"Prompt '{prompt}' should be classified as simple")

    def test_complex_prompt_classification(self):
        """Verify classification of math proofs, deep kernels, and complex synthesis as 'complex'."""
        complex_prompts = [
            "Prove by induction that the sum of the first n odd integers is n squared.",
            "Write a CUDA kernel for FlashAttention-3 with TMA warp-specialization and ping-pong GEMM.",
            "Derive the arithmetic intensity roofline boundary for Blackwell B200 with FP4 microscaling.",
            "Analyze the memory wall trade-offs between MLA and standard MHA under high batch size.",
            "Implement a Radix tree prefix cache with LRU eviction and memory recycling algorithm.",
            "Solve the following non-linear differential equation step-by-step with rigorous proof.",
        ]
        for prompt in complex_prompts:
            score = DynamicBudgetController.estimate_complexity(prompt)
            cls_label = DynamicBudgetController.classify_prompt(prompt)
            self.assertGreaterEqual(score, 0.3, f"Prompt '{prompt}' should have complexity >= 0.3, got {score}")
            self.assertEqual(cls_label, "complex", f"Prompt '{prompt}' should be classified as complex")

    def test_budget_clamping_strict_invariants(self):
        """Verify compute_budget clamps strictly within [min_tokens, max_tokens]."""
        min_tokens = 512
        max_tokens = 16384

        # Test extreme scores
        self.assertEqual(DynamicBudgetController.compute_budget(-5.0, min_tokens, max_tokens), min_tokens)
        self.assertEqual(DynamicBudgetController.compute_budget(0.0, min_tokens, max_tokens), min_tokens)
        self.assertEqual(DynamicBudgetController.compute_budget(1.0, min_tokens, max_tokens), max_tokens)
        self.assertEqual(DynamicBudgetController.compute_budget(99.0, min_tokens, max_tokens), max_tokens)

        # Range invalidation checks
        with self.assertRaises(ValueError):
            DynamicBudgetController.compute_budget(0.5, min_tokens=5000, max_tokens=1000)
        with self.assertRaises(ValueError):
            DynamicBudgetController.compute_budget(0.5, min_tokens=-100, max_tokens=1000)
        with self.assertRaises(ValueError):
            DynamicBudgetController.compute_budget(0.5, min_tokens=100, max_tokens=0)

    def test_scaling_functions(self):
        """Verify linear, exponential, quadratic, sqrt, and step scaling behaviors."""
        min_t = 1000
        max_t = 10000

        # Linear: min + score * (max - min)
        self.assertEqual(DynamicBudgetController.compute_budget(0.0, min_t, max_t, "linear"), 1000)
        self.assertEqual(DynamicBudgetController.compute_budget(0.5, min_t, max_t, "linear"), 5500)
        self.assertEqual(DynamicBudgetController.compute_budget(1.0, min_t, max_t, "linear"), 10000)

        # Exponential / Geometric: min * (max / min) ** score
        # For min=1024, max=16384, score=0.5 -> sqrt(1024 * 16384) = 4096
        self.assertEqual(DynamicBudgetController.compute_budget(0.0, 1024, 16384, "exponential"), 1024)
        self.assertEqual(DynamicBudgetController.compute_budget(0.5, 1024, 16384, "exponential"), 4096)
        self.assertEqual(DynamicBudgetController.compute_budget(1.0, 1024, 16384, "exponential"), 16384)

        # Quadratic: min + (score^2) * (max - min)
        # score=0.5 -> min + 0.25 * 9000 = 1000 + 2250 = 3250
        self.assertEqual(DynamicBudgetController.compute_budget(0.5, min_t, max_t, "quadratic"), 3250)

        # Sqrt: min + sqrt(score) * (max - min)
        expected_sqrt = int(round(min_t + math.sqrt(0.25) * (max_t - min_t)))  # 1000 + 0.5 * 9000 = 5500
        self.assertEqual(DynamicBudgetController.compute_budget(0.25, min_t, max_t, "sqrt"), expected_sqrt)

        # Step
        self.assertEqual(DynamicBudgetController.compute_budget(0.1, min_t, max_t, "step"), min_t)
        self.assertEqual(DynamicBudgetController.compute_budget(0.5, min_t, max_t, "step"), 5500)
        self.assertEqual(DynamicBudgetController.compute_budget(0.9, min_t, max_t, "step"), max_t)

        # Unknown scaling mode
        with self.assertRaises(ValueError):
            DynamicBudgetController.compute_budget(0.5, min_t, max_t, "hyperbolic")

    def test_controller_instance_methods(self):
        """Verify instance initialization and allocate_from_prompt method."""
        controller = DynamicBudgetController(
            model="deepseek-v4-pro",
            min_tokens=2048,
            max_tokens=65536,
            default_scaling="exponential",
        )
        self.assertEqual(controller.model, "deepseek-v4-pro")
        self.assertEqual(controller.min_tokens, 2048)
        self.assertEqual(controller.max_tokens, 65536)

        # Test prompt allocation
        simple_budget = controller.allocate_from_prompt("What is the capital of Italy?")
        self.assertEqual(simple_budget, 2048)

        complex_budget = controller.allocate_from_prompt(
            "Write a CUDA kernel for FlashAttention-3 with TMA warp-specialization and calculate roofline arithmetic intensity."
        )
        self.assertGreater(complex_budget, 2048)
        self.assertLessEqual(complex_budget, 65536)


class TestReasoningStreamDemuxer(unittest.TestCase):
    """Test suite for ReasoningStreamDemuxer token separation, streaming, and edge cases."""

    def setUp(self):
        self.demuxer = ReasoningStreamDemuxer()

    def test_standard_stream(self):
        """Verify standard stream with whole <think> and </think> tokens."""
        stream = ["<think>", "Analyze", " memory", " wall", "</think>", "Optimal", " solution."]
        for tok in stream:
            self.demuxer.feed_token(tok)
        self.demuxer.finalize()

        self.assertEqual("".join(self.demuxer.thinking_tokens), "Analyze memory wall")
        self.assertEqual("".join(self.demuxer.response_tokens), "Optimal solution.")
        self.assertFalse(self.demuxer.is_thinking)

    def test_partial_streaming_tags_split_in_two(self):
        """Verify partial tag splits like '<th' then 'ink>', and '</th' then 'ink>'."""
        stream = ["<th", "ink>", "Step 1: Check VRAM", "</th", "ink>", "The result is 16GB."]
        for tok in stream:
            self.demuxer.feed_token(tok)
        self.demuxer.finalize()

        self.assertEqual(self.demuxer.get_thinking_text(), "Step 1: Check VRAM")
        self.assertEqual(self.demuxer.get_response_text(), "The result is 16GB.")

    def test_partial_streaming_tags_split_char_by_char(self):
        """Verify extreme edge case where tags arrive 1 character per token chunk."""
        stream = [
            "<", "t", "h", "i", "n", "k", ">",
            "Deliberating", "...",
            "<", "/", "t", "h", "i", "n", "k", ">",
            "Final", " answer"
        ]
        for tok in stream:
            self.demuxer.feed_token(tok)
        self.demuxer.finalize()

        self.assertEqual(self.demuxer.get_thinking_text(), "Deliberating...")
        self.assertEqual(self.demuxer.get_response_text(), "Final answer")

    def test_partial_tags_arbitrary_splits(self):
        """Verify non-standard split patterns: '<', 'think>', '</', 'think>'."""
        stream = ["<", "think>", "Reasoning clause", "</", "think>", "Output."]
        for tok in stream:
            self.demuxer.feed_token(tok)
        self.demuxer.finalize()

        self.assertEqual(self.demuxer.get_thinking_text(), "Reasoning clause")
        self.assertEqual(self.demuxer.get_response_text(), "Output.")

    def test_embedded_tokens_with_tags_and_text(self):
        """Verify tokens containing both tags and text in the same chunk."""
        stream = ["<think>Quick internal deliberation", " and synthesis</think>Visible answer."]
        for tok in stream:
            self.demuxer.feed_token(tok)
        self.demuxer.finalize()

        self.assertEqual(self.demuxer.get_thinking_text(), "Quick internal deliberation and synthesis")
        self.assertEqual(self.demuxer.get_response_text(), "Visible answer.")

    def test_missing_closing_tag(self):
        """Verify stream termination when closing tag </think> is never produced (generation cutoff)."""
        stream = ["<think>", "Evaluating", " infinite", " search", " tree..."]
        for tok in stream:
            self.demuxer.feed_token(tok)
        self.demuxer.finalize()

        self.assertTrue(self.demuxer.is_thinking)
        self.assertEqual(self.demuxer.get_thinking_text(), "Evaluating infinite search tree...")
        self.assertEqual(self.demuxer.response_tokens, [])

    def test_multiple_think_blocks(self):
        """Verify multi-phase reasoning with multiple sequential <think>...</think> sections."""
        stream = [
            "<think>", "Phase 1: plan", "</think>",
            "Intermediate answer. ",
            "<think>", "Phase 2: verify", "</think>",
            "Final confirmed conclusion."
        ]
        for tok in stream:
            self.demuxer.feed_token(tok)
        self.demuxer.finalize()

        self.assertEqual(self.demuxer.get_thinking_text(), "Phase 1: planPhase 2: verify")
        self.assertEqual(self.demuxer.get_response_text(), "Intermediate answer. Final confirmed conclusion.")

    def test_empty_thoughts(self):
        """Verify handling of empty <think></think> blocks."""
        # Split empty thought
        demuxer1 = ReasoningStreamDemuxer()
        for tok in ["<think>", "</think>", "Direct response without thinking."]:
            demuxer1.feed_token(tok)
        demuxer1.finalize()
        self.assertEqual(demuxer1.thinking_tokens, [])
        self.assertFalse(demuxer1.has_thinking_content())
        self.assertEqual(demuxer1.get_response_text(), "Direct response without thinking.")

        # Combined empty thought
        demuxer2 = ReasoningStreamDemuxer()
        demuxer2.feed_token("<think></think>Immediate answer.")
        demuxer2.finalize()
        self.assertEqual(demuxer2.thinking_tokens, [])
        self.assertFalse(demuxer2.has_thinking_content())
        self.assertEqual(demuxer2.get_response_text(), "Immediate answer.")

    def test_non_thinking_outputs(self):
        """Verify pass-through behavior when the model produces no <think> tags."""
        stream = ["Standard", " non-reasoning", " model", " completion", " without", " tags."]
        for tok in stream:
            self.demuxer.feed_token(tok)
        self.demuxer.finalize()

        self.assertEqual(self.demuxer.thinking_tokens, [])
        self.assertFalse(self.demuxer.is_thinking)
        self.assertEqual(self.demuxer.get_response_text(), "Standard non-reasoning model completion without tags.")

    def test_false_alarm_tags(self):
        """Verify false alarm tags such as '<th' followed by normal text are not swallowed."""
        stream = ["<th", "is is an HTML-like tag <tag> in prompt>", " and regular output."]
        for tok in stream:
            self.demuxer.feed_token(tok)
        self.demuxer.finalize()

        self.assertEqual(self.demuxer.thinking_tokens, [])
        self.assertIn("<this is an HTML-like tag <tag> in prompt> and regular output.", self.demuxer.get_response_text())

    def test_math_and_comparison_brackets_inside_thinking(self):
        """Verify mathematical comparison operators (< and >) inside thinking deliberation."""
        stream = ["<think>", "Checking constraint: x < 10 and y </t error?", "</think>", "Valid."]
        for tok in stream:
            self.demuxer.feed_token(tok)
        self.demuxer.finalize()

        self.assertIn("x < 10 and y </t error?", self.demuxer.get_thinking_text())
        self.assertEqual(self.demuxer.get_response_text(), "Valid.")

    def test_trailing_partial_tag_flushed_on_finalize(self):
        """Verify trailing partial tag prefix is flushed to tokens when finalize() is called."""
        stream = ["<think>", "Pondering", "</th"]
        for tok in stream:
            self.demuxer.feed_token(tok)
        self.demuxer.finalize()

        self.assertEqual(self.demuxer.get_thinking_text(), "Pondering</th")

    def test_demuxer_reset(self):
        """Verify reset clears state and token buffers cleanly."""
        self.demuxer.feed_token("<think>Deliberation</think>Response")
        self.demuxer.finalize()
        self.assertTrue(bool(self.demuxer.thinking_tokens))

        self.demuxer.reset()
        self.assertEqual(self.demuxer.thinking_tokens, [])
        self.assertEqual(self.demuxer.response_tokens, [])
        self.assertEqual(self.demuxer._buffer, "")
        self.assertFalse(self.demuxer.is_thinking)

    def test_prefill_in_think_mode(self):
        """Verify initialization with in_think=True for servers pre-filling <think> in prompt."""
        demuxer = ReasoningStreamDemuxer(in_think=True)
        self.assertTrue(demuxer.is_thinking)

        stream = ["Thinking already started", "</think>", "Done."]
        for tok in stream:
            demuxer.feed_token(tok)
        demuxer.finalize()

        self.assertEqual(demuxer.get_thinking_text(), "Thinking already started")
        self.assertEqual(demuxer.get_response_text(), "Done.")
        self.assertFalse(demuxer.is_thinking)

    def test_feed_stream_generator(self):
        """Verify feed_stream generator yields (channel, text) tuples in real-time."""
        stream = ["<th", "ink>", "Fast thought", "</th", "ink>", "Fast answer"]
        events = list(self.demuxer.feed_stream(stream))

        expected = [
            ("think", "Fast thought"),
            ("response", "Fast answer"),
        ]
        self.assertEqual(events, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
