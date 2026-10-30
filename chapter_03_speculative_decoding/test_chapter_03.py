"""
Unit Test Suite for Chapter 3: Next-Gen Speculative Decoding
The Frontier of LLM Inference (2026 Edition)

Validates:
1. SpeculativeRejectionSampler (Leviathan-Chen Invariance):
   - Acceptance condition alpha = min(1, P(x) / Q(x))
   - Residual resampling from max(0, P - Q) normalized
   - 100% acceptance under identical distributions (P = Q)
   - 0% acceptance under completely disjoint distributions (P and Q disjoint)
   - Single candidate tokens vs multi-candidate chains (early exit, bonus token)
   - Monte Carlo Total Variation statistical invariance test D_TV <= 3 / sqrt(N)
   - Numerical edge cases: zero probability, unnormalized probs, temperature, top-p nucleus
2. EAGLE-2 Dynamic Speculative Tree Architecture:
   - 2D causal tree attention mask construction (ancestor visibility, lower-triangular causal)
   - Depth-aware RoPE position IDs generation (sibling depth sharing)
   - Structural validation (single root, topological acyclicity, depth consistency, branching limits)
   - Dynamic candidate tree expansion (path-confidence scoring, node budgets, depth pruning)
   - Verification pruning (ancestor dependency pruning, longest path selection, score tie-breaking)
   - Empty tree handling across all interfaces
"""
import math
import random
import unittest
from typing import List

from speculative_sampling import SpeculativeRejectionSampler
from speculative_tree import (
    TreeNode,
    build_tree_attention_mask,
    build_tree_position_ids,
    expand_tree_candidates,
    validate_tree_structure,
    verify_and_prune_tree,
)


class TestSpeculativeRejectionSampler(unittest.TestCase):
    """Unit tests for Leviathan-Chen speculative rejection sampling."""

    def setUp(self):
        self.sampler = SpeculativeRejectionSampler()
        random.seed(42)

    def test_identical_distributions_100_percent_acceptance(self):
        """
        Leviathan-Chen Property: When draft distribution Q exactly matches target P,
        acceptance probability min(1, P(x) / Q(x)) is identically 1.0.
        All candidate tokens must be accepted with 100% certainty across all trials.
        """
        p_dist = [0.10, 0.20, 0.40, 0.15, 0.15]
        q_dist = [0.10, 0.20, 0.40, 0.15, 0.15]
        
        # Test 200 trials of multi-candidate chains of length 3
        draft_chain = [2, 1, 3]  # token IDs
        draft_probs = [q_dist, q_dist, q_dist]
        target_probs = [p_dist, p_dist, p_dist]

        for _ in range(200):
            accepted, count = self.sampler.verify_speculative_tokens(
                draft_chain, draft_probs, target_probs
            )
            self.assertEqual(count, 3)
            self.assertEqual(accepted, draft_chain)

    def test_completely_disjoint_distributions_0_percent_acceptance(self):
        """
        Leviathan-Chen Property: When draft distribution Q has support disjoint from target P,
        target probability at candidate token is 0.0 -> acceptance probability is 0.0.
        Candidate token must ALWAYS be rejected (0% acceptance rate), and the recovery
        token must strictly be resampled from the target distribution.
        """
        # Target only places mass on tokens {0, 1}; Draft only places mass on tokens {2, 3}
        p_target = [0.60, 0.40, 0.00, 0.00]
        q_draft  = [0.00, 0.00, 0.70, 0.30]

        draft_tokens = [2]
        draft_probs = [q_draft]
        target_probs = [p_target]

        for _ in range(200):
            accepted, count = self.sampler.verify_speculative_tokens(
                draft_tokens, draft_probs, target_probs
            )
            # Rejection must occur at index 0 -> emitted length is 1 (the recovery token)
            self.assertEqual(count, 1)
            recovery_token = accepted[0]
            # Draft token 2 was rejected; recovery token must come from target support {0, 1}
            self.assertNotEqual(recovery_token, 2)
            self.assertIn(recovery_token, [0, 1])

    def test_leviathan_chen_verification_condition_p_ge_q(self):
        """
        Condition: P(x) >= Q(x) -> alpha = min(1.0, P(x) / Q(x)) = 1.0.
        Candidate token MUST be accepted deterministically.
        """
        p_target = [0.2, 0.8]  # token 1 has p = 0.8
        q_draft  = [0.6, 0.4]  # token 1 has q = 0.4 -> p / q = 2.0 >= 1.0

        draft_tokens = [1]
        draft_probs = [q_draft]
        target_probs = [p_target]

        for _ in range(100):
            accepted, count = self.sampler.verify_speculative_tokens(
                draft_tokens, draft_probs, target_probs
            )
            self.assertEqual(count, 1)
            self.assertEqual(accepted, [1])

    def test_leviathan_chen_verification_condition_p_lt_q(self):
        """
        Condition: P(x) < Q(x) -> alpha = P(x) / Q(x) < 1.0.
        Candidate token acceptance frequency must converge to P(x) / Q(x).
        """
        p_target = [0.75, 0.25]  # token 1 has p = 0.25
        q_draft  = [0.50, 0.50]  # token 1 has q = 0.50 -> alpha = 0.25 / 0.50 = 0.50

        draft_tokens = [1]
        draft_probs = [q_draft]
        target_probs = [p_target]

        n_trials = 5000
        accepted_draft_count = 0
        for _ in range(n_trials):
            accepted, _ = self.sampler.verify_speculative_tokens(
                draft_tokens, draft_probs, target_probs
            )
            if accepted[0] == 1:
                accepted_draft_count += 1

        empirical_alpha = accepted_draft_count / n_trials
        self.assertAlmostEqual(empirical_alpha, 0.50, delta=0.035)

    def test_residual_resampling_normalization(self):
        """
        Verify that residual distribution max(0, P - Q) correctly isolates excess probability mass
        and never resamples tokens where P(k) <= Q(k).
        """
        # Vocabulary size 4:
        # P = [0.50, 0.30, 0.10, 0.10]
        # Q = [0.20, 0.10, 0.50, 0.20]
        # P - Q = [+0.30, +0.20, -0.40, -0.10]
        # Residual = [+0.30, +0.20, 0.00, 0.00] -> Normalized: [0.60, 0.40, 0.00, 0.00]
        p_target = [0.50, 0.30, 0.10, 0.10]
        q_draft  = [0.20, 0.10, 0.50, 0.20]

        # Draft token 2 has p=0.10, q=0.50. It will be rejected frequently.
        draft_tokens = [2]
        draft_probs = [q_draft]
        target_probs = [p_target]

        for _ in range(200):
            accepted, _ = self.sampler.verify_speculative_tokens(
                draft_tokens, draft_probs, target_probs
            )
            emitted = accepted[0]
            if emitted != 2:  # It was rejected and resampled
                # Recovery token must strictly be 0 or 1, never 2 or 3
                self.assertIn(emitted, [0, 1])

    def test_monte_carlo_distribution_invariance(self):
        """
        Formal Leviathan-Chen Speculative Invariance Theorem Verification:
        Draw N samples using speculative rejection sampling from draft Q(x) verified by target P(x).
        Compute Total Variation distance: D_TV(P_hat, P) = 0.5 * sum(|P_hat(v) - P(v)|).
        Assert that D_TV <= 3 / sqrt(N), confirming distribution fidelity to target P.
        """
        p_target = [0.12, 0.28, 0.04, 0.36, 0.10, 0.05, 0.03, 0.02]
        q_draft  = [0.20, 0.15, 0.15, 0.20, 0.10, 0.10, 0.05, 0.05]
        v_size = len(p_target)
        n_samples = 15000

        counts = [0] * v_size
        for _ in range(n_samples):
            draft_token = self.sampler.sample_from_probs(q_draft)
            accepted, _ = self.sampler.verify_speculative_tokens(
                [draft_token], [q_draft], [p_target]
            )
            emitted = accepted[0]
            counts[emitted] += 1

        p_hat = [c / n_samples for c in counts]
        d_tv = 0.5 * sum(abs(p_hat[v] - p_target[v]) for v in range(v_size))
        theoretical_bound = 3.0 / math.sqrt(n_samples)

        self.assertLessEqual(
            d_tv, theoretical_bound,
            f"Total Variation distance {d_tv:.5f} exceeded 3/sqrt(N) bound {theoretical_bound:.5f}"
        )

    def test_multi_candidate_chain_full_acceptance_no_bonus(self):
        """
        Multi-candidate chain of length K=4 where all tokens are accepted,
        and exactly K target probability distributions are supplied.
        Emitted length should be exactly K.
        """
        vocab_size = 10
        draft_tokens = [1, 3, 5, 7]
        # Target has higher or equal probability than draft on all proposed tokens
        draft_probs = [[0.1] * vocab_size for _ in range(4)]
        target_probs = [[0.1] * vocab_size for _ in range(4)]

        accepted, count = self.sampler.verify_speculative_tokens(
            draft_tokens, draft_probs, target_probs
        )
        self.assertEqual(count, 4)
        self.assertEqual(accepted, draft_tokens)

    def test_multi_candidate_chain_full_acceptance_with_bonus(self):
        """
        Leviathan-Chen Bonus Token:
        When all K draft tokens are accepted and K+1 target distribution is provided,
        the sampler emits a bonus token sampled from P_target[K].
        Emitted length should be K + 1.
        """
        draft_tokens = [1, 2]
        draft_probs = [
            [0.1, 0.9, 0.0],
            [0.1, 0.0, 0.9],
        ]
        # 3 target distributions (K + 1): token 1 and 2 guaranteed accept, plus bonus token
        target_probs = [
            [0.0, 1.0, 0.0],  # p >= q on token 1
            [0.0, 0.0, 1.0],  # p >= q on token 2
            [1.0, 0.0, 0.0],  # Bonus distribution deterministically emits token 0
        ]

        accepted, count = self.sampler.verify_speculative_tokens(
            draft_tokens, draft_probs, target_probs
        )
        self.assertEqual(count, 3)
        self.assertEqual(accepted, [1, 2, 0])

    def test_multi_candidate_chain_early_rejection(self):
        """
        Rejection at intermediate position:
        Tokens 0 and 1 are accepted. Token 2 is rejected.
        Chain must immediately terminate at position 2, emitting the recovery token.
        Tokens 3 and 4 must never be accepted or processed.
        """
        draft_tokens = [0, 1, 2, 3, 4]
        # Positions 0 and 1: 100% target agreement
        # Position 2: Disjoint distribution (draft proposes 2, target has prob 0.0 for token 2)
        # Position 3 and 4: Dummy distributions
        draft_probs = [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
        target_probs = [
            [1.0, 0.0, 0.0, 0.0, 0.0],  # Accept token 0
            [0.0, 1.0, 0.0, 0.0, 0.0],  # Accept token 1
            [0.0, 0.0, 0.0, 0.0, 1.0],  # Reject token 2; recovery token deterministically 4
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]

        accepted, count = self.sampler.verify_speculative_tokens(
            draft_tokens, draft_probs, target_probs
        )
        self.assertEqual(count, 3)
        self.assertEqual(accepted, [0, 1, 4])

    def test_empty_draft_chain_handling(self):
        """Verify behavior when draft chain is empty."""
        # Case A: target distribution provided -> single next-token generation
        target_probs = [[0.0, 0.0, 1.0]]
        accepted, count = self.sampler.verify_speculative_tokens([], [], target_probs)
        self.assertEqual(count, 1)
        self.assertEqual(accepted, [2])

        # Case B: no draft tokens and no target probs
        accepted, count = self.sampler.verify_speculative_tokens([], [], [])
        self.assertEqual(count, 0)
        self.assertEqual(accepted, [])

    def test_input_dimension_validation(self):
        """Verify that dimension mismatches raise descriptive ValueErrors."""
        # draft_probs shorter than draft_tokens
        with self.assertRaises(ValueError):
            self.sampler.verify_speculative_tokens([1, 2], [[0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]])

        # target_probs shorter than draft_tokens
        with self.assertRaises(ValueError):
            self.sampler.verify_speculative_tokens([1, 2], [[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5]])

        # vocabulary mismatch between draft and target at position 0
        with self.assertRaises(ValueError):
            self.sampler.verify_speculative_tokens([0], [[0.5, 0.5]], [[0.3, 0.3, 0.4]])

        # draft token out of vocabulary range
        with self.assertRaises(ValueError):
            self.sampler.verify_speculative_tokens([5], [[0.5, 0.5]], [[0.5, 0.5]])

        with self.assertRaises(ValueError):
            self.sampler.verify_speculative_tokens([-1], [[0.5, 0.5]], [[0.5, 0.5]])

    def test_sample_from_probs_edge_cases(self):
        """Verify sample_from_probs behavior with empty, unnormalized, or uniform inputs."""
        # Empty input raises ValueError
        with self.assertRaises(ValueError):
            self.sampler.sample_from_probs([])

        # Unnormalized distribution [2.0, 8.0] samples index 1 ~80% of the time
        unnorm = [2.0, 8.0]
        samples = [self.sampler.sample_from_probs(unnorm) for _ in range(1000)]
        one_ratio = sum(1 for s in samples if s == 1) / 1000
        self.assertAlmostEqual(one_ratio, 0.80, delta=0.05)

        # All zeros fallback
        all_zeros = [0.0, 0.0, 0.0]
        fallback_sample = self.sampler.sample_from_probs(all_zeros)
        self.assertIn(fallback_sample, [0, 1, 2])

    def test_softmax_temperature_scaling(self):
        """Verify softmax scaling under high vs low temperatures."""
        logits = [1.0, 2.0, 3.0]
        
        # High temperature -> approaches uniform distribution [1/3, 1/3, 1/3]
        high_temp = self.sampler.softmax(logits, temperature=100.0)
        for p in high_temp:
            self.assertAlmostEqual(p, 1.0 / 3.0, places=2)

        # Low temperature -> approaches one-hot on argmax (index 2)
        low_temp = self.sampler.softmax(logits, temperature=0.01)
        self.assertAlmostEqual(low_temp[2], 1.0, places=3)
        self.assertAlmostEqual(low_temp[0], 0.0, places=3)

    def test_top_p_nucleus_filtering(self):
        """Verify top_p filtering truncates tail probabilities correctly."""
        logits = [10.0, 9.0, 1.0, 0.0]
        # At top_p=0.8, only tokens 0 and 1 should have non-zero probability
        filtered = self.sampler.apply_temperature_and_topp(logits, temperature=1.0, top_p=0.8)
        self.assertAlmostEqual(sum(filtered), 1.0, places=5)
        self.assertGreater(filtered[0], 0.0)
        self.assertGreater(filtered[1], 0.0)
        self.assertEqual(filtered[2], 0.0)
        self.assertEqual(filtered[3], 0.0)


class TestSpeculativeTree(unittest.TestCase):
    """Unit tests for EAGLE-2 dynamic speculative tree architecture."""

    def setUp(self):
        # Canonical 6-node tree from Chapter 3
        self.tree_nodes = [
            TreeNode(token_id=101, depth=0, score=1.00, parent_idx=-1),  # Node 0 (Root)
            TreeNode(token_id=202, depth=1, score=0.85, parent_idx=0),   # Node 1 (Branch A)
            TreeNode(token_id=203, depth=1, score=0.65, parent_idx=0),   # Node 2 (Branch B)
            TreeNode(token_id=305, depth=2, score=0.72, parent_idx=1),   # Node 3 (A -> Child 1)
            TreeNode(token_id=306, depth=2, score=0.68, parent_idx=1),   # Node 4 (A -> Child 2)
            TreeNode(token_id=307, depth=2, score=0.55, parent_idx=2),   # Node 5 (B -> Child 1)
        ]

    def test_build_tree_attention_mask_basic(self):
        """
        Verify that 2D tree attention mask enforces strict causal visibility:
        A token attends exclusively to itself and its direct ancestor path.
        """
        mask = build_tree_attention_mask(self.tree_nodes)
        self.assertEqual(len(mask), 6)
        self.assertEqual(len(mask[0]), 6)

        # Node 0 (Root): attends only to Node 0
        self.assertEqual(mask[0], [1, 0, 0, 0, 0, 0])
        # Node 1 (Branch A): attends to Root (0) and self (1)
        self.assertEqual(mask[1], [1, 1, 0, 0, 0, 0])
        # Node 2 (Branch B): attends to Root (0) and self (2)
        self.assertEqual(mask[2], [1, 0, 1, 0, 0, 0])
        # Node 3 (Child of Branch A): attends to Root (0), Node 1, and self (3)
        self.assertEqual(mask[3], [1, 1, 0, 1, 0, 0])
        # Node 4 (Child of Branch A): attends to Root (0), Node 1, and self (4)
        self.assertEqual(mask[4], [1, 1, 0, 0, 1, 0])
        # Node 5 (Child of Branch B): attends to Root (0), Node 2, and self (5); NOT Node 1, 3, or 4
        self.assertEqual(mask[5], [1, 0, 1, 0, 0, 1])

    def test_build_tree_attention_mask_empty(self):
        """Verify empty tree returns empty mask."""
        self.assertEqual(build_tree_attention_mask([]), [])

    def test_build_tree_attention_mask_single_root(self):
        """Verify single root node mask is 1x1 [[1]]."""
        root = [TreeNode(token_id=50, depth=0, score=1.0, parent_idx=-1)]
        mask = build_tree_attention_mask(root)
        self.assertEqual(mask, [[1]])

    def test_build_tree_attention_mask_linear_chain(self):
        """Verify linear chain produces lower-triangular causal attention mask."""
        linear_nodes = [
            TreeNode(token_id=10, depth=0, score=1.0, parent_idx=-1),
            TreeNode(token_id=11, depth=1, score=0.9, parent_idx=0),
            TreeNode(token_id=12, depth=2, score=0.8, parent_idx=1),
            TreeNode(token_id=13, depth=3, score=0.7, parent_idx=2),
        ]
        mask = build_tree_attention_mask(linear_nodes)
        expected = [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ]
        self.assertEqual(mask, expected)

    def test_build_tree_position_ids(self):
        """
        Verify depth-aware RoPE position IDs:
        Sibling nodes at the same tree depth must share identical position IDs.
        """
        # Base position = 0
        pos_0 = build_tree_position_ids(self.tree_nodes, base_position=0)
        self.assertEqual(pos_0, [0, 1, 1, 2, 2, 2])

        # Base position = 256
        pos_256 = build_tree_position_ids(self.tree_nodes, base_position=256)
        self.assertEqual(pos_256, [256, 257, 257, 258, 258, 258])

        # Empty nodes returns empty list
        self.assertEqual(build_tree_position_ids([]), [])

        # Negative base_position raises ValueError
        with self.assertRaises(ValueError):
            build_tree_position_ids(self.tree_nodes, base_position=-5)

    def test_validate_tree_structure_canonical(self):
        """Verify that the canonical 6-node tree passes validation."""
        self.assertTrue(validate_tree_structure(self.tree_nodes, max_depth=2, max_branching=2))

    def test_validate_tree_structure_empty(self):
        """Verify that an empty tree is structurally valid."""
        self.assertTrue(validate_tree_structure([]))

    def test_validate_tree_structure_invalid_roots(self):
        """Verify detection of missing, multiple, or mislocated root nodes."""
        # No root (all nodes have parent_idx >= 0)
        no_root = [
            TreeNode(token_id=1, depth=0, score=1.0, parent_idx=0),
        ]
        with self.assertRaises(ValueError):
            validate_tree_structure(no_root)

        # Multiple roots
        multi_root = [
            TreeNode(token_id=1, depth=0, score=1.0, parent_idx=-1),
            TreeNode(token_id=2, depth=0, score=1.0, parent_idx=-1),
        ]
        with self.assertRaises(ValueError):
            validate_tree_structure(multi_root)

        # Root at non-zero index
        root_not_at_zero = [
            TreeNode(token_id=2, depth=1, score=0.8, parent_idx=1),
            TreeNode(token_id=1, depth=0, score=1.0, parent_idx=-1),
        ]
        with self.assertRaises(ValueError):
            validate_tree_structure(root_not_at_zero)

        # Root with non-zero depth
        root_bad_depth = [
            TreeNode(token_id=1, depth=1, score=1.0, parent_idx=-1),
        ]
        with self.assertRaises(ValueError):
            validate_tree_structure(root_bad_depth)

    def test_validate_tree_structure_depth_and_branching_violations(self):
        """Verify depth and branching constraint enforcement."""
        # Child depth not equal to parent depth + 1
        bad_depth_increment = [
            TreeNode(token_id=1, depth=0, score=1.0, parent_idx=-1),
            TreeNode(token_id=2, depth=2, score=0.8, parent_idx=0),  # Jump from 0 to 2
        ]
        with self.assertRaises(ValueError):
            validate_tree_structure(bad_depth_increment)

        # Exceeds max_depth
        with self.assertRaises(ValueError):
            validate_tree_structure(self.tree_nodes, max_depth=1)

        # Exceeds max_branching (Root has 2 children; if max_branching=1, fails)
        with self.assertRaises(ValueError):
            validate_tree_structure(self.tree_nodes, max_branching=1)

    def test_validate_tree_structure_cycle_and_bounds(self):
        """Verify cycle detection and parent pointer bounds checking."""
        # Out-of-bounds parent pointer
        oob_parent = [
            TreeNode(token_id=1, depth=0, score=1.0, parent_idx=-1),
            TreeNode(token_id=2, depth=1, score=0.8, parent_idx=99),
        ]
        with self.assertRaises(ValueError):
            validate_tree_structure(oob_parent)

        # Self-referencing node
        self_ref = [
            TreeNode(token_id=1, depth=0, score=1.0, parent_idx=-1),
            TreeNode(token_id=2, depth=1, score=0.8, parent_idx=1),
        ]
        with self.assertRaises(ValueError):
            validate_tree_structure(self_ref)

        # Cycle: 1 -> 2 -> 1
        cyclic_tree = [
            TreeNode(token_id=1, depth=0, score=1.0, parent_idx=-1),
            TreeNode(token_id=2, depth=1, score=0.8, parent_idx=2),
            TreeNode(token_id=3, depth=2, score=0.6, parent_idx=1),
        ]
        with self.assertRaises(ValueError):
            validate_tree_structure(cyclic_tree)

    def test_expand_tree_candidates_basic(self):
        """
        Verify dynamic candidate tree expansion from a proposal map:
        score(child) = score(parent) * P(child | parent).
        """
        proposals = {
            101: [(202, 0.85), (203, 0.65)],
            202: [(305, 0.80), (306, 0.70)],
            203: [(307, 0.90)]
        }
        nodes = expand_tree_candidates(101, proposals, max_depth=2, max_nodes=6)
        self.assertEqual(len(nodes), 6)

        # Check Root
        self.assertEqual(nodes[0].token_id, 101)
        self.assertEqual(nodes[0].depth, 0)
        self.assertEqual(nodes[0].score, 1.0)
        self.assertEqual(nodes[0].parent_idx, -1)

        # Check Node 1 (Branch A)
        self.assertEqual(nodes[1].token_id, 202)
        self.assertEqual(nodes[1].depth, 1)
        self.assertAlmostEqual(nodes[1].score, 0.85, places=4)
        self.assertEqual(nodes[1].parent_idx, 0)

        # Check Node 3 (Child 1 of Branch A): score = 0.85 * 0.80 = 0.68
        self.assertEqual(nodes[3].token_id, 305)
        self.assertEqual(nodes[3].depth, 2)
        self.assertAlmostEqual(nodes[3].score, 0.68, places=4)
        self.assertEqual(nodes[3].parent_idx, 1)

    def test_expand_tree_candidates_limits_and_thresholds(self):
        """Verify node budget cap, depth limits, and score threshold pruning during expansion."""
        proposals = {
            100: [(101, 0.9), (102, 0.8), (103, 0.7)],
            101: [(201, 0.9), (202, 0.8)],
            102: [(203, 0.9)],
        }

        # Max nodes budget cap: exactly 3 nodes emitted
        nodes_budget = expand_tree_candidates(100, proposals, max_depth=3, max_nodes=3)
        self.assertEqual(len(nodes_budget), 3)

        # Max depth cap: depth 1 only
        nodes_depth = expand_tree_candidates(100, proposals, max_depth=1, max_nodes=10)
        for n in nodes_depth:
            self.assertLessEqual(n.depth, 1)

        # Score threshold: minimum score 0.75
        nodes_thresh = expand_tree_candidates(100, proposals, max_depth=2, max_nodes=10, score_threshold=0.75)
        for n in nodes_thresh:
            self.assertGreaterEqual(n.score, 0.75)

    def test_expand_tree_candidates_callable_and_edge_cases(self):
        """Verify callable proposal function and empty edge cases."""
        # Callable proposal
        def proposal_fn(token_id: int, depth: int):
            if depth == 0:
                return [(token_id + 1, 0.8)]
            return []

        expanded = expand_tree_candidates(10, proposal_fn, max_depth=2, max_nodes=5)
        self.assertEqual(len(expanded), 2)
        self.assertEqual(expanded[1].token_id, 11)

        # Zero max_nodes returns empty list
        self.assertEqual(expand_tree_candidates(10, {}, max_nodes=0), [])

        # Empty proposals returns single root node
        single_root = expand_tree_candidates(10, {}, max_nodes=5)
        self.assertEqual(len(single_root), 1)
        self.assertEqual(single_root[0].token_id, 10)

    def test_verify_and_prune_tree_longest_path(self):
        """
        Verify that verification pruning selects the longest accepted path.
        Branch A survives to depth 1 (Node 1 accepted, Nodes 3,4 rejected).
        Branch B survives to depth 2 (Node 2 accepted, Node 5 accepted).
        Engine must select Branch B path [101, 203, 307] of length 3.
        """
        # Accepted candidate indices: 1, 2, 5
        surviving, accepted_seq = verify_and_prune_tree(
            self.tree_nodes, accepted_node_indices={1, 2, 5}
        )
        self.assertEqual(accepted_seq, [101, 203, 307])
        # Surviving subtree nodes include 0, 1, 2, 5
        self.assertEqual([n.token_id for n in surviving], [101, 202, 203, 307])

    def test_verify_and_prune_tree_ancestor_pruning(self):
        """
        Critical Causal Invariance Property:
        If parent is rejected, ALL its children and descendants are PRUNED,
        even if child node indices were marked accepted.
        Branch A survives (Node 1 and Node 3 accepted).
        Branch B root (Node 2) is REJECTED, but Node 5 was marked accepted.
        Node 5 must be PRUNED because Node 2 was rejected.
        """
        # Accepted candidate indices: 1, 3, 5 (Node 2 rejected!)
        surviving, accepted_seq = verify_and_prune_tree(
            self.tree_nodes, accepted_node_indices={1, 3, 5}
        )
        # Longest accepted sequence must be Branch A: [101, 202, 305]
        self.assertEqual(accepted_seq, [101, 202, 305])
        # Node 5 (token 307) must NOT be in surviving nodes
        surviving_tokens = [n.token_id for n in surviving]
        self.assertNotIn(307, surviving_tokens)
        self.assertEqual(surviving_tokens, [101, 202, 305])

    def test_verify_and_prune_tree_score_tie_breaking(self):
        """
        When two branches survive to the exact same maximum depth,
        tie-breaking selects the path with the higher cumulative path score.
        """
        # Both Node 3 (score 0.72) and Node 4 (score 0.68) accepted under Node 1
        surviving, accepted_seq = verify_and_prune_tree(
            self.tree_nodes, accepted_node_indices={1, 3, 4}
        )
        # Node 3 has higher score (0.72 > 0.68), so [101, 202, 305] is chosen
        self.assertEqual(accepted_seq, [101, 202, 305])

    def test_verify_and_prune_tree_all_rejected(self):
        """When all proposed candidates are rejected, returns only the root prefix."""
        surviving, accepted_seq = verify_and_prune_tree(
            self.tree_nodes, accepted_node_indices=set()
        )
        self.assertEqual(accepted_seq, [101])
        self.assertEqual([n.token_id for n in surviving], [101])

    def test_verify_and_prune_tree_empty(self):
        """Verify empty tree returns empty tuple."""
        surviving, accepted_seq = verify_and_prune_tree([], accepted_node_indices={0, 1})
        self.assertEqual(surviving, [])
        self.assertEqual(accepted_seq, [])

    def test_verify_and_prune_tree_boolean_mask(self):
        """Verify that verify_and_prune_tree accepts a List[bool] mask as input."""
        # Mask for nodes 0..5: [True, True, False, True, False, True]
        bool_mask = [True, True, False, True, False, True]
        surviving, accepted_seq = verify_and_prune_tree(self.tree_nodes, bool_mask)
        self.assertEqual(accepted_seq, [101, 202, 305])


if __name__ == "__main__":
    print("=" * 80)
    print("  RUNNING CHAPTER 3 UNIT TESTS: SPECULATIVE DECODING & TREE VERIFICATION")
    print("=" * 80)
    unittest.main(verbosity=2)
