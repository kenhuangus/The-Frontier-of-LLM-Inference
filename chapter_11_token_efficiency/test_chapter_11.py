"""
Comprehensive Unit Test Suite for Chapter 11: Token Efficiency
Covers:
  - factory_spend_decomposition.py:
      * 0 tokens and zero multipliers
      * 100% cached vs 0% cached
      * prefill vs decode cost weighting
      * negative or invalid inputs (type & value errors)
      * granular decomposition consistency
  - gateway_affinity_router.py:
      * consistent hashing node distribution
      * cache hit rate under repeated prefixes (deterministic affinity)
      * node addition/removal rebalancing churn
      * empty node pool handling and worker eviction
      * sticky session pinning and TTL expiration
  - prompt_cache_multipliers.py:
      * multi-turn conversation depth scaling and asymptotic convergence
      * run-length session billing identity [True, True, False, True] -> 2.70
      * tiered cache pricing models and token threshold activation
      * provider-specific cache tariff profiles (Anthropic, OpenAI, Gemini, DeepSeek)
"""
import math
import os
import sys
import unittest
from typing import List, Dict

# Ensure local chapter module directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from factory_spend_decomposition import (
    factory_spend,
    decompose_factory_spend,
)
from gateway_affinity_router import (
    ConsistentHashRing,
    GatewayAffinityRouter,
)
from prompt_cache_multipliers import (
    billed_prefix,
    session_prefix_bill,
    multi_turn_cache_depth,
    tiered_cache_cost,
    PROVIDER_CACHE_PROFILES,
)


class TestFactorySpendDecomposition(unittest.TestCase):
    """Test suite for factory spend decomposition and cost equations."""

    def test_baseline_and_optimized_spend(self):
        """Verify standard baseline and optimized factory spend from Chapter 11."""
        base = dict(users=100, sessions=5, turns=20, reqs=3, tokens=80000, price=3e-6)
        optimized = dict(users=100, sessions=5, turns=20, reqs=3, tokens=20000, price=1e-6)
        
        spend_base = factory_spend(**base)
        spend_opt = factory_spend(**optimized)
        
        self.assertAlmostEqual(spend_base, 7200.0, places=2)
        self.assertAlmostEqual(spend_opt, 600.0, places=2)
        self.assertAlmostEqual(spend_base - spend_opt, 6600.0, places=2)

    def test_zero_tokens_returns_zero_spend(self):
        """Verify spend is 0.0 when tokens per request is zero."""
        spend = factory_spend(users=100, sessions=5, turns=20, reqs=3, tokens=0, price=3e-6)
        self.assertEqual(spend, 0.0)

    def test_zero_multipliers_edge_cases(self):
        """Verify that zero in any multiplier dimension yields zero spend."""
        self.assertEqual(factory_spend(0, 5, 20, 3, 80000, 3e-6), 0.0)
        self.assertEqual(factory_spend(100, 0, 20, 3, 80000, 3e-6), 0.0)
        self.assertEqual(factory_spend(100, 5, 0, 3, 80000, 3e-6), 0.0)
        self.assertEqual(factory_spend(100, 5, 20, 0, 80000, 3e-6), 0.0)
        self.assertEqual(factory_spend(100, 5, 20, 3, 80000, 0.0), 0.0)

    def test_negative_inputs_raise_value_error(self):
        """Verify negative values in any multiplier parameter raise ValueError."""
        with self.assertRaises(ValueError):
            factory_spend(-1, 5, 20, 3, 80000, 3e-6)
        with self.assertRaises(ValueError):
            factory_spend(100, -5, 20, 3, 80000, 3e-6)
        with self.assertRaises(ValueError):
            factory_spend(100, 5, -20, 3, 80000, 3e-6)
        with self.assertRaises(ValueError):
            factory_spend(100, 5, 20, -3, 80000, 3e-6)
        with self.assertRaises(ValueError):
            factory_spend(100, 5, 20, 3, -1000, 3e-6)
        with self.assertRaises(ValueError):
            factory_spend(100, 5, 20, 3, 80000, -3e-6)

    def test_invalid_types_raise_type_error(self):
        """Verify passing non-numeric or boolean types raises TypeError."""
        with self.assertRaises(TypeError):
            factory_spend("100", 5, 20, 3, 80000, 3e-6)  # type: ignore
        with self.assertRaises(TypeError):
            factory_spend(100, True, 20, 3, 80000, 3e-6)  # type: ignore
        with self.assertRaises(TypeError):
            factory_spend(100, 5, None, 3, 80000, 3e-6)  # type: ignore

    def test_cache_hit_rate_bounds_validation(self):
        """Verify cache_hit_rate must be in [0.0, 1.0]."""
        with self.assertRaises(ValueError):
            factory_spend(10, 2, 5, 1, 1000, 1e-6, cache_hit_rate=-0.1)
        with self.assertRaises(ValueError):
            factory_spend(10, 2, 5, 1, 1000, 1e-6, cache_hit_rate=1.1)

    def test_100_percent_cached_vs_0_percent_cached(self):
        """
        Verify economics of 100% cached vs 0% cached.
        With cache_read_mult=0.10, 100% cached prefill costs exactly 10% of 0% cached.
        """
        spend_0_cached = factory_spend(
            users=50, sessions=4, turns=10, reqs=2, tokens=50000, price=2e-6,
            cache_hit_rate=0.0, cache_read_mult=0.10, prefill_ratio=1.0
        )
        spend_100_cached = factory_spend(
            users=50, sessions=4, turns=10, reqs=2, tokens=50000, price=2e-6,
            cache_hit_rate=1.0, cache_read_mult=0.10, prefill_ratio=1.0
        )

        self.assertAlmostEqual(spend_0_cached, 400.0, places=2)
        self.assertAlmostEqual(spend_100_cached, 40.0, places=2)
        # 100% cache reduces cost by 90% (10x reduction)
        self.assertAlmostEqual(spend_100_cached / spend_0_cached, 0.10, places=4)

    def test_prefill_vs_decode_cost_weighting(self):
        """
        Verify prefill vs decode cost weighting where decode is 5x more expensive than prefill.
        Tokens split 80% prefill, 20% decode.
        """
        users, sessions, turns, reqs = 10, 5, 10, 1  # 500 requests
        tokens = 10000
        price = 1e-6
        # Total tokens = 5,000,000
        # Prefill: 80% = 4,000,000 tokens @ 1.0x price = $4.00
        # Decode:  20% = 1,000,000 tokens @ 5.0x price = $5.00
        # Total expected = $9.00
        spend = factory_spend(
            users, sessions, turns, reqs, tokens, price,
            prefill_ratio=0.8,
            prefill_weight=1.0,
            decode_weight=5.0,
            cache_hit_rate=0.0
        )
        self.assertAlmostEqual(spend, 9.00, places=4)

    def test_decomposed_factory_spend_consistency(self):
        """Verify decompose_factory_spend produces self-consistent metrics."""
        decomp = decompose_factory_spend(
            users=20, sessions=3, turns=15, reqs=2, tokens=40000, price=2e-6,
            cache_hit_rate=0.70, cache_read_mult=0.10, prefill_ratio=0.85,
            prefill_weight=1.0, decode_weight=4.0
        )
        self.assertEqual(decomp["total_requests"], 20 * 3 * 15 * 2)
        self.assertEqual(decomp["total_tokens"], decomp["total_requests"] * 40000)
        self.assertEqual(decomp["prefill_tokens"] + decomp["decode_tokens"], decomp["total_tokens"])
        self.assertAlmostEqual(
            decomp["cached_prefill_tokens"] + decomp["uncached_prefill_tokens"],
            decomp["prefill_tokens"],
            places=2
        )
        self.assertGreater(decomp["cache_savings"], 0.0)
        self.assertLess(decomp["total_spend"], decomp["baseline_uncached_spend"])


class TestGatewayAffinityRouter(unittest.TestCase):
    """Test suite for Gateway Affinity Router, consistent hashing, and worker pools."""

    def setUp(self):
        self.workers = [
            {"id": "worker-gpu-1", "load": 40},
            {"id": "worker-gpu-2", "load": 15},
            {"id": "worker-gpu-3", "load": 25},
        ]
        self.router = GatewayAffinityRouter(self.workers)

    def test_least_loaded_routing(self):
        """Verify turn 1 routes to the worker with minimum load (worker-gpu-2)."""
        target = self.router.route("session-abc", pin=False)
        self.assertEqual(target, "worker-gpu-2")

    def test_sticky_session_pinning_within_ttl(self):
        """Verify session pinning maintains affinity even if worker loads change."""
        # Turn 1 lands on worker-gpu-2 and pins
        w1 = self.router.route("session-xyz", pin=True, now=100.0, ttl_s=300.0)
        self.assertEqual(w1, "worker-gpu-2")

        # Worker 2 load spikes to 99
        for w in self.router.workers:
            if w["id"] == "worker-gpu-2":
                w["load"] = 99

        # Turn 2 within TTL remains pinned to worker-gpu-2
        w2 = self.router.route("session-xyz", pin=True, now=200.0, ttl_s=300.0)
        self.assertEqual(w2, "worker-gpu-2")

    def test_session_pinning_ttl_expiration(self):
        """Verify expired session pin re-routes based on current load."""
        w1 = self.router.route("session-ttl", pin=True, now=10.0, ttl_s=100.0)
        self.assertEqual(w1, "worker-gpu-2")

        # Worker 2 load now heavy, worker 3 is lightest
        for w in self.router.workers:
            if w["id"] == "worker-gpu-2":
                w["load"] = 80
            if w["id"] == "worker-gpu-3":
                w["load"] = 10

        # At now=120.0 (elapsed 110s > 100s TTL), session re-routes to worker-gpu-3
        w2 = self.router.route("session-ttl", pin=True, now=120.0, ttl_s=100.0)
        self.assertEqual(w2, "worker-gpu-3")

    def test_empty_node_pool_handling(self):
        """Verify empty worker pool raises RuntimeError gracefully."""
        empty_router = GatewayAffinityRouter([])
        with self.assertRaises(RuntimeError) as ctx:
            empty_router.route("session-fail")
        self.assertIn("empty", str(ctx.exception).lower())

    def test_worker_pool_drain_and_recovery(self):
        """Verify dynamic removal of all workers raises RuntimeError, and recovery works."""
        self.router.remove_worker("worker-gpu-1")
        self.router.remove_worker("worker-gpu-2")
        self.router.remove_worker("worker-gpu-3")
        self.assertFalse(self.router.has_workers())

        with self.assertRaises(RuntimeError):
            self.router.route("sess-after-drain")

        # Recover by adding a new worker
        self.router.add_worker({"id": "worker-gpu-recovered", "load": 5})
        self.assertTrue(self.router.has_workers())
        target = self.router.route("sess-after-drain")
        self.assertEqual(target, "worker-gpu-recovered")

    def test_pinned_worker_removal_purges_pin(self):
        """Verify that removing a worker cleans up its sticky references."""
        w1 = self.router.route("sess-pinned", pin=True, now=10.0)
        self.assertEqual(w1, "worker-gpu-2")

        # Remove worker 2
        self.router.remove_worker("worker-gpu-2")
        # Next route should not return removed worker-gpu-2, but route to an active worker
        w2 = self.router.route("sess-pinned", pin=True, now=20.0)
        self.assertNotEqual(w2, "worker-gpu-2")
        self.assertIn(w2, ["worker-gpu-1", "worker-gpu-3"])

    def test_consistent_hashing_distribution(self):
        """
        Verify consistent hash ring distributes keys fairly across workers.
        With 3 workers and 600 distinct keys, no worker should receive 0% or > 60%.
        """
        ring = ConsistentHashRing(["worker-1", "worker-2", "worker-3"], vnodes=100)
        counts = {"worker-1": 0, "worker-2": 0, "worker-3": 0}

        num_keys = 600
        for i in range(num_keys):
            node = ring.get_node(f"prefix_key_prompt_{i:04d}")
            self.assertIsNotNone(node)
            counts[node] += 1

        for worker_id, count in counts.items():
            fraction = count / num_keys
            # Each worker gets a fair share (between 20% and 48% with 100 vnodes)
            self.assertGreater(fraction, 0.20, f"{worker_id} under-allocated: {fraction:.2f}")
            self.assertLess(fraction, 0.48, f"{worker_id} over-allocated: {fraction:.2f}")

    def test_cache_hit_rate_under_repeated_prefixes(self):
        """
        Verify consistent hashing gives 100% cache hit rate for repeated prefixes.
        """
        router = GatewayAffinityRouter(self.workers, strategy="consistent_hash")
        prefixes = [f"shared_system_prompt_v{i}" for i in range(10)]

        # Route 100 requests among the 10 prefixes
        first_routing = {p: router.route_prefix(p) for p in prefixes}

        for p in prefixes:
            for _ in range(10):
                target = router.route_prefix(p)
                self.assertEqual(target, first_routing[p], f"Prefix {p} routed inconsistently")

    def test_node_addition_rebalancing_minimal_churn(self):
        """
        Verify consistent hashing rebalancing when adding a 4th node.
        Only approximately 1/(N+1) ~ 25% of keys should rebalance to the new node,
        while ~75% remain on their existing node.
        """
        initial_nodes = ["worker-1", "worker-2", "worker-3"]
        ring = ConsistentHashRing(initial_nodes, vnodes=100)

        keys = [f"document_context_block_{i:05d}" for i in range(1000)]
        initial_assignments = {k: ring.get_node(k) for k in keys}

        # Add 4th node
        ring.add_node("worker-4")
        rebalanced_assignments = {k: ring.get_node(k) for k in keys}

        moved_to_new_node = 0
        moved_between_old_nodes = 0
        remained_stable = 0

        for k in keys:
            orig = initial_assignments[k]
            new = rebalanced_assignments[k]
            if new == "worker-4":
                moved_to_new_node += 1
            elif new == orig:
                remained_stable += 1
            else:
                moved_between_old_nodes += 1

        # Keys moving between surviving nodes must be zero in consistent hashing
        self.assertEqual(moved_between_old_nodes, 0)
        # Fraction moving to new node should be roughly 25% (allowed tolerance 18% - 32%)
        new_node_fraction = moved_to_new_node / len(keys)
        self.assertGreaterEqual(new_node_fraction, 0.18)
        self.assertLessEqual(new_node_fraction, 0.32)
        # Remainder stayed stable on original workers
        self.assertEqual(remained_stable + moved_to_new_node, len(keys))

    def test_node_removal_rebalancing_minimal_churn(self):
        """
        Verify consistent hashing rebalancing when removing a node.
        Keys assigned to non-removed nodes MUST remain unchanged (100% stability).
        """
        ring = ConsistentHashRing(["worker-1", "worker-2", "worker-3", "worker-4"], vnodes=100)
        keys = [f"query_prefix_hash_{i:05d}" for i in range(1000)]
        initial_assignments = {k: ring.get_node(k) for k in keys}

        # Remove worker-2
        ring.remove_node("worker-2")
        post_removal_assignments = {k: ring.get_node(k) for k in keys}

        for k in keys:
            orig = initial_assignments[k]
            new = post_removal_assignments[k]
            if orig != "worker-2":
                # Keys that did not belong to worker-2 must stay on their existing node
                self.assertEqual(orig, new, f"Key {k} migrated unnecessarily from {orig} to {new}")
            else:
                # Keys that were on worker-2 must now be on one of the surviving nodes
                self.assertIn(new, ["worker-1", "worker-3", "worker-4"])


class TestPromptCacheMultipliers(unittest.TestCase):
    """Test suite for prompt cache multipliers, session billing, and depth scaling."""

    def test_billed_prefix_two_use_identity(self):
        """
        Verify Chapter 11 two-use identity:
        With write_mult=1.25 and read_mult=0.10, two uses bill at 1.35x.
        """
        self.assertAlmostEqual(billed_prefix(1), 1.25, places=3)
        self.assertAlmostEqual(billed_prefix(2), 1.35, places=3)
        self.assertAlmostEqual(billed_prefix(3), 1.45, places=3)
        self.assertAlmostEqual(billed_prefix(5), 1.65, places=3)
        self.assertAlmostEqual(billed_prefix(20), 3.15, places=3)

    def test_billed_prefix_zero_and_negative_uses(self):
        """Verify billed_prefix returns 0.0 for 0 or negative uses."""
        self.assertEqual(billed_prefix(0), 0.0)
        self.assertEqual(billed_prefix(-1), 0.0)
        self.assertEqual(billed_prefix(-100), 0.0)

    def test_session_prefix_bill_four_turn_log(self):
        """
        Verify Chapter 11 Paragraph 244 canonical example:
        session_prefix_bill([True, True, False, True]) == 2.70.
        The False turn starts a new write, forming two 2-use runs: 1.35 + 1.35 = 2.70.
        """
        cost = session_prefix_bill([True, True, False, True])
        self.assertAlmostEqual(cost, 2.70, places=2)

    def test_session_prefix_bill_edge_cases(self):
        """Verify session_prefix_bill on empty list, all cold, and all warm runs."""
        # Empty list -> 0.0
        self.assertEqual(session_prefix_bill([]), 0.0)

        # Single cold turn -> 1 write = 1.25
        self.assertAlmostEqual(session_prefix_bill([False]), 1.25, places=2)

        # Single warm turn -> 1 write = 1.25
        self.assertAlmostEqual(session_prefix_bill([True]), 1.25, places=2)

        # Two consecutive cold writes -> 1.25 + 1.25 = 2.50
        self.assertAlmostEqual(session_prefix_bill([False, False]), 2.50, places=2)

        # Three uninterrupted warm turns -> billed_prefix(3) = 1.45
        self.assertAlmostEqual(session_prefix_bill([True, True, True]), 1.45, places=2)

    def test_multi_turn_conversation_depth(self):
        """
        Verify multi-turn conversation depth unit economics:
        - Turn 1 pays write tax (billed_factor > uncached_factor, negative savings)
        - Turn 2 breaks even and yields positive discount
        - As turn depth approaches 100, discount asymptotically approaches 90% (for read_mult=0.10).
        """
        t1 = multi_turn_cache_depth(1)
        self.assertAlmostEqual(t1["billed_factor"], 1.25)
        self.assertLess(t1["discount_percent"], 0.0)  # -25% penalty on 1st turn

        t2 = multi_turn_cache_depth(2)
        self.assertAlmostEqual(t2["billed_factor"], 1.35)
        self.assertGreater(t2["discount_percent"], 30.0)  # (2.0 - 1.35) / 2.0 = 32.5%

        t20 = multi_turn_cache_depth(20)
        self.assertAlmostEqual(t20["billed_factor"], 3.15)
        self.assertGreater(t20["discount_percent"], 84.0)

        t100 = multi_turn_cache_depth(100)
        # At 100 turns: billed = 1.25 + 99 * 0.10 = 11.15x vs 100x uncached
        self.assertAlmostEqual(t100["billed_factor"], 11.15)
        self.assertAlmostEqual(t100["discount_percent"], 88.85, places=2)

        # Marginal cost for turn > 1 is strictly read_mult (0.10)
        self.assertAlmostEqual(t2["marginal_cost"], 0.10)
        self.assertAlmostEqual(t100["marginal_cost"], 0.10)

        # Monotonically decreasing average cost per turn
        depths = [1, 2, 5, 10, 20, 50, 100]
        avg_costs = [multi_turn_cache_depth(d)["avg_cost_per_turn"] for d in depths]
        for i in range(len(avg_costs) - 1):
            self.assertGreater(avg_costs[i], avg_costs[i + 1])

    def test_multi_turn_cache_depth_zero_or_negative(self):
        """Verify depth <= 0 returns zero factors."""
        res_zero = multi_turn_cache_depth(0)
        self.assertEqual(res_zero["billed_factor"], 0.0)
        res_neg = multi_turn_cache_depth(-5)
        self.assertEqual(res_neg["billed_factor"], 0.0)

    def test_tiered_provider_profiles_tariffs(self):
        """
        Verify dated provider cache profiles (Anthropic, OpenAI, Gemini, DeepSeek):
        - Anthropic 5m vs 1h vs Fable
        - OpenAI GPT-5 vs Sol
        - Gemini 2.0 explicit (75% off) vs Gemini 2.5 implicit (90% off)
        """
        # Anthropic 5m
        p_anth_5m = PROVIDER_CACHE_PROFILES["anthropic_5m"]
        self.assertEqual(p_anth_5m["write_mult"], 1.25)
        self.assertEqual(p_anth_5m["read_mult"], 0.10)

        # Anthropic 1h
        p_anth_1h = PROVIDER_CACHE_PROFILES["anthropic_1h"]
        self.assertEqual(p_anth_1h["write_mult"], 2.00)
        self.assertEqual(p_anth_1h["read_mult"], 0.10)

        # Anthropic Fable / Mythos (0.025x read)
        p_fable = PROVIDER_CACHE_PROFILES["anthropic_fable"]
        self.assertEqual(p_fable["read_mult"], 0.025)

        # OpenAI Sol (0.05x read)
        p_sol = PROVIDER_CACHE_PROFILES["openai_sol"]
        self.assertEqual(p_sol["read_mult"], 0.05)

        # Gemini 2.0 (25% read / 75% off) vs Gemini 2.5 (10% read / 90% off)
        self.assertEqual(PROVIDER_CACHE_PROFILES["gemini_2_0_explicit"]["read_mult"], 0.25)
        self.assertEqual(PROVIDER_CACHE_PROFILES["gemini_2_5_implicit"]["read_mult"], 0.10)

        # DeepSeek V4 (10% read)
        self.assertEqual(PROVIDER_CACHE_PROFILES["deepseek_v4"]["read_mult"], 0.10)

    def test_tiered_cache_token_threshold_activation(self):
        """
        Verify tiered cache cost activation thresholds:
        - Below 1024 tokens: cache inactive, billed at 1.0x per turn uncached.
        - At or above 1024 tokens: cache activated, billed with provider multiplier.
        """
        # 500 tokens (below 1024 min_tokens)
        sub_thresh = tiered_cache_cost(tokens=500, n_uses=5, provider_profile="anthropic_5m")
        self.assertFalse(sub_thresh["cache_active"])
        self.assertEqual(sub_thresh["billed_factor"], 5.0)  # full uncached price
        self.assertEqual(sub_thresh["discount_percent"], 0.0)

        # 2048 tokens (above 1024 min_tokens)
        above_thresh = tiered_cache_cost(tokens=2048, n_uses=5, provider_profile="anthropic_5m")
        self.assertTrue(above_thresh["cache_active"])
        # 1.25 + 4 * 0.10 = 1.65x
        self.assertAlmostEqual(above_thresh["billed_factor"], 1.65)
        self.assertAlmostEqual(above_thresh["discount_percent"], 67.0, places=1)


if __name__ == "__main__":
    unittest.main()
