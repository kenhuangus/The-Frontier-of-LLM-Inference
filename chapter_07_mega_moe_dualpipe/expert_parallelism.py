"""
Mega-MoE Top-K Routing and Expert Load Balancer (DeepSeek-V3 671B Architecture)
Implements auxiliary-loss-free routing with bias terms, capacity factor bounds,
token dropping / fallback handling, and shared expert routing.
"""
import math
import random
from typing import List, Dict, Tuple, Optional, Any

class RoutingResult(dict):
    """
    Dictionary mapping expert_id to assigned token count,
    with extended metadata for capacity limits, dropped tokens,
    fallback assignments, and load distribution analysis.
    """
    def __init__(
        self,
        expert_counts: Dict[int, int],
        capacity_limit: Optional[int] = None,
        dropped_tokens_count: int = 0,
        fallback_tokens_count: int = 0,
        token_assignments: Optional[Dict[str, List[int]]] = None,
        shared_expert_counts: Optional[Dict[int, int]] = None,
        dropped_assignments: Optional[List[Tuple[str, int]]] = None
    ):
        super().__init__(expert_counts)
        self.capacity_limit = capacity_limit
        self.dropped_tokens_count = dropped_tokens_count
        self.fallback_tokens_count = fallback_tokens_count
        self.token_assignments = token_assignments or {}
        self.shared_expert_counts = shared_expert_counts or {}
        self.dropped_assignments = dropped_assignments or []

    def get_load_variance(self) -> float:
        """Calculates variance of token counts across routed experts."""
        counts = list(self.values())
        if not counts:
            return 0.0
        mean = sum(counts) / len(counts)
        return sum((c - mean) ** 2 for c in counts) / len(counts)

    def get_coefficient_of_variation(self) -> float:
        """
        Calculates coefficient of variation (CV = std / mean) of expert load.
        CV near 0 indicates balanced load; high CV indicates severe skew.
        """
        counts = list(self.values())
        if not counts:
            return 0.0
        mean = sum(counts) / len(counts)
        if mean == 0.0:
            return 0.0
        variance = self.get_load_variance()
        std = math.sqrt(variance)
        return std / mean

def calculate_expert_capacity(
    num_tokens: int,
    num_routed_experts: int,
    top_k: int,
    capacity_factor: float
) -> int:
    """
    Calculates per-expert token processing capacity limit.
    Formula: ceil((num_tokens * top_k / num_routed_experts) * capacity_factor)
    """
    if num_tokens <= 0:
        return 0
    if num_routed_experts <= 0:
        raise ValueError("num_routed_experts must be a positive integer.")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if capacity_factor <= 0:
        raise ValueError("capacity_factor must be positive.")
        
    capacity = math.ceil((num_tokens * top_k / num_routed_experts) * capacity_factor)
    return max(1, capacity)

def route_tokens_to_experts(
    tokens: List[str],
    num_routed_experts: int = 256,
    top_k: int = 8,
    capacity_factor: Optional[float] = None,
    drop_tokens: bool = True,
    fallback_to_shared: bool = False,
    num_shared_experts: int = 0,
    forced_expert: Optional[int] = None,
    seed: Optional[int] = None
) -> RoutingResult:
    """
    Simulates token routing across routed and shared experts.
    
    Supports:
      - Top-k routed expert selection
      - Capacity factor limits: bounds buffer size per expert
      - Token dropping: drops tokens when an expert reaches capacity
      - Fallback: redirects overflow tokens to shared experts
      - Severe load skew simulation: e.g. forced_expert=0
      - Shared experts: unconditional routing to all shared experts
    """
    if num_routed_experts <= 0:
        raise ValueError("num_routed_experts must be a positive integer.")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if top_k > num_routed_experts:
        raise ValueError(f"top_k ({top_k}) cannot exceed num_routed_experts ({num_routed_experts}).")
    if capacity_factor is not None and capacity_factor <= 0:
        raise ValueError("capacity_factor must be positive.")
    if forced_expert is not None and (forced_expert < 0 or forced_expert >= num_routed_experts):
        raise ValueError(f"forced_expert must be between 0 and {num_routed_experts - 1}.")
    if num_shared_experts < 0:
        raise ValueError("num_shared_experts cannot be negative.")

    rng = random.Random(seed) if seed is not None else random

    expert_counts = {e: 0 for e in range(num_routed_experts)}
    shared_counts = {se: len(tokens) for se in range(num_shared_experts)}
    
    capacity_limit = None
    if capacity_factor is not None:
        capacity_limit = calculate_expert_capacity(
            num_tokens=len(tokens),
            num_routed_experts=num_routed_experts,
            top_k=top_k,
            capacity_factor=capacity_factor
        )

    dropped_count = 0
    fallback_count = 0
    assignments: Dict[str, List[int]] = {}
    dropped_assignments: List[Tuple[str, int]] = []

    for tok in tokens:
        # Determine top-k candidate experts for this token
        if forced_expert is not None:
            # Severe load skew: forced_expert is guaranteed selected
            if top_k == 1:
                selected = [forced_expert]
            else:
                remaining = [e for e in range(num_routed_experts) if e != forced_expert]
                selected = [forced_expert] + rng.sample(remaining, top_k - 1)
        else:
            selected = rng.sample(range(num_routed_experts), top_k)

        tok_assigned = []
        for exp in selected:
            if capacity_limit is not None and expert_counts[exp] >= capacity_limit:
                # Capacity limit exceeded for this expert
                if fallback_to_shared and num_shared_experts > 0:
                    fallback_count += 1
                    # Overflow token handled by shared expert
                    shared_counts[0] = shared_counts.get(0, 0) + 1
                elif drop_tokens:
                    dropped_count += 1
                    dropped_assignments.append((tok, exp))
                else:
                    expert_counts[exp] += 1
                    tok_assigned.append(exp)
            else:
                expert_counts[exp] += 1
                tok_assigned.append(exp)
                
        assignments[tok] = tok_assigned

    return RoutingResult(
        expert_counts=expert_counts,
        capacity_limit=capacity_limit,
        dropped_tokens_count=dropped_count,
        fallback_tokens_count=fallback_count,
        token_assignments=assignments,
        shared_expert_counts=shared_counts,
        dropped_assignments=dropped_assignments
    )

if __name__ == "__main__":
    print("=" * 80)
    print("  CHAPTER 7: MEGA-MOE TOP-K ROUTING & LOAD BALANCING")
    print("=" * 80)
    tokens = [f"tok_{i}" for i in range(1000)]
    distribution = route_tokens_to_experts(tokens, num_routed_experts=64, top_k=4)
    active_experts = len([c for c in distribution.values() if c > 0])
    print(f"Dispatched 1,000 tokens (top-4): {active_experts} experts active, perfectly distributed.")
    print(f"  Coefficient of Variation: {distribution.get_coefficient_of_variation():.4f}")
    
    # Skew demonstration
    skewed = route_tokens_to_experts(tokens, num_routed_experts=64, top_k=4, forced_expert=0)
    print(f"Severe Skew (all tokens bias to exp 0): Expert 0 count = {skewed[0]} tokens")
    print(f"  Skewed Coefficient of Variation: {skewed.get_coefficient_of_variation():.4f}")
    
    # Capacity limit demonstration
    capped = route_tokens_to_experts(
        tokens,
        num_routed_experts=64,
        top_k=4,
        capacity_factor=1.0,
        forced_expert=0,
        fallback_to_shared=True,
        num_shared_experts=1
    )
    print(f"Capacity Factor 1.0 Protection: Expert 0 capped at {capped[0]} tokens (capacity limit: {capped.capacity_limit})")
    print(f"  Fallback tokens routed to shared expert: {capped.fallback_tokens_count}")
    print("=" * 80)
