"""
Provider Prompt Cache Multipliers & Billed Token Arithmetic
Calculates unit economics of multi-turn prefix reuse across leading foundation model APIs,
including multi-turn depth scaling, run-length session billing, and tiered provider models.
"""
from typing import List, Dict, Optional, Any

PROVIDER_CACHE_PROFILES: Dict[str, Dict[str, Any]] = {
    "anthropic_5m": {"write_mult": 1.25, "read_mult": 0.10, "ttl_s": 300, "min_tokens": 1024},
    "anthropic_1h": {"write_mult": 2.00, "read_mult": 0.10, "ttl_s": 3600, "min_tokens": 1024},
    "anthropic_fable": {"write_mult": 1.25, "read_mult": 0.025, "ttl_s": 300, "min_tokens": 1024},
    "openai_gpt5": {"write_mult": 1.25, "read_mult": 0.10, "ttl_s": 300, "min_tokens": 1024},
    "openai_sol": {"write_mult": 1.25, "read_mult": 0.05, "ttl_s": 300, "min_tokens": 1024},
    "gemini_2_0_explicit": {"write_mult": 1.00, "read_mult": 0.25, "ttl_s": 300, "min_tokens": 32768},
    "gemini_2_5_implicit": {"write_mult": 1.00, "read_mult": 0.10, "ttl_s": 300, "min_tokens": 32768},
    "deepseek_v4": {"write_mult": 1.00, "read_mult": 0.10, "ttl_s": 300, "min_tokens": 64},
}


def billed_prefix(n_uses: int, write_mult: float = 1.25, read_mult: float = 0.10) -> float:
    """Return prefix cost in units of one uncached send."""
    if n_uses <= 0:
        return 0.0
    return write_mult + (n_uses - 1) * read_mult


def session_prefix_bill(ttl_ok_flags: List[bool], write_mult: float = 1.25, read_mult: float = 0.10) -> float:
    """
    Calculates total session bill across runs of warm cache hits and cold writes.
    ttl_ok_flags[i] True means turn i shares the warm prefix of the current run.
    A False turn forces a new cache write and starts a new prefix run.
    """
    if not ttl_ok_flags:
        return 0.0

    cost, run = 0.0, 0
    for warm in ttl_ok_flags:
        if warm:
            run += 1
        else:
            cost += billed_prefix(run, write_mult, read_mult)
            run = 1
    return cost + billed_prefix(run, write_mult, read_mult)


def multi_turn_cache_depth(
    depth: int,
    write_mult: float = 1.25,
    read_mult: float = 0.10,
) -> Dict[str, float]:
    """
    Evaluates unit economics across multi-turn conversation depth.
    Computes billed factor, baseline uncached cost, discount percentage,
    average amortized cost per turn, and marginal cost for the latest turn.
    """
    if depth <= 0:
        return {
            "depth": 0,
            "billed_factor": 0.0,
            "uncached_factor": 0.0,
            "discount_percent": 0.0,
            "avg_cost_per_turn": 0.0,
            "marginal_cost": 0.0,
        }

    billed = billed_prefix(depth, write_mult=write_mult, read_mult=read_mult)
    uncached = float(depth)
    discount = (uncached - billed) / uncached * 100.0 if uncached > 0 else 0.0
    avg_cost = billed / depth
    marginal = write_mult if depth == 1 else read_mult

    return {
        "depth": depth,
        "billed_factor": billed,
        "uncached_factor": uncached,
        "discount_percent": discount,
        "avg_cost_per_turn": avg_cost,
        "marginal_cost": marginal,
    }


def tiered_cache_cost(
    tokens: int,
    n_uses: int,
    provider_profile: str = "anthropic_5m",
    custom_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Calculates billed cost considering token activation thresholds and provider-specific tiers.
    If tokens < min_tokens, caching is not activated and each call bills at 1.0x uncached.
    """
    profile = custom_profile or PROVIDER_CACHE_PROFILES.get(provider_profile, PROVIDER_CACHE_PROFILES["anthropic_5m"])
    min_tokens = profile.get("min_tokens", 1024)
    write_mult = profile.get("write_mult", 1.25)
    read_mult = profile.get("read_mult", 0.10)

    uncached_total_factor = float(n_uses) if n_uses > 0 else 0.0

    if tokens < min_tokens or n_uses <= 0:
        # Below cache threshold: no discount available
        billed_factor = uncached_total_factor
        cache_active = False
    else:
        billed_factor = billed_prefix(n_uses, write_mult=write_mult, read_mult=read_mult)
        cache_active = True

    discount = ((uncached_total_factor - billed_factor) / uncached_total_factor * 100.0) if uncached_total_factor > 0 else 0.0

    return {
        "tokens": tokens,
        "n_uses": n_uses,
        "cache_active": cache_active,
        "min_tokens_required": min_tokens,
        "write_mult": write_mult,
        "read_mult": read_mult,
        "billed_factor": billed_factor,
        "uncached_factor": uncached_total_factor,
        "discount_percent": discount,
    }


if __name__ == "__main__":
    print("=" * 80)
    print("  CHAPTER 11: PROMPT CACHE MULTIPLIER CALCULATOR")
    print("=" * 80)
    for turns in [1, 5, 20, 50]:
        cost = billed_prefix(turns)
        uncached = float(turns)
        discount = (uncached - cost) / uncached * 100.0
        print(f"Turn Count: {turns:>2} | Billed Factor: {cost:>5.2f}x | Uncached: {uncached:>5.2f}x | Savings: {discount:>5.1f}%")
    print("=" * 80)
    
    # 4-turn run verification: [True, True, False, True] -> 2.70
    demo_run = session_prefix_bill([True, True, False, True])
    print(f"Session [True, True, False, True] Billed Factor: {demo_run:.2f}x (Expected: 2.70x)")
    print("=" * 80)
