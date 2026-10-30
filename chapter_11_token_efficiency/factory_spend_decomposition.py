"""
Software-Factory Cost Equation Decomposition
Models the six multiplying terms: Spend = Users * Sessions * Turns * Reqs * Tokens * Price
Expanded with prompt-cache reuse, prefill vs decode cost weighting, and input validation.
"""
from typing import Dict, Any, Union


def factory_spend(
    users: int,
    sessions: int,
    turns: int,
    reqs: int,
    tokens: int,
    price: float,
    *,
    cache_hit_rate: float = 0.0,
    cache_read_mult: float = 0.10,
    prefill_ratio: float = 1.0,
    prefill_weight: float = 1.0,
    decode_weight: float = 1.0,
) -> float:
    """
    Calculates total software-factory spend across the 6 multiplying terms,
    with support for prompt cache hit rate and prefill vs decode cost weighting.

    Parameters:
        users: Active user count
        sessions: Sessions per user
        turns: Turns per session
        reqs: Requests per turn
        tokens: Tokens per request
        price: Base price per token
        cache_hit_rate: Fraction of prefill tokens served from warm cache [0.0, 1.0]
        cache_read_mult: Multiplier for cached reads (e.g., 0.10 for 90% discount)
        prefill_ratio: Fraction of tokens that are prefill/prompt [0.0, 1.0]
        prefill_weight: Relative cost weighting of prefill tokens
        decode_weight: Relative cost weighting of decode tokens
    """
    # Type validation
    factors = {"users": users, "sessions": sessions, "turns": turns, "reqs": reqs, "tokens": tokens, "price": price}
    for name, val in factors.items():
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"Argument '{name}' must be numeric (int or float), got {type(val).__name__}")
        if val < 0:
            raise ValueError(f"Argument '{name}' must be non-negative, got {val}")

    if not isinstance(cache_hit_rate, (int, float)) or isinstance(cache_hit_rate, bool):
        raise TypeError("cache_hit_rate must be numeric")
    if not (0.0 <= cache_hit_rate <= 1.0):
        raise ValueError(f"cache_hit_rate must be between 0.0 and 1.0, got {cache_hit_rate}")

    if not isinstance(prefill_ratio, (int, float)) or isinstance(prefill_ratio, bool):
        raise TypeError("prefill_ratio must be numeric")
    if not (0.0 <= prefill_ratio <= 1.0):
        raise ValueError(f"prefill_ratio must be between 0.0 and 1.0, got {prefill_ratio}")

    if prefill_weight < 0 or decode_weight < 0:
        raise ValueError("Prefill and decode weights must be non-negative")

    # Zero multiplier fast-paths
    if users == 0 or sessions == 0 or turns == 0 or reqs == 0 or tokens == 0 or price == 0.0:
        return 0.0

    total_requests = users * sessions * turns * reqs
    total_tokens = total_requests * tokens

    # Prefill vs decode partitioning
    prefill_tokens = total_tokens * prefill_ratio
    decode_tokens = total_tokens * (1.0 - prefill_ratio)

    # Cache application to prefill tokens
    uncached_prefill = prefill_tokens * (1.0 - cache_hit_rate)
    cached_prefill = prefill_tokens * cache_hit_rate

    prefill_unit_cost = price * prefill_weight
    decode_unit_cost = price * decode_weight

    prefill_cost = (uncached_prefill * prefill_unit_cost) + (cached_prefill * prefill_unit_cost * cache_read_mult)
    decode_cost = decode_tokens * decode_unit_cost

    return prefill_cost + decode_cost


def decompose_factory_spend(
    users: int,
    sessions: int,
    turns: int,
    reqs: int,
    tokens: int,
    price: float,
    cache_hit_rate: float = 0.0,
    cache_read_mult: float = 0.10,
    prefill_ratio: float = 0.8,
    prefill_weight: float = 1.0,
    decode_weight: float = 5.0,
) -> Dict[str, Any]:
    """
    Decomposes software-factory spend into granular structural components.
    """
    total_spend = factory_spend(
        users, sessions, turns, reqs, tokens, price,
        cache_hit_rate=cache_hit_rate,
        cache_read_mult=cache_read_mult,
        prefill_ratio=prefill_ratio,
        prefill_weight=prefill_weight,
        decode_weight=decode_weight,
    )
    baseline_spend = factory_spend(
        users, sessions, turns, reqs, tokens, price,
        cache_hit_rate=0.0,
        cache_read_mult=cache_read_mult,
        prefill_ratio=prefill_ratio,
        prefill_weight=prefill_weight,
        decode_weight=decode_weight,
    )
    total_requests = users * sessions * turns * reqs
    total_tokens = total_requests * tokens
    prefill_tokens = total_tokens * prefill_ratio
    decode_tokens = total_tokens * (1.0 - prefill_ratio)

    return {
        "users": users,
        "sessions": sessions,
        "turns": turns,
        "reqs": reqs,
        "tokens": tokens,
        "base_price": price,
        "total_requests": total_requests,
        "total_tokens": total_tokens,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "cached_prefill_tokens": prefill_tokens * cache_hit_rate,
        "uncached_prefill_tokens": prefill_tokens * (1.0 - cache_hit_rate),
        "total_spend": total_spend,
        "baseline_uncached_spend": baseline_spend,
        "cache_savings": baseline_spend - total_spend,
        "effective_price_per_token": (total_spend / total_tokens) if total_tokens > 0 else 0.0,
    }


if __name__ == "__main__":
    print("=" * 80)
    print("  CHAPTER 11: SOFTWARE-FACTORY COST EQUATION DECOMPOSITION")
    print("=" * 80)
    base = dict(users=100, sessions=5, turns=20, reqs=3, tokens=80000, price=3e-6)
    optimized = dict(users=100, sessions=5, turns=20, reqs=3, tokens=20000, price=1e-6) # 4x prefix cache + routing
    
    spend_base = factory_spend(**base)
    spend_opt = factory_spend(**optimized)
    print(f"Baseline Enterprise Spend : ${spend_base:,.2f} / month")
    print(f"Optimized Factory Spend    : ${spend_opt:,.2f} / month (Saved ${spend_base - spend_opt:,.2f})")
    
    # Detailed decomposition example
    decomp = decompose_factory_spend(**base, cache_hit_rate=0.75, prefill_ratio=0.85, decode_weight=5.0)
    print(f"Decomposed Spend (75% cache): ${decomp['total_spend']:,.2f} / month (Saved ${decomp['cache_savings']:,.2f})")
    print("=" * 80)
