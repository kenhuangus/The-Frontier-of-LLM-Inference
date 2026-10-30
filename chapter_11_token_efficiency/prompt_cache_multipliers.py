"""
Provider Prompt Cache Multipliers & Billed Token Arithmetic
Calculates unit economics of multi-turn prefix reuse across leading foundation model APIs.
"""
def billed_prefix(n_uses: int, write_mult: float = 1.25, read_mult: float = 0.10) -> float:
    """Return prefix cost in units of one uncached send."""
    if n_uses <= 0:
        return 0.0
    return write_mult + (n_uses - 1) * read_mult

def session_prefix_bill(ttl_ok_flags: list[bool], write_mult: float = 1.25, read_mult: float = 0.10) -> float:
    """Calculates session bill where ttl_ok_flags[i] is True if turn i shares warm cache."""
    total_cost = 0.0
    for ok in ttl_ok_flags:
        if ok:
            total_cost += read_mult
        else:
            total_cost += write_mult
    return total_cost

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
