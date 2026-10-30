"""
Min-Waste Optimizer: Retain vs Offload vs Discard during Tool-Wait
Evaluates GPU memory holding cost against re-prefill computational cost.
"""
def kv_during_wait(wait_ms: float, prefix_tok: int, hold_per_ms: float = 1e-6, prefill_per_tok: float = 1e-4) -> str:
    hold = wait_ms * hold_per_ms * prefix_tok
    recompute = prefix_tok * prefill_per_tok
    
    if wait_ms <= 400:
        return "retain"
    if hold < recompute and wait_ms <= 8000:
        return "retain"
    if wait_ms <= 20000:
        return "offload"
    return "discard"

if __name__ == "__main__":
    print("=" * 80)
    print("  CHAPTER 12: TOOL-WAIT KV CACHE OPTIMIZATION DECISION ENGINE")
    print("=" * 80)
    scenarios = [
        ("Fast Web API", 250, 52000),
        ("Database Join", 3500, 52000),
        ("Human-in-the-Loop Approval", 45000, 52000),
    ]
    for label, wait_ms, prefix in scenarios:
        decision = kv_during_wait(wait_ms, prefix)
        print(f"Scenario: {label:<28} ({wait_ms} ms wait, {prefix:,} tok) -> Action: {decision.upper()}")
    print("=" * 80)
