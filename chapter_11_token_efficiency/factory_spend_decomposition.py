"""
Software-Factory Cost Equation Decomposition
Models the six multiplying terms: Spend = Users * Sessions * Turns * Reqs * Tokens * Price
"""
def factory_spend(users: int, sessions: int, turns: int, reqs: int, tokens: int, price: float) -> float:
    return users * sessions * turns * reqs * tokens * price

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
    print("=" * 80)
