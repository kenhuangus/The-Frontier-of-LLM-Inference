"""
Test-Time Reasoning Compute Budget Controller
Dynamically scales reasoning tokens based on query complexity and verifier feedback.
"""
from typing import Dict

class ReasoningBudgetController:
    EFFORT_BUDGETS = {
        "deepseek-v4-pro": {"low": 1024, "high": 8192, "max": 32768},
        "glm-5.3": {"low": 2048, "high": 16384, "max": 65536},
        "kimi-k3": {"max": 65536}
    }
    
    @classmethod
    def allocate_thinking_budget(cls, model: str, complexity_score: float) -> int:
        """Allocates thinking tokens proportional to verified task difficulty."""
        if complexity_score < 0.3:
            effort = "low"
        elif complexity_score < 0.7:
            effort = "high"
        else:
            effort = "max"
        return cls.EFFORT_BUDGETS.get(model, {"low": 1024, "high": 4096, "max": 16384})[effort]

if __name__ == "__main__":
    print("=" * 80)
    print("  CHAPTER 8: TEST-TIME REASONING BUDGET CONTROLLER")
    print("=" * 80)
    for score, label in [(0.15, "Simple Lookup"), (0.55, "Complex Code Synthesis"), (0.92, "Formal Proof")]:
        tokens = ReasoningBudgetController.allocate_thinking_budget("deepseek-v4-pro", score)
        print(f"Task: {label:<24} (Score: {score}) -> Allocated Thinking Budget: {tokens:,} tokens")
    print("=" * 80)
