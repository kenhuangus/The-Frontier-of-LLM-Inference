"""
Test-Time Reasoning Compute Budget Controller
Dynamically scales reasoning tokens based on query complexity, prompt characteristics, and verifier feedback.
"""
import math
import re
from typing import Dict, Optional, Union


class DynamicBudgetController:
    """
    Dynamic reasoning budget controller for frontier LLMs (e.g. DeepSeek-V4-Pro, GLM-5.3, Kimi K3).
    Supports discrete effort tiers, prompt complexity estimation, continuous scaling functions,
    and strict token clamping.
    """
    EFFORT_BUDGETS = {
        "deepseek-v4-pro": {"low": 1024, "high": 8192, "max": 32768},
        "glm-5.3": {"low": 2048, "high": 16384, "max": 65536},
        "kimi-k3": {"low": 2048, "high": 16384, "max": 65536},
    }

    DEFAULT_BUDGETS = {"low": 1024, "high": 4096, "max": 16384}

    # Complexity classification keywords
    HIGH_COMPLEXITY_PATTERNS = [
        r"\bproof\b", r"\bprove\b", r"\btheorem\b", r"\blemma\b",
        r"\bderive\b", r"\bderivation\b", r"\binduction\b", r"\bcontradiction\b",
        r"\bcuda\b", r"\bkernel\b", r"\broofline\b", r"\bwarp\b",
        r"\bmcts\b", r"\bformal\b", r"\boptimal\b", r"\brigor(?:ous)?\b",
        r"\bnp-hard\b", r"\bconcurrency\b", r"\bdeadlock\b",
    ]

    MODERATE_COMPLEXITY_PATTERNS = [
        r"\banalyze\b", r"\bcompare\b", r"\btrade-?offs?\b", r"\bimplement\b",
        r"\balgorithm\b", r"\bcalculate\b", r"\bexplain\b", r"\barchitecture\b",
        r"\bsynthesize\b", r"\bstep-by-step\b", r"\brefactor\b", r"\bevaluate\b",
    ]

    SIMPLE_PATTERNS = [
        r"\bhello\b", r"\bhi\b", r"\bwhat is\b", r"\bdefine\b",
        r"\bcapital of\b", r"\btranslate\b", r"\bwho is\b", r"\bhow are you\b",
        r"\bsummarize in one word\b", r"\byes or no\b",
    ]

    def __init__(
        self,
        model: str = "deepseek-v4-pro",
        min_tokens: int = 1024,
        max_tokens: int = 32768,
        default_scaling: str = "linear",
    ):
        if min_tokens > max_tokens:
            raise ValueError(f"min_tokens ({min_tokens}) cannot exceed max_tokens ({max_tokens})")
        if min_tokens < 0:
            raise ValueError("min_tokens cannot be negative")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        self.model = model
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.default_scaling = default_scaling

    @classmethod
    def estimate_complexity(cls, prompt: Optional[str]) -> float:
        """
        Estimates the reasoning complexity score [0.0, 1.0] of a given prompt.
        Empty or whitespace-only prompts safely evaluate to 0.0.
        """
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            return 0.0

        text = prompt.lower().strip()

        # Base score from prompt length
        score = 0.05
        if len(text) > 200:
            score += 0.25
        elif len(text) > 60:
            score += 0.12

        # Pattern scoring
        high_matches = sum(1 for p in cls.HIGH_COMPLEXITY_PATTERNS if re.search(p, text))
        mod_matches = sum(1 for p in cls.MODERATE_COMPLEXITY_PATTERNS if re.search(p, text))
        simple_matches = sum(1 for p in cls.SIMPLE_PATTERNS if re.search(p, text))

        score += min(0.60, high_matches * 0.25)
        score += min(0.30, mod_matches * 0.12)
        score -= simple_matches * 0.15

        # Code block and mathematical symbol heuristics
        if "```" in prompt:
            score += 0.20
        if any(c in prompt for c in ["∫", "∑", "√", "≤", "≥", "≠", "∈", "∀", "∃"]):
            score += 0.20

        return round(max(0.0, min(1.0, score)), 4)

    @classmethod
    def classify_prompt(cls, prompt: Optional[str], threshold: float = 0.3) -> str:
        """
        Classifies prompt difficulty into 'simple' or 'complex'.
        Empty or whitespace prompts return 'simple'.
        """
        score = cls.estimate_complexity(prompt)
        return "complex" if score >= threshold else "simple"

    @classmethod
    def allocate_thinking_budget(cls, model: str, complexity_score: float) -> int:
        """
        Allocates thinking tokens proportional to verified task difficulty using discrete effort tiers.
        Safely clamps out-of-range complexity scores to [0.0, 1.0] and handles incomplete model configurations.
        """
        score = max(0.0, min(1.0, float(complexity_score)))
        if score < 0.3:
            effort = "low"
        elif score < 0.7:
            effort = "high"
        else:
            effort = "max"

        model_budgets = cls.EFFORT_BUDGETS.get(model, cls.DEFAULT_BUDGETS)
        if effort in model_budgets:
            return model_budgets[effort]
        return model_budgets.get("max", cls.DEFAULT_BUDGETS[effort])

    @classmethod
    def compute_budget(
        cls,
        complexity_score: float,
        min_tokens: int = 1024,
        max_tokens: int = 32768,
        scaling: str = "linear",
    ) -> int:
        """
        Calculates a continuously scaled token budget clamped strictly within [min_tokens, max_tokens].
        Supported scaling modes: 'linear', 'exponential' (geometric), 'quadratic' (power), 'sqrt', 'step'.
        """
        if min_tokens > max_tokens:
            raise ValueError(f"min_tokens ({min_tokens}) cannot be greater than max_tokens ({max_tokens})")
        if min_tokens < 0:
            raise ValueError("min_tokens cannot be negative")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        # Clamp input score to [0.0, 1.0]
        score = max(0.0, min(1.0, float(complexity_score)))

        scaling_lower = scaling.lower()
        if scaling_lower == "linear":
            raw = min_tokens + score * (max_tokens - min_tokens)
        elif scaling_lower in ("exponential", "geometric"):
            if min_tokens == 0:
                raw = score * max_tokens
            else:
                raw = min_tokens * ((max_tokens / min_tokens) ** score)
        elif scaling_lower in ("quadratic", "power"):
            raw = min_tokens + (score ** 2) * (max_tokens - min_tokens)
        elif scaling_lower == "sqrt":
            raw = min_tokens + math.sqrt(score) * (max_tokens - min_tokens)
        elif scaling_lower == "step":
            if score < 0.3:
                raw = min_tokens
            elif score < 0.7:
                raw = min_tokens + (max_tokens - min_tokens) / 2.0
            else:
                raw = max_tokens
        else:
            raise ValueError(f"Unsupported scaling function '{scaling}'. Choose from: linear, exponential, quadratic, sqrt, step.")

        # Strict clamping to bounds
        clamped = max(min_tokens, min(max_tokens, int(round(raw))))
        return clamped

    def allocate_from_prompt(
        self,
        prompt: Optional[str],
        scaling: Optional[str] = None,
    ) -> int:
        """Instance method allocating reasoning budget directly from input prompt."""
        score = self.estimate_complexity(prompt)
        use_scaling = scaling or self.default_scaling
        return self.compute_budget(score, self.min_tokens, self.max_tokens, use_scaling)


# Backward compatibility alias
ReasoningBudgetController = DynamicBudgetController


if __name__ == "__main__":
    print("=" * 80)
    print("  CHAPTER 8: TEST-TIME REASONING BUDGET CONTROLLER")
    print("=" * 80)
    for score, label in [(0.15, "Simple Lookup"), (0.55, "Complex Code Synthesis"), (0.92, "Formal Proof")]:
        tokens = ReasoningBudgetController.allocate_thinking_budget("deepseek-v4-pro", score)
        print(f"Task: {label:<24} (Score: {score}) -> Allocated Thinking Budget: {tokens:,} tokens")
    print("=" * 80)

    # Prompt Complexity Classification Demo
    print("\n  PROMPT COMPLEXITY CLASSIFICATION & CONTINUOUS BUDGETING:")
    sample_prompts = [
        "What is the capital of France?",
        "Define Key-Value Cache and GQA.",
        "Implement a CUDA kernel for FlashAttention-3 with TMA warp-specialization.",
        "Prove that every planar graph is 4-colorable using proof by contradiction.",
    ]
    controller = DynamicBudgetController(model="deepseek-v4-pro", min_tokens=1024, max_tokens=32768)
    for p in sample_prompts:
        score = DynamicBudgetController.estimate_complexity(p)
        cls_label = DynamicBudgetController.classify_prompt(p)
        tokens_linear = controller.allocate_from_prompt(p, scaling="linear")
        tokens_exp = controller.allocate_from_prompt(p, scaling="exponential")
        print(f"Prompt : {p}")
        print(f"  Class: {cls_label.upper()} | Score: {score:.2f} | Linear: {tokens_linear:,} tokens | Exp: {tokens_exp:,} tokens\n")
    print("=" * 80)
