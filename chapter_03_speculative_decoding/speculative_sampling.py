"""
Exact Lossless Speculative Rejection Sampler (Leviathan-Chen Invariance)
Mathematically guarantees that the accepted token distribution exactly matches
the target foundation model distribution without quality drift.
"""
import math
import random
from typing import List, Optional, Tuple

class SpeculativeRejectionSampler:
    def __init__(self, temperature: float = 1.0, top_p: float = 1.0):
        self.temperature = max(1e-5, temperature)
        self.top_p = max(1e-5, min(1.0, top_p))

    @staticmethod
    def softmax(logits: List[float], temperature: float = 1.0) -> List[float]:
        """Converts raw logits to a normalized probability distribution with temperature scaling."""
        temp = max(1e-5, temperature)
        scaled = [x / temp for x in logits]
        max_val = max(scaled)
        exp_vals = [math.exp(x - max_val) for x in scaled]
        sum_exp = sum(exp_vals)
        return [e / sum_exp for e in exp_vals]

    def apply_temperature_and_topp(
        self,
        logits: List[float],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> List[float]:
        """Applies temperature scaling and top-p (nucleus) truncation to logits."""
        temp = self.temperature if temperature is None else max(1e-5, temperature)
        p_cut = self.top_p if top_p is None else max(1e-5, min(1.0, top_p))
        
        probs = self.softmax(logits, temp)
        if p_cut >= 1.0:
            return probs
            
        # Top-p nucleus filtering
        indexed_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        cumulative = 0.0
        selected_indices = set()
        for idx, prob in indexed_probs:
            selected_indices.add(idx)
            cumulative += prob
            if cumulative >= p_cut:
                break
                
        filtered_probs = [p if i in selected_indices else 0.0 for i, p in enumerate(probs)]
        total = sum(filtered_probs)
        return [p / total for p in filtered_probs]

    def sample_from_probs(self, probs: List[float]) -> int:
        """Draws a discrete categorical sample from a probability distribution."""
        if not probs:
            raise ValueError("Probability distribution cannot be empty")
        total = sum(probs)
        if total <= 0.0:
            return random.randint(0, len(probs) - 1)
        r = random.random() * total
        cumulative = 0.0
        for idx, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return idx
        return len(probs) - 1

    def verify_speculative_tokens(
        self,
        draft_tokens: List[int],
        draft_probs: List[List[float]],
        target_probs: List[List[float]]
    ) -> Tuple[List[int], int]:
        """
        Rejection sampling verification (Leviathan et al., 2023; Chen et al., 2023):
        Accept token with probability min(1, P_target(x) / P_draft(x)).
        If rejected, resample from max(0, P_target(x) - P_draft(x)) normalized.
        If all K draft tokens are accepted and K+1 target distribution is provided,
        emits a bonus token sampled directly from P_target(K+1).
        """
        # Edge case: empty draft chain
        if not draft_tokens:
            if target_probs and len(target_probs) > 0:
                bonus_token = self.sample_from_probs(target_probs[0])
                return [bonus_token], 1
            return [], 0

        # Input dimension validation
        if len(draft_probs) < len(draft_tokens):
            raise ValueError(
                f"draft_probs length ({len(draft_probs)}) must be >= draft_tokens length ({len(draft_tokens)})"
            )
        if len(target_probs) < len(draft_tokens):
            raise ValueError(
                f"target_probs length ({len(target_probs)}) must be >= draft_tokens length ({len(draft_tokens)})"
            )

        accepted: List[int] = []
        for i, token in enumerate(draft_tokens):
            v_draft = len(draft_probs[i])
            v_target = len(target_probs[i])
            if v_draft != v_target:
                raise ValueError(
                    f"Position {i} vocabulary mismatch: draft {v_draft} vs target {v_target}"
                )
            if token < 0 or token >= v_target:
                raise ValueError(
                    f"Draft token {token} at position {i} out of vocabulary range [0, {v_target - 1}]"
                )

            p_target = target_probs[i][token]
            p_draft = draft_probs[i][token]
            
            accept_prob = min(1.0, p_target / max(p_draft, 1e-9))
            if random.random() < accept_prob:
                accepted.append(token)
            else:
                # Resample from normalized residual distribution: max(0, P_target - P_draft)
                vocab_size = v_target
                residual = [max(0.0, target_probs[i][k] - draft_probs[i][k]) for k in range(vocab_size)]
                s = sum(residual)
                if s > 0.0:
                    norm_residual = [r / s for r in residual]
                    recovery_token = self.sample_from_probs(norm_residual)
                else:
                    recovery_token = self.sample_from_probs(target_probs[i])
                accepted.append(recovery_token)
                break  # Rejection ends the speculative chain

        # Leviathan-Chen bonus token: if all K draft tokens accepted and target distribution exists for K+1
        if len(accepted) == len(draft_tokens) and len(target_probs) > len(draft_tokens):
            bonus_token = self.sample_from_probs(target_probs[len(draft_tokens)])
            accepted.append(bonus_token)

        return accepted, len(accepted)

if __name__ == "__main__":
    print("=" * 80)
    print("  CHAPTER 3: EXACT SPECULATIVE REJECTION SAMPLER (Leviathan-Chen)")
    print("=" * 80)
    sampler = SpeculativeRejectionSampler()
    
    # Vocabulary size 105
    vocab_size = 105
    draft_tokens = [12, 45, 99, 102]
    
    # Helper to create normalized distribution with a peak on a given token
    def make_distribution(peak_token: int, peak_prob: float, size: int) -> List[float]:
        remaining = 1.0 - peak_prob
        background = remaining / (size - 1)
        dist = [background] * size
        dist[peak_token] = peak_prob
        return dist

    # High target agreement on first 3, sharp disagreement on 4th
    draft_probs = [
        make_distribution(12, 0.85, vocab_size),
        make_distribution(45, 0.80, vocab_size),
        make_distribution(99, 0.90, vocab_size),
        make_distribution(102, 0.75, vocab_size),
    ]
    target_probs = [
        make_distribution(12, 0.90, vocab_size),  # p >= q -> 100% accept
        make_distribution(45, 0.85, vocab_size),  # p >= q -> 100% accept
        make_distribution(99, 0.95, vocab_size),  # p >= q -> 100% accept
        make_distribution(25, 0.85, vocab_size),  # target prefers token 25; token 102 prob is low -> triggers resample
    ]
    
    accepted, count = sampler.verify_speculative_tokens(draft_tokens, draft_probs, target_probs)
    print(f"Draft Tokens Submitted: {draft_tokens}")
    print(f"Verified Token Sequence: {accepted} (Emitted length: {count})")
    print(f"Effective Step Speedup: {count:.2f}x tokens per verification step!")
    print("=" * 80)
