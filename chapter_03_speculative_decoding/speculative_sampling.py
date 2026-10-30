"""
Exact Lossless Speculative Rejection Sampler (Leviathan-Chen Invariance)
Mathematically guarantees that the accepted token distribution exactly matches
the target foundation model distribution without quality drift.
"""
import random
from typing import List, Tuple

class SpeculativeRejectionSampler:
    def __init__(self, temperature: float = 1.0):
        self.temperature = max(1e-5, temperature)

    def sample_from_probs(self, probs: List[float]) -> int:
        r = random.random()
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
        Rejection sampling verification:
        Accept token with probability min(1, P_target(x) / P_draft(x)).
        If rejected, resample from max(0, P_target(x) - P_draft(x)) normalized.
        """
        accepted = []
        for i, token in enumerate(draft_tokens):
            p_target = target_probs[i][token]
            p_draft = draft_probs[i][token]
            
            accept_prob = min(1.0, p_target / max(p_draft, 1e-9))
            if random.random() < accept_prob:
                accepted.append(token)
            else:
                # Resample from residual distribution
                vocab_size = len(target_probs[i])
                residual = [max(0.0, target_probs[i][k] - draft_probs[i][k]) for k in range(vocab_size)]
                s = sum(residual)
                if s > 0:
                    norm_residual = [r / s for r in residual]
                    recovery_token = self.sample_from_probs(norm_residual)
                else:
                    recovery_token = self.sample_from_probs(target_probs[i])
                accepted.append(recovery_token)
                break # Rejection ends the speculative chain
        return accepted, len(accepted)

if __name__ == "__main__":
    print("=" * 80)
    print("  CHAPTER 3: EXACT SPECULATIVE REJECTION SAMPLER (Leviathan-Chen)")
    print("=" * 80)
    sampler = SpeculativeRejectionSampler()
    
    # Draft 4 tokens with high overlap probability
    draft_tokens = [12, 45, 99, 102]
    # Simulated probabilities: high target agreement on first 3, disagreement on 4th
    draft_probs = [
        [0.1]*12 + [0.85] + [0.05]*87,
        [0.1]*45 + [0.80] + [0.10]*54,
        [0.05]*99 + [0.90] + [0.05]*0,
        [0.1]*102 + [0.75] + [0.15]*0,
    ]
    target_probs = [
        [0.1]*12 + [0.90] + [0.0]*87,
        [0.1]*45 + [0.85] + [0.05]*54,
        [0.05]*99 + [0.88] + [0.07]*0,
        [0.3]*102 + [0.10] + [0.60]*0, # Low target probability -> triggers resample
    ]
    
    accepted, count = sampler.verify_speculative_tokens(draft_tokens, draft_probs, target_probs)
    print(f"Draft Tokens Submitted: {draft_tokens}")
    print(f"Verified Token Sequence: {accepted} (Accepted length: {count})")
    print(f"Effective Step Speedup: {count:.2f}x tokens per verification step!")
    print("=" * 80)
