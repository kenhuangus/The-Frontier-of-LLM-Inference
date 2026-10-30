"""
High-Precision Inference Telemetry Tracker
Captures Time-To-First-Token (TTFT), Inter-Token Latency (ITL), P50/P90/P95/P99 tail percentiles,
and jitter synthesis for production SLO monitoring.
"""
import time
import statistics
from typing import List, Dict, Optional, Any

class InferenceTelemetryTracker:
    def __init__(self, request_id: str, t_submit: Optional[float] = None):
        self.request_id = request_id
        self.t_submit: float = time.perf_counter() if t_submit is None else t_submit
        self.t_first_token: Optional[float] = None
        self.token_timestamps: List[float] = []

    def record_token(self, timestamp: Optional[float] = None):
        """Records a token timestamp. If timestamp is None, uses time.perf_counter()."""
        t_now = time.perf_counter() if timestamp is None else timestamp
        if self.t_first_token is None:
            self.t_first_token = t_now
        self.token_timestamps.append(t_now)

    def compute_summary(self) -> Dict[str, Any]:
        if not self.token_timestamps or self.t_first_token is None:
            raise ValueError("No tokens recorded.")
        ttft_ms = (self.t_first_token - self.t_submit) * 1000.0
        total_gen_tokens = len(self.token_timestamps)
        
        itls_ms = [
            (self.token_timestamps[i] - self.token_timestamps[i-1]) * 1000.0
            for i in range(1, total_gen_tokens)
        ]
        sorted_itls = sorted(itls_ms) if itls_ms else [0.0]
        n = len(sorted_itls)
        p50 = sorted_itls[int(n * 0.50)]
        p90 = sorted_itls[min(int(n * 0.90), n - 1)]
        p95 = sorted_itls[min(int(n * 0.95), n - 1)]
        p99 = sorted_itls[min(int(n * 0.99), n - 1)]
        return {
            "TTFT_ms": round(ttft_ms, 2),
            "Total_Tokens": total_gen_tokens,
            "P50_ITL_ms": round(p50, 2),
            "P90_ITL_ms": round(p90, 2),
            "P95_ITL_ms": round(p95, 2),
            "P99_ITL_ms": round(p99, 2)
        }

    def compute_jitter_and_throughput(self) -> Dict[str, Any]:
        summary = self.compute_summary()
        total_gen_tokens = len(self.token_timestamps)
        t_complete = self.token_timestamps[-1]
        
        itls_ms = [
            (self.token_timestamps[i] - self.token_timestamps[i-1]) * 1000.0
            for i in range(1, total_gen_tokens)
        ]
        itl_jitter_std = statistics.stdev(itls_ms) if len(itls_ms) > 1 else 0.0
        duration = max(t_complete - self.t_first_token, 1e-6)
        gen_tps = (total_gen_tokens - 1) / duration
        
        summary["ITL_Jitter_Std_ms"] = round(itl_jitter_std, 2)
        summary["Generation_Tokens_Per_Sec"] = round(gen_tps, 2)
        return summary

if __name__ == "__main__":
    import random
    print("Simulating synthetic token stream...")
    tracker = InferenceTelemetryTracker("req-test-001")
    time.sleep(0.04) # Simulate 40ms prefill
    
    # Generate 50 tokens with slight jitter
    for _ in range(50):
        tracker.record_token()
        time.sleep(random.uniform(0.015, 0.025)) # 15-25ms decode step
        
    metrics = tracker.compute_jitter_and_throughput()
    print("Telemetry metrics successfully captured:")
    for k, v in metrics.items():
        print(f"  {k:26s}: {v}")
