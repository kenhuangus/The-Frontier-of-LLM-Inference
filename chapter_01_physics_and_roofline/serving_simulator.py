"""
Discrete-Event Continuous Batching Serving Simulator
Simulates iteration-level scheduling, monolithic prefill head-of-line blocking,
and chunked prefill latency isolation under heavy load.
"""
import random
from typing import List, Dict, Optional

class ServingRequest:
    def __init__(self, req_id: str, prompt_len: int, gen_len: int, arrival_ms: float = 0.0):
        if prompt_len < 0:
            raise ValueError("prompt_len cannot be negative.")
        if gen_len < 0:
            raise ValueError("gen_len cannot be negative.")
        self.req_id = req_id
        self.prompt_len = prompt_len
        self.gen_len = gen_len
        self.arrival_ms = arrival_ms
        self.remaining_prefill = prompt_len
        self.remaining_decode = gen_len
        self.start_ms = 0.0
        self.complete_ms = 0.0
        self.itls_ms: List[float] = []

def simulate_serving_policy(
    chunk_size: int = 512,
    num_requests: int = 20,
    custom_requests: Optional[List[ServingRequest]] = None
) -> Dict[str, float]:
    """Simulates execution with a specified prefill chunk size."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if num_requests < 0:
        raise ValueError("num_requests cannot be negative.")

    if custom_requests is not None:
        requests = [
            ServingRequest(r.req_id, r.prompt_len, r.gen_len, r.arrival_ms)
            for r in custom_requests
        ]
    elif num_requests == 0:
        return {"Chunk_Size": chunk_size, "P50_ITL_ms": 0.0, "P99_ITL_ms": 0.0}
    else:
        random.seed(42)
        requests = [
            ServingRequest(
                f"req-{i}",
                prompt_len=random.randint(500, 2000),
                gen_len=random.randint(50, 150),
                arrival_ms=i * 10.0
            )
            for i in range(num_requests)
        ]
    
    active_decode: List[ServingRequest] = []
    prefill_queue: List[ServingRequest] = list(requests)
    current_time_ms = 0.0
    all_itls: List[float] = []
    
    while prefill_queue or active_decode:
        step_cost_ms = 15.0 # Base decode step cost
        
        # Service prefill chunk if present
        if prefill_queue:
            req = prefill_queue[0]
            chunk = min(req.remaining_prefill, chunk_size)
            req.remaining_prefill -= chunk
            # Prefill compute takes roughly 0.015ms per token
            step_cost_ms += chunk * 0.015
            if req.remaining_prefill <= 0:
                prefill_queue.pop(0)
                if req.remaining_decode > 0:
                    active_decode.append(req)
                
        # Advance all active decode requests by 1 token
        finished = []
        for d in active_decode:
            d.itls_ms.append(step_cost_ms)
            all_itls.append(step_cost_ms)
            d.remaining_decode -= 1
            if d.remaining_decode <= 0:
                finished.append(d)
        for f in finished:
            active_decode.remove(f)
            
        current_time_ms += step_cost_ms
        
    if not all_itls:
        return {"Chunk_Size": chunk_size, "P50_ITL_ms": 0.0, "P99_ITL_ms": 0.0}

    all_itls.sort()
    n = len(all_itls)
    p50 = all_itls[int(n * 0.50)]
    p99 = all_itls[min(int(n * 0.99), n - 1)]
    return {"Chunk_Size": chunk_size, "P50_ITL_ms": round(p50, 2), "P99_ITL_ms": round(p99, 2)}

if __name__ == "__main__":
    print("=" * 80)
    print("  CHAPTER 1: CONTINUOUS BATCHING SIMULATOR (Chunked vs Monolithic)")
    print("=" * 80)
    monolithic = simulate_serving_policy(chunk_size=4096)
    chunked = simulate_serving_policy(chunk_size=512)
    
    print(f"Monolithic Prefill (4k chunk) -> P50 ITL: {monolithic['P50_ITL_ms']} ms | P99 Tail: {monolithic['P99_ITL_ms']} ms")
    print(f"Chunked Prefill    (512 chunk)-> P50 ITL: {chunked['P50_ITL_ms']} ms | P99 Tail: {chunked['P99_ITL_ms']} ms")
    reduction = ((monolithic['P99_ITL_ms'] - chunked['P99_ITL_ms']) / monolithic['P99_ITL_ms']) * 100.0
    print(f"Chunked prefill eliminates decode tail latency spikes by {reduction:.1f}%!")
    print("=" * 80)
