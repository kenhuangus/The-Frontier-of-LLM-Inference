"""
Gateway Affinity Router with Sticky Session Pinning
Directs multi-turn requests to warm KV cache replicas to maximize cache hit rates.
"""
from typing import List, Dict, Optional

class GatewayAffinityRouter:
    def __init__(self, workers: List[Dict]):
        self.workers = workers # list of {"id": str, "load": int}
        self.sticky_map: Dict[str, Dict] = {}

    def route(self, session_id: str, pin: bool = True, now: float = 0.0, ttl_s: float = 300.0) -> str:
        rec = self.sticky_map.get(session_id)
        if rec and (now - rec["t"] < ttl_s) and pin:
            return rec["wid"]
            
        # Select worker with lowest projected load
        best_worker = min(self.workers, key=lambda w: w["load"])
        wid = best_worker["id"]
        if pin:
            self.sticky_map[session_id] = {"wid": wid, "t": now}
        return wid

if __name__ == "__main__":
    workers = [{"id": "worker-gpu-1", "load": 40}, {"id": "worker-gpu-2", "load": 15}]
    router = GatewayAffinityRouter(workers)
    
    # Request turn 1: lands on least loaded
    w1 = router.route("sess-99", pin=True, now=10.0)
    # Request turn 2: pinned to same worker to reuse warm prefix
    w2 = router.route("sess-99", pin=True, now=25.0)
    print(f"Turn 1 Target: {w1} | Turn 2 Pinned Target: {w2} (Cache Hit Guaranteed!)")
