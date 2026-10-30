"""
Gateway Affinity Router with Sticky Session Pinning and Consistent Hashing
Directs multi-turn requests to warm KV cache replicas to maximize cache hit rates,
with support for consistent hash distribution, node rebalancing, and empty pool resilience.
"""
import bisect
import hashlib
from typing import List, Dict, Optional, Any


class ConsistentHashRing:
    """
    Consistent Hash Ring with virtual nodes for uniform key distribution
    and minimal rebalancing churn upon node addition or removal.
    """
    def __init__(self, nodes: Optional[List[str]] = None, vnodes: int = 100):
        self.vnodes = vnodes
        self.ring: List[int] = []
        self.ring_map: Dict[int, str] = {}
        self.nodes: set = set()
        if nodes:
            for node in nodes:
                self.add_node(node)

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)

    def add_node(self, node: str) -> None:
        if node in self.nodes:
            return
        self.nodes.add(node)
        for i in range(self.vnodes):
            h = self._hash(f"{node}:vnode_{i}")
            self.ring_map[h] = node
            bisect.insort(self.ring, h)

    def remove_node(self, node: str) -> None:
        if node not in self.nodes:
            return
        self.nodes.remove(node)
        for i in range(self.vnodes):
            h = self._hash(f"{node}:vnode_{i}")
            if h in self.ring_map:
                del self.ring_map[h]
                idx = bisect.bisect_left(self.ring, h)
                if idx < len(self.ring) and self.ring[idx] == h:
                    self.ring.pop(idx)

    def get_node(self, key: str) -> Optional[str]:
        if not self.ring:
            return None
        h = self._hash(key)
        idx = bisect.bisect_right(self.ring, h)
        if idx == len(self.ring):
            idx = 0
        return self.ring_map[self.ring[idx]]


class GatewayAffinityRouter:
    """
    Gateway Router balancing load and affinity.
    Supports sticky session pinning with TTL and consistent hash prefix routing.
    """
    def __init__(self, workers: List[Dict[str, Any]], strategy: str = "least_loaded", vnodes: int = 100):
        self.workers = list(workers)  # list of {"id": str, "load": int, ...}
        self.strategy = strategy
        self.sticky_map: Dict[str, Dict[str, Any]] = {}
        self.hash_ring = ConsistentHashRing([w["id"] for w in self.workers], vnodes=vnodes)

    def has_workers(self) -> bool:
        return len(self.workers) > 0

    def add_worker(self, worker: Dict[str, Any]) -> None:
        # Avoid duplicate IDs
        self.remove_worker(worker["id"])
        self.workers.append(worker)
        self.hash_ring.add_node(worker["id"])

    def remove_worker(self, worker_id: str) -> None:
        self.workers = [w for w in self.workers if w["id"] != worker_id]
        self.hash_ring.remove_node(worker_id)
        # Purge dead worker from sticky cache
        self.sticky_map = {k: v for k, v in self.sticky_map.items() if v["wid"] != worker_id}

    def route(
        self,
        session_id: str,
        prefix: Optional[str] = None,
        pin: bool = True,
        now: float = 0.0,
        ttl_s: float = 300.0,
        strategy: Optional[str] = None,
    ) -> str:
        """
        Routes a request to a worker replica.
        
        Raises RuntimeError if the worker pool is empty.
        """
        if not self.workers:
            raise RuntimeError("Worker pool is empty; cannot route request")

        rec = self.sticky_map.get(session_id)
        if rec and (now - rec["t"] < ttl_s) and pin:
            # Confirm pinned worker is still active
            if any(w["id"] == rec["wid"] for w in self.workers):
                return rec["wid"]
            else:
                self.sticky_map.pop(session_id, None)

        active_strategy = strategy or self.strategy
        if active_strategy == "consistent_hash":
            routing_key = prefix if prefix is not None else session_id
            wid = self.hash_ring.get_node(routing_key)
            if wid is None:
                raise RuntimeError("Consistent hash ring produced no worker")
        else:
            # Default least-loaded routing
            best_worker = min(self.workers, key=lambda w: w.get("load", 0))
            wid = best_worker["id"]

        if pin:
            self.sticky_map[session_id] = {"wid": wid, "t": now}
        return wid

    def route_prefix(
        self,
        prefix: str,
        session_id: Optional[str] = None,
        pin: bool = True,
        now: float = 0.0,
        ttl_s: float = 300.0,
    ) -> str:
        """
        Convenience method to route requests by prompt prefix using consistent hashing.
        """
        sess = session_id if session_id is not None else prefix
        return self.route(sess, prefix=prefix, pin=pin, now=now, ttl_s=ttl_s, strategy="consistent_hash")


if __name__ == "__main__":
    workers = [{"id": "worker-gpu-1", "load": 40}, {"id": "worker-gpu-2", "load": 15}]
    router = GatewayAffinityRouter(workers)
    
    # Request turn 1: lands on least loaded
    w1 = router.route("sess-99", pin=True, now=10.0)
    # Request turn 2: pinned to same worker to reuse warm prefix
    w2 = router.route("sess-99", pin=True, now=25.0)
    print(f"Turn 1 Target: {w1} | Turn 2 Pinned Target: {w2} (Cache Hit Guaranteed!)")

    # Consistent hashing demonstration
    hash_router = GatewayAffinityRouter(workers, strategy="consistent_hash")
    p1 = hash_router.route_prefix("sys_prompt_agent_v1")
    p2 = hash_router.route_prefix("sys_prompt_agent_v1")
    print(f"Consistent Hash Prefix Route 1: {p1} | Route 2: {p2} (Deterministic Affinity!)")
