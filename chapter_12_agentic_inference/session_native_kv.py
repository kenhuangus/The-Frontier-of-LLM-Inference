"""
Session-Native KV Cache Retention with Subagent Isolation
Pins subagent sessions to local replicas while allowing lead orchestrators to load-balance.
Provides multi-turn agentic session KV management, block allocation, and LRU/TTL eviction.
"""
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def pin_replica(session_id: str, workers: list, sticky: dict, ttl_s: float, now: float, pin: bool) -> str:
    """
    Pins subagent sessions to local replicas while allowing lead orchestrators to load-balance.
    """
    if not workers:
        raise ValueError("workers list cannot be empty")
    if not session_id:
        raise ValueError("session_id cannot be empty")
    if ttl_s < 0:
        raise ValueError("ttl_s must be non-negative")

    rec = sticky.get(session_id)
    if rec and now - rec["t"] < ttl_s and pin:
        return rec["wid"]
    wid = min(workers, key=lambda w: w["load"])["id"]
    if pin:
        sticky[session_id] = {"wid": wid, "t": now}
    return wid


@dataclass
class SessionRecord:
    session_id: str
    worker_id: Optional[str] = None
    is_pinned: bool = True
    created_at: float = 0.0
    last_active: float = 0.0
    block_ids: List[int] = field(default_factory=list)
    total_tokens: int = 0
    turns: List[Dict[str, Any]] = field(default_factory=list)


class SessionNativeKV:
    """
    Session-Native KV Cache Retention Engine with Subagent Isolation.
    Manages paged KV cache blocks across multi-turn agent interactions,
    worker replica affinity pinning, and LRU/TTL eviction under memory pressure.
    """
    def __init__(
        self,
        total_blocks: int = 1024,
        block_size: int = 16,
        ttl_s: float = 300.0,
        max_sessions: Optional[int] = None
    ):
        if total_blocks <= 0:
            raise ValueError("total_blocks must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        if max_sessions is not None and max_sessions <= 0:
            raise ValueError("max_sessions must be positive")

        self.total_blocks = total_blocks
        self.block_size = block_size
        self.ttl_s = ttl_s
        self.max_sessions = max_sessions

        self.free_blocks: List[int] = list(range(total_blocks))
        self.sessions: Dict[str, SessionRecord] = {}
        self.sticky_table: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.stats = {
            "sessions_created": 0,
            "sessions_evicted_ttl": 0,
            "sessions_evicted_lru": 0,
            "blocks_allocated": 0,
            "blocks_freed": 0,
        }

    def create_session(
        self,
        session_id: str,
        pin: bool = True,
        now: float = 0.0,
        worker_id: Optional[str] = None
    ) -> SessionRecord:
        if not session_id or not isinstance(session_id, str):
            raise ValueError("session_id must be a non-empty string")

        with self.lock:
            # Check max_sessions capacity
            if self.max_sessions and len(self.sessions) >= self.max_sessions and session_id not in self.sessions:
                self.evict_lru(1)

            if session_id in self.sessions:
                rec = self.sessions[session_id]
                rec.is_pinned = pin
                rec.last_active = now
                if worker_id is not None:
                    rec.worker_id = worker_id
                return rec

            rec = SessionRecord(
                session_id=session_id,
                worker_id=worker_id,
                is_pinned=pin,
                created_at=now,
                last_active=now,
            )
            self.sessions[session_id] = rec
            self.stats["sessions_created"] += 1
            return rec

    def append_turn(
        self,
        session_id: str,
        tokens: List[int],
        tool_context: Optional[Dict[str, Any]] = None,
        now: float = 0.0
    ) -> List[int]:
        """
        Appends a conversational or tool execution turn to an active session,
        allocating new paged KV cache blocks as needed.
        """
        with self.lock:
            if not session_id or session_id not in self.sessions:
                raise KeyError(f"Session '{session_id}' not found")

            rec = self.sessions[session_id]
            rec.last_active = now

            if not tokens:
                rec.turns.append({
                    "turn_idx": len(rec.turns),
                    "num_tokens": 0,
                    "tool_context": tool_context or {},
                    "allocated_blocks": [],
                })
                return []

            prev_tokens = rec.total_tokens
            new_tokens = prev_tokens + len(tokens)
            prev_blocks_needed = math.ceil(prev_tokens / self.block_size) if prev_tokens > 0 else 0
            new_blocks_needed = math.ceil(new_tokens / self.block_size)
            blocks_to_allocate = new_blocks_needed - prev_blocks_needed

            if len(self.free_blocks) < blocks_to_allocate:
                # Attempt to free expired sessions first
                self.evict_expired(now)

            if len(self.free_blocks) < blocks_to_allocate:
                raise MemoryError(
                    f"Insufficient KV cache blocks: needed {blocks_to_allocate}, available {len(self.free_blocks)}"
                )

            allocated: List[int] = []
            for _ in range(blocks_to_allocate):
                blk = self.free_blocks.pop(0)
                allocated.append(blk)
                rec.block_ids.append(blk)

            rec.total_tokens = new_tokens
            rec.turns.append({
                "turn_idx": len(rec.turns),
                "num_tokens": len(tokens),
                "tool_context": tool_context or {},
                "allocated_blocks": allocated,
            })

            self.stats["blocks_allocated"] += len(allocated)
            return allocated

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """
        Retrieves a session record. Safely handles empty string and non-existent IDs.
        """
        with self.lock:
            if not session_id or session_id not in self.sessions:
                return None
            return self.sessions[session_id]

    def get_session_blocks(self, session_id: str) -> List[int]:
        """
        Retrieves allocated KV block IDs for a session.
        Safely returns empty list for empty/missing sessions.
        """
        with self.lock:
            if not session_id or session_id not in self.sessions:
                return []
            return list(self.sessions[session_id].block_ids)

    def pin_session(
        self,
        session_id: str,
        workers: list,
        pin: bool = True,
        now: float = 0.0
    ) -> str:
        """
        Pins session to a worker replica according to load and affinity rules.
        """
        with self.lock:
            wid = pin_replica(session_id, workers, self.sticky_table, self.ttl_s, now, pin)
            if session_id in self.sessions:
                rec = self.sessions[session_id]
                rec.worker_id = wid
                rec.is_pinned = pin
                rec.last_active = now
            return wid

    def evict_session(self, session_id: str) -> bool:
        """
        Evicts a session and returns its allocated KV blocks to the free pool.
        """
        with self.lock:
            if session_id not in self.sessions:
                return False

            rec = self.sessions.pop(session_id)
            if session_id in self.sticky_table:
                del self.sticky_table[session_id]

            num_freed = len(rec.block_ids)
            self.free_blocks.extend(rec.block_ids)
            self.stats["blocks_freed"] += num_freed
            return True

    def evict_expired(self, now: float) -> List[str]:
        """
        Evicts all sessions that have exceeded the time-to-live threshold.
        """
        with self.lock:
            expired_ids = [
                sid for sid, rec in self.sessions.items()
                if (now - rec.last_active) >= self.ttl_s
            ]
            for sid in expired_ids:
                self.evict_session(sid)
                self.stats["sessions_evicted_ttl"] += 1
            return expired_ids

    def evict_lru(self, count: int = 1) -> List[str]:
        """
        Evicts the N least-recently-active sessions.
        """
        with self.lock:
            if not self.sessions or count <= 0:
                return []

            sorted_sessions = sorted(self.sessions.values(), key=lambda s: s.last_active)
            evicted: List[str] = []
            for s in sorted_sessions[:count]:
                self.evict_session(s.session_id)
                self.stats["sessions_evicted_lru"] += 1
                evicted.append(s.session_id)
            return evicted

    def available_blocks(self) -> int:
        with self.lock:
            return len(self.free_blocks)

    def total_allocated_blocks(self) -> int:
        with self.lock:
            return self.total_blocks - len(self.free_blocks)


SessionNativeKVManager = SessionNativeKV


if __name__ == "__main__":
    workers = [{"id": "w1", "load": 80}, {"id": "w2", "load": 20}]
    sticky_sessions = {}

    # Subagent request with pin=True
    wid_sub = pin_replica("agent-sub-1", workers, sticky_sessions, 300.0, 10.0, pin=True)
    print(f"Subagent assigned to: {wid_sub}")

    # Subsequent turn stays on same worker
    wid_sub2 = pin_replica("agent-sub-1", workers, sticky_sessions, 300.0, 20.0, pin=True)
    print(f"Subagent follow-up pinned to: {wid_sub2} (Isolation preserved)")

    # Lead agent with pin=False (load balances)
    wid_lead = pin_replica("lead-orchestrator", workers, sticky_sessions, 300.0, 25.0, pin=False)
    print(f"Lead orchestrator load-balanced to: {wid_lead}")

    # SessionNativeKV Manager Demo
    kv_manager = SessionNativeKV(total_blocks=128, block_size=16, ttl_s=300.0)
    kv_manager.create_session("agent-sub-1", pin=True, now=10.0)
    blocks_turn1 = kv_manager.append_turn("agent-sub-1", tokens=list(range(32)), now=10.0)
    print(f"Turn 1 allocated blocks: {blocks_turn1}")
    blocks_turn2 = kv_manager.append_turn("agent-sub-1", tokens=list(range(16)), now=20.0)
    print(f"Turn 2 allocated blocks: {blocks_turn2}")
    print(f"Total session blocks: {kv_manager.get_session_blocks('agent-sub-1')}")
