"""
Session-Native KV Cache Retention with Subagent Isolation
Pins subagent sessions to local replicas while allowing lead orchestrators to load-balance.
"""
def pin_replica(session_id: str, workers: list, sticky: dict, ttl_s: float, now: float, pin: bool) -> str:
    rec = sticky.get(session_id)
    if rec and now - rec["t"] < ttl_s and pin:
        return rec["wid"]
    wid = min(workers, key=lambda w: w["load"])["id"]
    if pin:
        sticky[session_id] = {"wid": wid, "t": now}
    return wid

if __name__ == "__main__":
    workers = [{"id": "w1", "load": 80}, {"id": "w2", "load": 20}]
    sticky_sessions = {}
    
    # Subagent request with pin=True
    wid_sub = pin_replica("agent-sub-1", workers, sticky_sessions, 300.0, 10.0, pin=True)
    print(f"Subagent assigned to: {wid_sub}")
    
    # Subsequent turn stays on same worker
    wid_sub2 = pin_replica("agent-sub-1", workers, sticky_sessions, 300.0, 20.0, pin=True)
    print(f"Subagent follow-up pinned to: {wid_sub2} (Isolation preserved)")
