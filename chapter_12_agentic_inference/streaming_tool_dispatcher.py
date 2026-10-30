"""
Streaming Tool Call Dispatcher
Demuxes streaming JSON tokens and immediately dispatches asynchronous execution
upon tool closure to eliminate pipeline idle bubbles.
"""
from typing import Dict, List, Optional

def tool_call_dispatch(chunk: Dict, buf: Dict, parsed: List) -> Optional[Dict]:
    if chunk.get("type") == "tool_use_close":
        parsed.append({"event": "tool_call_dispatch", "tool_call": dict(buf)})
        buf.clear()
        return parsed[-1]
    if "delta" in chunk:
        buf["args"] = buf.get("args", "") + chunk["delta"]
    if "name" in chunk:
        buf["name"] = chunk["name"]
    return None

if __name__ == "__main__":
    buf = {}
    parsed_events = []
    stream = [
        {"type": "tool_use_open", "name": "sql_query"},
        {"delta": "SELECT * FROM "},
        {"delta": "inference_logs"},
        {"type": "tool_use_close"}
    ]
    for chunk in stream:
        event = tool_call_dispatch(chunk, buf, parsed_events)
        if event:
            print("Streaming Tool Dispatch Triggered:", event)
