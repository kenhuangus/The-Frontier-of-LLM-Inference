"""
Streaming Tool Call Dispatcher
Demuxes streaming JSON tokens and immediately dispatches asynchronous execution
upon tool closure to eliminate pipeline idle bubbles.
Supports parallel execution, fast vs slow tool interleaving, and resilient error handling.
"""
import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional


def tool_call_dispatch(chunk: Dict, buf: Dict, parsed: List) -> Optional[Dict]:
    """
    Incremental SSE chunk demuxer. Emits typed tool_call_dispatch event upon structural completion.
    """
    if not isinstance(chunk, dict):
        return None

    if chunk.get("type") == "tool_use_close":
        event = {"event": "tool_call_dispatch", "tool_call": dict(buf)}
        parsed.append(event)
        buf.clear()
        return event

    if "name" in chunk:
        buf["name"] = chunk["name"]
    if "id" in chunk:
        buf["id"] = chunk["id"]
    if "tool_id" in chunk:
        buf["tool_id"] = chunk["tool_id"]
    if "delta" in chunk:
        buf["args"] = buf.get("args", "") + str(chunk["delta"])

    return None


@dataclass
class DispatchedToolTask:
    task_id: int
    tool_name: str
    raw_args: str
    parsed_args: Any
    status: str  # "pending", "running", "completed", "error"
    future: Optional[Future] = None
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0


class StreamingToolDispatcher:
    """
    Asynchronous Streaming Tool Call Dispatcher.
    Executes tools immediately upon JSON structural completion in parallel background threads,
    preventing inference pipeline idle bubbles.
    """
    def __init__(self, max_workers: int = 4):
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tool_registry: Dict[str, Callable] = {}
        self.buf: Dict[str, Any] = {}
        self.parsed_events: List[Dict[str, Any]] = []
        self.tasks: List[DispatchedToolTask] = []
        self.lock = threading.RLock()
        self._task_counter = 0

    def register_tool(self, name: str, fn: Callable) -> None:
        """
        Registers an executable tool function.
        """
        if not name or not callable(fn):
            raise ValueError("Tool name must be non-empty and fn must be callable")
        with self.lock:
            self.tool_registry[name] = fn

    def feed_chunk(self, chunk: Dict) -> Optional[Dict]:
        """
        Processes a single streaming chunk. If a tool_use_close chunk is encountered,
        immediately triggers asynchronous dispatch.
        """
        event = None
        with self.lock:
            event = tool_call_dispatch(chunk, self.buf, self.parsed_events)
        if event:
            self.dispatch_async(event["tool_call"])
        return event

    def feed_stream(self, chunks: Iterable[Dict]) -> List[Dict]:
        """
        Processes an iterable stream of chunks, returning all emitted dispatch events.
        """
        emitted: List[Dict] = []
        for chunk in chunks:
            ev = self.feed_chunk(chunk)
            if ev:
                emitted.append(ev)
        return emitted

    def dispatch_async(self, tool_call: Dict) -> DispatchedToolTask:
        """
        Dispatches a closed tool call for immediate background execution.
        """
        with self.lock:
            self._task_counter += 1
            task_id = self._task_counter

        tool_name = tool_call.get("name", "")
        raw_args = tool_call.get("args", "")

        parsed_args: Any = raw_args
        if isinstance(raw_args, str) and raw_args.strip():
            try:
                parsed_args = json.loads(raw_args)
            except Exception:
                parsed_args = raw_args

        task = DispatchedToolTask(
            task_id=task_id,
            tool_name=tool_name,
            raw_args=raw_args,
            parsed_args=parsed_args,
            status="pending",
        )

        with self.lock:
            self.tasks.append(task)

        def _worker():
            task.status = "running"
            task.start_time = time.time()
            try:
                with self.lock:
                    if task.tool_name not in self.tool_registry:
                        raise KeyError(f"Tool '{task.tool_name}' not registered")
                    fn = self.tool_registry[task.tool_name]

                args = task.parsed_args
                if isinstance(args, dict):
                    try:
                        res = fn(**args)
                    except TypeError:
                        res = fn(args)
                elif isinstance(args, list):
                    try:
                        res = fn(*args)
                    except TypeError:
                        res = fn(args)
                elif args is not None and args != "":
                    res = fn(args)
                else:
                    try:
                        res = fn()
                    except TypeError:
                        res = fn("")

                task.result = res
                task.status = "completed"
            except Exception as e:
                task.error = str(e)
                task.error_type = type(e).__name__
                task.status = "error"
            finally:
                task.end_time = time.time()
                task.duration_ms = (task.end_time - task.start_time) * 1000.0

        future = self.executor.submit(_worker)
        task.future = future
        return task

    def get_completed_tasks(self) -> List[DispatchedToolTask]:
        """
        Returns all tasks that have completed execution (non-blocking).
        """
        with self.lock:
            return [t for t in self.tasks if t.future is not None and t.future.done()]

    def wait_all(self, timeout: Optional[float] = None) -> List[DispatchedToolTask]:
        """
        Blocks until all submitted tasks have completed or timeout expires.
        """
        with self.lock:
            futures = [t.future for t in self.tasks if t.future is not None]
        if futures:
            for f in as_completed(futures, timeout=timeout):
                pass
        return self.tasks

    def shutdown(self, wait: bool = True) -> None:
        """
        Shuts down the thread pool executor.
        """
        self.executor.shutdown(wait=wait)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)


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

    # Demo StreamingToolDispatcher with concurrent execution
    with StreamingToolDispatcher(max_workers=2) as dispatcher:
        dispatcher.register_tool("sql_query", lambda q: f"Result of [{q}]: 100 records")
        dispatcher.feed_stream(stream)
        dispatcher.wait_all(timeout=2.0)
        for task in dispatcher.tasks:
            print(f"Async Tool Task {task.task_id} ({task.tool_name}): status={task.status}, result={task.result}")
