"""
Comprehensive Unit Test Suite for Chapter 12: Agentic Inference Systems
The Frontier of LLM Inference

Covers:
  - session_native_kv.py:
      * pin_replica subagent sticky affinity vs lead orchestrator load balancing
      * TTL expiration and replica rebalancing
      * Edge cases: empty workers list, empty session_id, negative TTL
      * Multi-turn session creation and lifecycle tracking
      * Paged KV cache block allocation and block math
      * Session eviction on timeout (TTL) and LRU memory reclamation
      * Concurrent sessions thread safety and replica isolation
      * Empty and non-existent session queries
      * KV cache block exhaustion and MemoryError handling
  - streaming_tool_dispatcher.py:
      * SSE streaming chunk demuxing (tool_use_open, delta, tool_use_close)
      * Multiple sequential tool call dispatches in a single turn
      * Malformed, empty, and non-dict chunk handling
      * Parallel asynchronous background tool execution
      * Fast vs slow tool execution (out-of-order completion, bubble elimination)
      * Tool error handling (runtime exception vs unregistered tool)
      * Automatic JSON arguments decoding and parameter unpacking
      * Context manager resource cleanup and thread pool shutdown
  - toolwait_offload_decision.py:
      * Canonical scenarios from Chapter 12: Fast Web API, Database Join, Human-in-the-Loop
      * Edge cases: zero tokens, zero wait time, negative values validation
      * Round-trip transfer latency scaling across PCIe Gen4, PCIe Gen5, and CXL 3.0
      * Dynamic breakeven wait time calculations under different interconnect bandwidths
      * VRAM pressure adaptation and threshold elasticity
      * Offload vs discard breakeven calculation
      * Dynamic decision matrix validation across parameter regimes
"""
import concurrent.futures
import json
import math
import os
import sys
import threading
import time
import unittest
from typing import Any, Dict, List

# Ensure current chapter directory is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from session_native_kv import (
    SessionNativeKV,
    SessionNativeKVManager,
    SessionRecord,
    pin_replica,
)
from streaming_tool_dispatcher import (
    DispatchedToolTask,
    StreamingToolDispatcher,
    tool_call_dispatch,
)
from toolwait_offload_decision import (
    ToolwaitOffloadDecision,
    ToolwaitOptimizer,
    kv_during_wait,
)


class TestSessionNativeKV(unittest.TestCase):
    """Test suite for Session-Native KV Cache retention, replica pinning, and memory management."""

    def setUp(self):
        self.workers = [
            {"id": "worker-gpu-0", "load": 80},
            {"id": "worker-gpu-1", "load": 20},
            {"id": "worker-gpu-2", "load": 50},
        ]
        self.sticky = {}
        self.ttl = 300.0

    def test_pin_replica_subagent_affinity_retention(self):
        """Verify subagents with pin=True remain sticky to the selected replica across turns."""
        wid_turn1 = pin_replica("subagent-alpha", self.workers, self.sticky, self.ttl, now=10.0, pin=True)
        self.assertEqual(wid_turn1, "worker-gpu-1")  # lowest load worker

        # Simulate load changes on workers
        self.workers[1]["load"] = 99  # worker-gpu-1 is now heavily loaded
        self.workers[0]["load"] = 10  # worker-gpu-0 is now least loaded

        # Subsequent turn within TTL must remain on worker-gpu-1 due to pin=True
        wid_turn2 = pin_replica("subagent-alpha", self.workers, self.sticky, self.ttl, now=25.0, pin=True)
        self.assertEqual(wid_turn2, "worker-gpu-1")

    def test_pin_replica_lead_orchestrator_load_balancing(self):
        """Verify lead orchestrators with pin=False dynamically load balance to the least-loaded worker."""
        # Initial request lands on least-loaded worker-gpu-1 (load=20)
        wid_1 = pin_replica("lead-orchestrator", self.workers, self.sticky, self.ttl, now=10.0, pin=False)
        self.assertEqual(wid_1, "worker-gpu-1")
        self.assertNotIn("lead-orchestrator", self.sticky)

        # Shift loads
        self.workers[1]["load"] = 95
        self.workers[0]["load"] = 5

        # Next turn must migrate to new least-loaded worker-gpu-0 (load=5)
        wid_2 = pin_replica("lead-orchestrator", self.workers, self.sticky, self.ttl, now=20.0, pin=False)
        self.assertEqual(wid_2, "worker-gpu-0")

    def test_pin_replica_ttl_expiration_rebalancing(self):
        """Verify that when TTL expires, sticky affinity expires and rebalances to least-loaded worker."""
        wid_init = pin_replica("session-expiring", self.workers, self.sticky, ttl_s=60.0, now=10.0, pin=True)
        self.assertEqual(wid_init, "worker-gpu-1")

        # Worker-gpu-1 becomes saturated, worker-gpu-2 is now lowest load
        self.workers[1]["load"] = 95
        self.workers[2]["load"] = 15

        # Call at now=75.0 (elapsed 65s > 60s TTL)
        wid_expired = pin_replica("session-expiring", self.workers, self.sticky, ttl_s=60.0, now=75.0, pin=True)
        self.assertEqual(wid_expired, "worker-gpu-2")
        self.assertEqual(self.sticky["session-expiring"]["wid"], "worker-gpu-2")

    def test_pin_replica_edge_cases_validation(self):
        """Verify input validation for pin_replica on empty or invalid arguments."""
        with self.assertRaises(ValueError):
            pin_replica("session-1", [], self.sticky, 300.0, 10.0, pin=True)
        with self.assertRaises(ValueError):
            pin_replica("", self.workers, self.sticky, 300.0, 10.0, pin=True)
        with self.assertRaises(ValueError):
            pin_replica("session-1", self.workers, self.sticky, -10.0, 10.0, pin=True)

    def test_session_manager_alias_compatibility(self):
        """Verify SessionNativeKVManager alias points to SessionNativeKV."""
        self.assertIs(SessionNativeKVManager, SessionNativeKV)

    def test_multiturn_session_creation_and_lifecycle(self):
        """Verify multi-turn session creation, turn appending, and token accounting."""
        kv = SessionNativeKV(total_blocks=100, block_size=16, ttl_s=300.0)

        # Create session
        rec = kv.create_session("agent-101", pin=True, now=5.0)
        self.assertEqual(rec.session_id, "agent-101")
        self.assertTrue(rec.is_pinned)
        self.assertEqual(rec.total_tokens, 0)
        self.assertEqual(len(rec.block_ids), 0)

        # Turn 1: 30 tokens -> requires ceil(30/16) = 2 blocks
        blks_1 = kv.append_turn("agent-101", tokens=list(range(30)), tool_context={"tool": "search"}, now=10.0)
        self.assertEqual(len(blks_1), 2)
        self.assertEqual(kv.get_session("agent-101").total_tokens, 30)
        self.assertEqual(len(kv.get_session_blocks("agent-101")), 2)

        # Turn 2: 10 tokens -> total 40 tokens -> requires ceil(40/16) = 3 blocks (1 new block)
        blks_2 = kv.append_turn("agent-101", tokens=list(range(10)), tool_context={"tool": "calc"}, now=20.0)
        self.assertEqual(len(blks_2), 1)
        self.assertEqual(kv.get_session("agent-101").total_tokens, 40)
        self.assertEqual(len(kv.get_session_blocks("agent-101")), 3)

        # Turn 3: 0 tokens -> empty turn doesn't allocate blocks
        blks_3 = kv.append_turn("agent-101", tokens=[], tool_context={}, now=30.0)
        self.assertEqual(blks_3, [])
        self.assertEqual(len(kv.get_session_blocks("agent-101")), 3)

    def test_kv_block_paged_allocation_and_math(self):
        """Verify exact block calculation matches ceiling division and block boundaries."""
        kv = SessionNativeKV(total_blocks=50, block_size=16)
        kv.create_session("session-blocks")

        # 16 tokens -> 1 block
        blks = kv.append_turn("session-blocks", tokens=list(range(16)))
        self.assertEqual(len(blks), 1)

        # 16 more tokens -> 1 block
        blks = kv.append_turn("session-blocks", tokens=list(range(16)))
        self.assertEqual(len(blks), 1)

        # 1 token -> 1 block (boundary crossing)
        blks = kv.append_turn("session-blocks", tokens=[999])
        self.assertEqual(len(blks), 1)

        # 15 tokens -> 0 new blocks (fits in existing partially filled block)
        blks = kv.append_turn("session-blocks", tokens=list(range(15)))
        self.assertEqual(len(blks), 0)

        # Total allocated blocks must be 3
        self.assertEqual(len(kv.get_session_blocks("session-blocks")), 3)
        self.assertEqual(kv.total_allocated_blocks(), 3)
        self.assertEqual(kv.available_blocks(), 47)

    def test_session_eviction_on_timeout(self):
        """Verify TTL expiration evicts inactive sessions and reclaims blocks to free pool."""
        kv = SessionNativeKV(total_blocks=20, block_size=16, ttl_s=50.0)

        kv.create_session("session-old", now=10.0)
        kv.append_turn("session-old", tokens=list(range(32)), now=10.0)  # 2 blocks

        kv.create_session("session-active", now=40.0)
        kv.append_turn("session-active", tokens=list(range(48)), now=40.0)  # 3 blocks

        self.assertEqual(kv.total_allocated_blocks(), 5)
        self.assertEqual(kv.available_blocks(), 15)

        # At now=70.0:
        # session-old inactive for 60s (> 50s TTL) -> evicted
        # session-active inactive for 30s (< 50s TTL) -> retained
        evicted = kv.evict_expired(now=70.0)
        self.assertEqual(evicted, ["session-old"])
        self.assertIsNone(kv.get_session("session-old"))
        self.assertIsNotNone(kv.get_session("session-active"))

        # Reclaimed 2 blocks -> available blocks is 17
        self.assertEqual(kv.available_blocks(), 17)
        self.assertEqual(kv.total_allocated_blocks(), 3)

    def test_session_eviction_lru(self):
        """Verify least-recently-active session is evicted under LRU policy."""
        kv = SessionNativeKV(total_blocks=100, block_size=16, max_sessions=3)

        kv.create_session("s1", now=10.0)
        kv.append_turn("s1", tokens=list(range(16)), now=10.0)

        kv.create_session("s2", now=20.0)
        kv.append_turn("s2", tokens=list(range(16)), now=20.0)

        kv.create_session("s3", now=30.0)
        kv.append_turn("s3", tokens=list(range(16)), now=30.0)

        # Access s1 again at now=40.0, making s2 the oldest (LRU)
        kv.append_turn("s1", tokens=list(range(16)), now=40.0)

        # Creating 4th session exceeds max_sessions=3, should trigger LRU eviction of s2
        kv.create_session("s4", now=50.0)
        self.assertIsNone(kv.get_session("s2"))
        self.assertIsNotNone(kv.get_session("s1"))
        self.assertIsNotNone(kv.get_session("s3"))
        self.assertIsNotNone(kv.get_session("s4"))

    def test_empty_session_query_and_edge_cases(self):
        """Verify safe handling of empty session query, missing sessions, and bad inputs."""
        kv = SessionNativeKV(total_blocks=50, block_size=16)

        # Non-existent session
        self.assertIsNone(kv.get_session("non-existent"))
        self.assertEqual(kv.get_session_blocks("non-existent"), [])

        # Empty session query
        self.assertIsNone(kv.get_session(""))
        self.assertEqual(kv.get_session_blocks(""), [])

        # Append turn to missing session raises KeyError
        with self.assertRaises(KeyError):
            kv.append_turn("missing-session", tokens=[1, 2, 3])

        # Invalid session creation raises ValueError
        with self.assertRaises(ValueError):
            kv.create_session("")
        with self.assertRaises(ValueError):
            kv.create_session(None)  # type: ignore

        # Invalid constructor arguments
        with self.assertRaises(ValueError):
            SessionNativeKV(total_blocks=0)
        with self.assertRaises(ValueError):
            SessionNativeKV(block_size=-1)
        with self.assertRaises(ValueError):
            SessionNativeKV(ttl_s=0)

    def test_out_of_memory_block_exhaustion(self):
        """Verify MemoryError is raised when GPU KV blocks are exhausted."""
        kv = SessionNativeKV(total_blocks=4, block_size=16)  # only 4 blocks total
        kv.create_session("sess-1")
        kv.append_turn("sess-1", tokens=list(range(64)))  # uses all 4 blocks

        self.assertEqual(kv.available_blocks(), 0)

        # Requesting another block raises MemoryError
        with self.assertRaises(MemoryError):
            kv.append_turn("sess-1", tokens=[1, 2, 3])

        # Evict session restores blocks
        kv.evict_session("sess-1")
        self.assertEqual(kv.available_blocks(), 4)

    def test_concurrent_sessions_thread_safety(self):
        """Verify thread safety when multiple worker threads concurrently access and mutate sessions."""
        kv = SessionNativeKV(total_blocks=200, block_size=16)
        num_threads = 10
        turns_per_thread = 5

        def worker_task(thread_id: int):
            sid = f"subagent-thread-{thread_id}"
            kv.create_session(sid, pin=True, now=1.0)
            for turn in range(turns_per_thread):
                tokens = list(range(10))
                kv.append_turn(sid, tokens=tokens, now=2.0 + turn)
                time.sleep(0.001)

        threads = [threading.Thread(target=worker_task, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all sessions exist with correct token counts
        for i in range(num_threads):
            sid = f"subagent-thread-{i}"
            sess = kv.get_session(sid)
            self.assertIsNotNone(sess)
            self.assertEqual(sess.total_tokens, turns_per_thread * 10)
            self.assertEqual(len(sess.turns), turns_per_thread)


class TestStreamingToolDispatcher(unittest.TestCase):
    """Test suite for Streaming Tool Call Dispatcher, asynchronous execution, and error resilience."""

    def test_tool_call_dispatch_demuxing_lifecycle(self):
        """Verify standard SSE token stream demuxing into structured tool_call_dispatch event."""
        buf = {}
        parsed = []
        stream = [
            {"type": "tool_use_open", "name": "vector_search", "id": "call_123"},
            {"delta": '{"query": "'},
            {"delta": "frontier llm "},
            {"delta": 'inference", "top_k": 5}'},
            {"type": "tool_use_close"},
        ]

        event = None
        for chunk in stream:
            ev = tool_call_dispatch(chunk, buf, parsed)
            if ev:
                event = ev

        self.assertIsNotNone(event)
        self.assertEqual(event["event"], "tool_call_dispatch")
        self.assertEqual(event["tool_call"]["name"], "vector_search")
        self.assertEqual(event["tool_call"]["id"], "call_123")
        self.assertEqual(event["tool_call"]["args"], '{"query": "frontier llm inference", "top_k": 5}')
        self.assertEqual(len(buf), 0)  # buffer cleared after closure

    def test_multiple_sequential_tool_dispatches(self):
        """Verify streaming multiple tool calls in sequence within a single turn."""
        buf = {}
        parsed = []
        stream = [
            {"name": "fetch_doc", "id": "t1"},
            {"delta": "doc_A"},
            {"type": "tool_use_close"},
            {"name": "fetch_doc", "id": "t2"},
            {"delta": "doc_B"},
            {"type": "tool_use_close"},
        ]

        events = []
        for chunk in stream:
            ev = tool_call_dispatch(chunk, buf, parsed)
            if ev:
                events.append(ev)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["tool_call"]["id"], "t1")
        self.assertEqual(events[0]["tool_call"]["args"], "doc_A")
        self.assertEqual(events[1]["tool_call"]["id"], "t2")
        self.assertEqual(events[1]["tool_call"]["args"], "doc_B")

    def test_tool_call_dispatch_malformed_and_empty_chunks(self):
        """Verify handling of empty or non-dict streaming chunks without crashing."""
        buf = {}
        parsed = []

        # Non-dict inputs
        self.assertIsNone(tool_call_dispatch("invalid_string", buf, parsed))  # type: ignore
        self.assertIsNone(tool_call_dispatch(None, buf, parsed))  # type: ignore

        # Empty dict chunk
        self.assertIsNone(tool_call_dispatch({}, buf, parsed))

        # Close without open/args emits empty tool_call safely
        ev = tool_call_dispatch({"type": "tool_use_close"}, buf, parsed)
        self.assertIsNotNone(ev)
        self.assertEqual(ev["tool_call"], {})

    def test_parallel_tool_dispatch_execution(self):
        """Verify dispatcher executes multiple tool calls concurrently in background worker threads."""
        with StreamingToolDispatcher(max_workers=4) as dispatcher:
            dispatcher.register_tool("add", lambda a, b: a + b)
            dispatcher.register_tool("mul", lambda a, b: a * b)

            stream = [
                {"name": "add"},
                {"delta": '{"a": 10, "b": 20}'},
                {"type": "tool_use_close"},
                {"name": "mul"},
                {"delta": '{"a": 5, "b": 6}'},
                {"type": "tool_use_close"},
            ]

            events = dispatcher.feed_stream(stream)
            self.assertEqual(len(events), 2)

            dispatcher.wait_all(timeout=2.0)
            self.assertEqual(len(dispatcher.tasks), 2)

            task_add = dispatcher.tasks[0]
            task_mul = dispatcher.tasks[1]

            self.assertEqual(task_add.status, "completed")
            self.assertEqual(task_add.result, 30)
            self.assertEqual(task_mul.status, "completed")
            self.assertEqual(task_mul.result, 30)

    def test_fast_vs_slow_tool_execution_interleaving(self):
        """Verify fast tool completes and is available before slow tool finishes, eliminating bubbles."""
        with StreamingToolDispatcher(max_workers=2) as dispatcher:
            def slow_tool():
                time.sleep(0.08)
                return "slow_done"

            def fast_tool():
                time.sleep(0.005)
                return "fast_done"

            dispatcher.register_tool("slow", slow_tool)
            dispatcher.register_tool("fast", fast_tool)

            # Dispatch slow tool first, then fast tool
            dispatcher.dispatch_async({"name": "slow", "args": ""})
            dispatcher.dispatch_async({"name": "fast", "args": ""})

            # Wait briefly so only fast tool has time to finish
            time.sleep(0.02)

            completed = dispatcher.get_completed_tasks()
            completed_names = [t.tool_name for t in completed]

            # Fast tool must be completed while slow tool is still running
            self.assertIn("fast", completed_names)
            self.assertNotIn("slow", completed_names)

            # Wait for all to finish
            dispatcher.wait_all(timeout=2.0)
            all_completed = [t.tool_name for t in dispatcher.get_completed_tasks()]
            self.assertIn("slow", all_completed)
            self.assertIn("fast", all_completed)

    def test_tool_error_handling_runtime_exception(self):
        """Verify dispatcher catches exceptions raised inside tool functions without crashing."""
        with StreamingToolDispatcher(max_workers=2) as dispatcher:
            def faulty_tool(val):
                if val == 0:
                    raise ZeroDivisionError("division by zero in tool kernel")
                return 100 / val

            dispatcher.register_tool("faulty", faulty_tool)

            stream = [
                {"name": "faulty"},
                {"delta": '{"val": 0}'},
                {"type": "tool_use_close"},
            ]
            dispatcher.feed_stream(stream)
            dispatcher.wait_all(timeout=2.0)

            self.assertEqual(len(dispatcher.tasks), 1)
            task = dispatcher.tasks[0]
            self.assertEqual(task.status, "error")
            self.assertIn("division by zero", task.error)
            self.assertEqual(task.error_type, "ZeroDivisionError")
            self.assertIsNone(task.result)

    def test_tool_error_handling_unregistered_tool(self):
        """Verify dispatching an unregistered tool sets error status and does not hang."""
        with StreamingToolDispatcher(max_workers=2) as dispatcher:
            stream = [
                {"name": "non_existent_tool"},
                {"delta": "args"},
                {"type": "tool_use_close"},
            ]
            dispatcher.feed_stream(stream)
            dispatcher.wait_all(timeout=2.0)

            self.assertEqual(len(dispatcher.tasks), 1)
            task = dispatcher.tasks[0]
            self.assertEqual(task.status, "error")
            self.assertIn("not registered", task.error)
            self.assertEqual(task.error_type, "KeyError")

    def test_dispatcher_invalid_initialization(self):
        """Verify StreamingToolDispatcher raises ValueError on invalid worker count or tool registration."""
        with self.assertRaises(ValueError):
            StreamingToolDispatcher(max_workers=0)

        dispatcher = StreamingToolDispatcher(max_workers=1)
        with self.assertRaises(ValueError):
            dispatcher.register_tool("", lambda x: x)
        with self.assertRaises(ValueError):
            dispatcher.register_tool("tool", "not_callable")  # type: ignore
        dispatcher.shutdown()


class TestToolwaitOffloadDecision(unittest.TestCase):
    """Test suite for Toolwait Offload Decision Engine, breakeven wait time math, and PCIe/CXL modeling."""

    def test_canonical_chapter_scenarios(self):
        """Verify exact canonical decisions from Chapter 12 book scenarios."""
        # 1. Fast Web API (250 ms, 52,000 tok) -> RETAIN
        self.assertEqual(kv_during_wait(250, 52000), "retain")

        # 2. Database Join (3500 ms, 52,000 tok) -> OFFLOAD
        self.assertEqual(kv_during_wait(3500, 52000), "offload")

        # 3. Human-in-the-Loop Approval (45000 ms, 52,000 tok) -> DISCARD
        self.assertEqual(kv_during_wait(45000, 52000), "discard")

    def test_toolwait_optimizer_alias_compatibility(self):
        """Verify ToolwaitOptimizer alias points to ToolwaitOffloadDecision."""
        self.assertIs(ToolwaitOptimizer, ToolwaitOffloadDecision)

    def test_kv_during_wait_boundary_and_edge_cases(self):
        """Verify edge cases for kv_during_wait: zero wait, zero tokens, negative inputs."""
        # 0 ms wait -> retain
        self.assertEqual(kv_during_wait(0, 52000), "retain")

        # 0 tokens -> retain
        self.assertEqual(kv_during_wait(1000, 0), "retain")

        # Negative inputs must raise ValueError
        with self.assertRaises(ValueError):
            kv_during_wait(-100, 52000)
        with self.assertRaises(ValueError):
            kv_during_wait(250, -500)

        # Boundary checks around 400ms, 8000ms, 20000ms
        self.assertEqual(kv_during_wait(400, 52000), "retain")
        self.assertEqual(kv_during_wait(401, 52000), "offload")
        self.assertEqual(kv_during_wait(20000, 52000), "offload")
        self.assertEqual(kv_during_wait(20001, 52000), "discard")

    def test_transfer_time_latency_scaling(self):
        """Verify transfer latency scales inversely with interconnect bandwidth (PCIe Gen4 vs Gen5 vs CXL 3.0)."""
        optimizer = ToolwaitOffloadDecision(bytes_per_tok=128)
        prefix_tok = 52000

        t_gen4 = optimizer.calculate_transfer_time_ms(prefix_tok, bandwidth_gb_s=32.0)
        t_gen5 = optimizer.calculate_transfer_time_ms(prefix_tok, bandwidth_gb_s=64.0)
        t_cxl = optimizer.calculate_transfer_time_ms(prefix_tok, bandwidth_gb_s=128.0)

        # CXL (128 GB/s) is 2x faster than Gen5 (64 GB/s) and 4x faster than Gen4 (32 GB/s)
        self.assertAlmostEqual(t_gen4 / t_gen5, 2.0, places=4)
        self.assertAlmostEqual(t_gen5 / t_cxl, 2.0, places=4)
        self.assertAlmostEqual(t_gen4 / t_cxl, 4.0, places=4)

        # 0 tokens transfer time must be 0.0
        self.assertEqual(optimizer.calculate_transfer_time_ms(0, 64.0), 0.0)

    def test_breakeven_retain_vs_offload_bandwidth_hierarchy(self):
        """Verify higher interconnect bandwidth lowers breakeven wait time, enabling earlier offloads."""
        optimizer = ToolwaitOffloadDecision()
        prefix_tok = 52000

        be_gen4 = optimizer.calculate_breakeven_retain_vs_offload_ms(prefix_tok, bandwidth_gb_s=32.0)
        be_gen5 = optimizer.calculate_breakeven_retain_vs_offload_ms(prefix_tok, bandwidth_gb_s=64.0)
        be_cxl = optimizer.calculate_breakeven_retain_vs_offload_ms(prefix_tok, bandwidth_gb_s=128.0)

        # Higher bandwidth -> lower transfer penalty -> lower breakeven threshold
        self.assertGreater(be_gen4, be_gen5)
        self.assertGreater(be_gen5, be_cxl)

        # Values around 800ms for Gen4, 400ms for Gen5, 200ms for CXL
        self.assertGreater(be_gen4, 750.0)
        self.assertLess(be_gen4, 850.0)
        self.assertGreater(be_gen5, 380.0)
        self.assertLess(be_gen5, 420.0)
        self.assertGreater(be_cxl, 180.0)
        self.assertLess(be_cxl, 220.0)

    def test_dynamic_vram_pressure_adaptation(self):
        """Verify high VRAM pressure forces earlier offloading to host DRAM to prevent OOM."""
        optimizer = ToolwaitOffloadDecision()
        prefix_tok = 52000
        bw = 64.0

        be_low_pressure = optimizer.calculate_breakeven_retain_vs_offload_ms(prefix_tok, bw, vram_pressure=0.0)
        be_med_pressure = optimizer.calculate_breakeven_retain_vs_offload_ms(prefix_tok, bw, vram_pressure=0.5)
        be_high_pressure = optimizer.calculate_breakeven_retain_vs_offload_ms(prefix_tok, bw, vram_pressure=0.9)

        # As VRAM pressure mounts, breakeven threshold collapses to prioritize VRAM eviction
        self.assertGreater(be_low_pressure, be_med_pressure)
        self.assertGreater(be_med_pressure, be_high_pressure)
        self.assertLess(be_high_pressure, 150.0)  # Severe pressure offloads quickly

    def test_offload_vs_discard_breakeven_calculation(self):
        """Verify breakeven between host DRAM retention and prefill recomputation."""
        optimizer = ToolwaitOffloadDecision()
        prefix_tok = 52000

        be_discard = optimizer.calculate_breakeven_offload_vs_discard_ms(prefix_tok, bandwidth_gb_s=64.0)
        # Recompute budget / dram_rate ~ 1e-4 / 5e-9 = 20,000 ms minus transfer cost
        self.assertGreater(be_discard, 19500.0)
        self.assertLess(be_discard, 20000.0)

        # 0 tokens returns infinity
        self.assertEqual(optimizer.calculate_breakeven_offload_vs_discard_ms(0, 64.0), float("inf"))

    def test_dynamic_decision_matrix(self):
        """Verify complete decision matrix across regimes (retain, offload, discard)."""
        optimizer = ToolwaitOffloadDecision()
        prefix = 52000

        # Short wait under PCIe Gen5 (64 GB/s) -> retain
        self.assertEqual(optimizer.decide(wait_ms=250.0, prefix_tok=prefix, bandwidth_gb_s=64.0), "retain")

        # Medium wait under PCIe Gen5 (64 GB/s) -> offload
        self.assertEqual(optimizer.decide(wait_ms=3500.0, prefix_tok=prefix, bandwidth_gb_s=64.0), "offload")

        # Long wait -> discard
        self.assertEqual(optimizer.decide(wait_ms=45000.0, prefix_tok=prefix, bandwidth_gb_s=64.0), "discard")

        # CXL enables offloading for 300 ms wait where Gen5 would retain
        self.assertEqual(optimizer.decide(wait_ms=300.0, prefix_tok=prefix, bandwidth_gb_s=64.0), "retain")
        self.assertEqual(optimizer.decide(wait_ms=300.0, prefix_tok=prefix, bandwidth_gb_s=128.0), "offload")

        # High VRAM pressure (0.9) offloads 250 ms wait where normal pressure would retain
        self.assertEqual(optimizer.decide(wait_ms=250.0, prefix_tok=prefix, bandwidth_gb_s=64.0, vram_pressure=0.0), "retain")
        self.assertEqual(optimizer.decide(wait_ms=250.0, prefix_tok=prefix, bandwidth_gb_s=64.0, vram_pressure=0.9), "offload")

    def test_optimizer_invalid_inputs(self):
        """Verify input validation raises ValueError on illegal parameters."""
        optimizer = ToolwaitOffloadDecision()
        with self.assertRaises(ValueError):
            optimizer.calculate_transfer_time_ms(100, bandwidth_gb_s=0.0)
        with self.assertRaises(ValueError):
            optimizer.calculate_breakeven_retain_vs_offload_ms(100, bandwidth_gb_s=64.0, vram_pressure=-0.1)
        with self.assertRaises(ValueError):
            optimizer.calculate_breakeven_retain_vs_offload_ms(100, bandwidth_gb_s=64.0, vram_pressure=1.5)
        with self.assertRaises(ValueError):
            optimizer.decide(wait_ms=-1.0, prefix_tok=100)
        with self.assertRaises(ValueError):
            optimizer.decide(wait_ms=100.0, prefix_tok=-10)

        # Invalid constructor params
        with self.assertRaises(ValueError):
            ToolwaitOffloadDecision(bytes_per_tok=0)
        with self.assertRaises(ValueError):
            ToolwaitOffloadDecision(vram_hold_per_ms_tok=-1.0)
        with self.assertRaises(ValueError):
            ToolwaitOffloadDecision(prefill_cost_per_tok=0.0)
        with self.assertRaises(ValueError):
            ToolwaitOffloadDecision(min_retain_ms=-5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
