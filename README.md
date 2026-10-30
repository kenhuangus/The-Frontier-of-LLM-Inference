# The Frontier of LLM Inference (2026 Edition)
## Exclusive Companion Repository for Book Readers

**Authors:**
* **Ken Huang** — Adjunct Professor, University of San Francisco (USF); CEO & Chief AI Officer, DistributedApps.ai
* **Manish Bhatt** — Application Security Tech Lead for AI & Crypto, Amazon Leo

---

## About the Book

***The Frontier of LLM Inference: The Physics and Engineering of Frontier LLM Inference (2026 Edition)*** provides a rigorous, first-principles examination of high-throughput, low-latency AI serving systems. 

Moving beyond high-level abstractions, the book covers the physical and architectural constraints governing modern LLM serving infrastructure:
* **The Physics of LLM Inference & Roofline Models:** Hardware limits, arithmetic intensity collapse during decode, and compute-vs-memory bound transitions across NVIDIA Hopper (H100) and Blackwell (B200) architectures.
* **The KV Cache Frontier:** Multi-Head Latent Attention (MLA), low-rank key-value compression, global prefix caching with Radix trees, and cross-layer memory recycling.
* **Speculative Decoding Engines:** Leviathan-Chen verification, $(K+1)$-th bonus token dynamics, Multi-Token Prediction (MTP), and EAGLE-2 dynamic candidate trees.
* **Extreme Quantization & Precision Engineering:** Blackwell OCP MXFP4 micro-scaling (E2M1), native FP8 matrix multiplication, and KIVI asymmetric 2-bit KV quantization.
* **Hardware-Aware Attention Kernels:** FlashAttention-3 warp-specialized asynchronous TMA pipelines and FlashDecoding Split-K parallel reduction trees.
* **Disaggregated Serving Architectures:** Prefill-Decode (PD) contention decoupling, Mooncake-style 4-tier KV caching, and sub-millisecond RoCEv2 RDMA transfers.
* **Mega-MoE at Scale:** DeepSeek DualPipe bidirectional pipeline scheduling and Expert Parallelism (EP) all-to-all token routing under capacity constraints.
* **Test-Time Compute & Reasoning Dynamics:** Serving inverted 95% decode loops, adaptive thinking token budgets, and stream demuxing for frontier reasoning models (DeepSeek-V4-Pro, GLM-5.3, Kimi K3).
* **Ultra-Long Context Mastery:** Dual-Chunk Attention (DCA) and YaRN RoPE spectral frequency modulation.
* **Production SRE & Constrained Decoding:** GPU bitmask evaluations with XGrammar FSMs and rigorous cluster TCO modeling.
* **Token Efficiency & Unit Economics:** Consistent-hashing gateway affinity routing, prefix caching multipliers, and software-factory cost equations.
* **Agentic Inference Systems:** Session-native paged KV caching, streaming tool dispatchers, and tool-wait offloading economics (VRAM vs Host DRAM via PCIe Gen5 / CXL 3.0).

---

## Reader Bonus Material

This GitHub repository is provided as an **exclusive practical bonus for readers of the book**. It contains complete, standalone reference implementations and mathematical simulations corresponding directly to the equations and system architectures analyzed throughout each chapter.

### Design Principles:
1. **Zero External Dependencies:** Every algorithm and simulation is implemented using Python 3.9+ standard library modules (`math`, `statistics`, `dataclasses`, `random`, `time`, `typing`). You do not need GPU hardware, specialized CUDA drivers, or third-party compilers to execute and study these systems.
2. **Comprehensive Unit Test Coverage:** Each chapter includes a dedicated unit test suite exercising realistic inputs, edge cases, boundary conditions, and mathematical invariants (336 verified unit tests in total).

---

## Quickstart

Verify all 30 reference implementations and 12 unit test suites with the master test runner:

```bash
python run_all_code_tests.py
```

### Running Individual Chapters
You can execute any reference simulation or its unit test suite directly:

```bash
# Chapter 1: Physics, Roofline Models, and Continuous Batching
python chapter_01_physics_and_roofline/roofline_analyzer.py
python chapter_01_physics_and_roofline/test_chapter_01.py

# Chapter 2: Radix-Tree Prefix Cache & Multi-Head Latent Attention (MLA)
python chapter_02_kv_cache_mla/radix_prefix_cache.py
python chapter_02_kv_cache_mla/test_chapter_02.py

# Chapter 5: FlashAttention-3 Pipeline & FlashDecoding Split-K
python chapter_05_attention_kernels/flashdecoding_split_kv.py
python chapter_05_attention_kernels/test_chapter_05.py

# Chapter 7: DualPipe Schedule & Expert Parallelism
python chapter_07_mega_moe_dualpipe/dualpipe_schedule.py
python chapter_07_mega_moe_dualpipe/test_chapter_07.py

# Chapter 12: Session-Native KV & Streaming Tool Dispatcher
python chapter_12_agentic_inference/session_native_kv.py
python chapter_12_agentic_inference/test_chapter_12.py
```

---

## Repository Structure

```text
├── run_all_code_tests.py               # Master automated verification test harness
├── chapter_01_physics_and_roofline/    # Roofline models, KV sizing, serving simulator, telemetry
├── chapter_02_kv_cache_mla/            # Radix prefix tree, MLA simulation, KV benchmark
├── chapter_03_speculative_decoding/    # Speculative rejection sampling, candidate tree
├── chapter_04_quantization_fp4_fp8/    # Blackwell MXFP4 microscaling, KIVI quantizer, FastRMSNorm
├── chapter_05_attention_kernels/       # FlashAttention-3 pipeline, FlashDecoding Split-K
├── chapter_06_disaggregated_serving/   # Disaggregated P/D queue simulator, RDMA KV transfer
├── chapter_07_mega_moe_dualpipe/       # DualPipe bidirectional schedule, expert parallelism
├── chapter_08_test_time_compute/       # Dynamic thinking budget controller, stream demuxer
├── chapter_09_long_context_dca_yarn/   # Dual-Chunk Attention, YaRN RoPE spectral scaling
├── chapter_10_production_sre_xgrammar/ # XGrammar FSM bitmask evaluator, cluster sizing & TCO
├── chapter_11_token_efficiency/        # Factory spend decomposition, gateway affinity router
└── chapter_12_agentic_inference/       # Session-native KV, streaming tool dispatcher, offload optimizer
```
