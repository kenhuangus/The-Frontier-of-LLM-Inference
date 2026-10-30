# Companion Code: The Frontier of LLM Inference (2026 Edition)

**Authors:**
* **Ken Huang** — Adjunct Professor, University of San Francisco (USF); CEO & Chief AI Officer, DistributedApps.ai
* **Manish Bhatt** — Application Security Tech Lead for AI & Crypto, Amazon Leo

---

## Overview

This repository contains the official, verified companion reference implementations and mathematical simulations for the book ***The Frontier of LLM Inference (2026 Edition): The Physics and Engineering of Frontier LLM Inference***, published by Packt Publishing.

All 30 reference implementations and 12 dedicated unit test suites (covering 336 individual test assertions across all 12 chapters) are engineered to execute out-of-the-box on standard Python 3.9+ environments across Linux, macOS, and Windows. Every algorithm, simulation, and benchmark has been tested with zero runtime errors.

---

## Architecture and Engineering Principles

1. **Zero External Dependency Barrier:** The core reference algorithms rely exclusively on standard Python 3.9+ runtime libraries (`math`, `statistics`, `dataclasses`, `random`, `time`, `typing`). Readers can clone and run any chapter immediately without encountering broken CUDA drivers, compiler toolchain mismatches, or platform incompatibilities.
2. **Bit-Exact Architectural Fidelity:** Each script models the real physical behaviors of hardware architectures (NVIDIA Blackwell B200, Hopper H100, Google TPU Trillium) and serving frameworks (vLLM V1, SGLang, Mooncake, DeepSeek-V4).
3. **Comprehensive Unit Testing (336 Test Cases):** Every chapter includes a dedicated `test_chapter_XX.py` suite exercising boundary conditions, zero inputs, concurrency, and mathematical invariants.
4. **Automated Verification:** The repository includes a master test harness (`run_all_code_tests.py`) that executes all 42 scripts and test suites in sequence and verifies successful completion.

---

## Quickstart

### Prerequisites
* Python 3.9 or higher.

### Run All Tests
To verify all 42 chapter scripts and unit test suites in one automated pass:
```bash
python run_all_code_tests.py
```
Expected output:
```text
================================================================================
  RUNNING ALL PRODUCTION CODE TESTS: THE FRONTIER OF LLM INFERENCE
================================================================================
  [PASS] chapter_01_physics_and_roofline\kv_cache_sizing.py
  [PASS] chapter_01_physics_and_roofline\roofline_analyzer.py
  [PASS] chapter_01_physics_and_roofline\serving_simulator.py
  [PASS] chapter_01_physics_and_roofline\telemetry_tracker.py
  [PASS] chapter_01_physics_and_roofline\test_chapter_01.py
  [PASS] chapter_02_kv_cache_mla\kv_cache_benchmark.py
  [PASS] chapter_02_kv_cache_mla\mla_simulation.py
  [PASS] chapter_02_kv_cache_mla\radix_prefix_cache.py
  [PASS] chapter_02_kv_cache_mla\test_chapter_02.py
  [PASS] chapter_03_speculative_decoding\speculative_sampling.py
  [PASS] chapter_03_speculative_decoding\speculative_tree.py
  [PASS] chapter_03_speculative_decoding\test_chapter_03.py
  [PASS] chapter_04_quantization_fp4_fp8\fast_rmsnorm.py
  [PASS] chapter_04_quantization_fp4_fp8\kivi_quantizer.py
  [PASS] chapter_04_quantization_fp4_fp8\mxfp4_microscaling.py
  [PASS] chapter_04_quantization_fp4_fp8\test_chapter_04.py
  [PASS] chapter_05_attention_kernels\flashattention3_pipeline.py
  [PASS] chapter_05_attention_kernels\flashdecoding_split_kv.py
  [PASS] chapter_05_attention_kernels\test_chapter_05.py
  [PASS] chapter_06_disaggregated_serving\disaggregated_pd_simulator.py
  [PASS] chapter_06_disaggregated_serving\rdma_kv_transfer.py
  [PASS] chapter_06_disaggregated_serving\test_chapter_06.py
  [PASS] chapter_07_mega_moe_dualpipe\dualpipe_schedule.py
  [PASS] chapter_07_mega_moe_dualpipe\expert_parallelism.py
  [PASS] chapter_07_mega_moe_dualpipe\test_chapter_07.py
  [PASS] chapter_08_test_time_compute\dynamic_budget_controller.py
  [PASS] chapter_08_test_time_compute\reasoning_stream_demuxer.py
  [PASS] chapter_08_test_time_compute\test_chapter_08.py
  [PASS] chapter_09_long_context_dca_yarn\dual_chunk_attention.py
  [PASS] chapter_09_long_context_dca_yarn\test_chapter_09.py
  [PASS] chapter_09_long_context_dca_yarn\yarn_rope_scaling.py
  [PASS] chapter_10_production_sre_xgrammar\cluster_sizing_tco.py
  [PASS] chapter_10_production_sre_xgrammar\test_chapter_10.py
  [PASS] chapter_10_production_sre_xgrammar\xgrammar_fsm_evaluator.py
  [PASS] chapter_11_token_efficiency\factory_spend_decomposition.py
  [PASS] chapter_11_token_efficiency\gateway_affinity_router.py
  [PASS] chapter_11_token_efficiency\prompt_cache_multipliers.py
  [PASS] chapter_11_token_efficiency\test_chapter_11.py
  [PASS] chapter_12_agentic_inference\session_native_kv.py
  [PASS] chapter_12_agentic_inference\streaming_tool_dispatcher.py
  [PASS] chapter_12_agentic_inference\test_chapter_12.py
  [PASS] chapter_12_agentic_inference\toolwait_offload_decision.py
================================================================================
  TEST SUITE RESULTS: 42 / 42 PASSED (in 5.19s)
  ALL CHAPTER SCRIPTS EXECUTED SUCCESSFULLY WITH ZERO ERRORS!
================================================================================
```

### Run Individual Chapter Unit Tests
You can run any chapter unit test suite directly:
```bash
python chapter_01_physics_and_roofline/test_chapter_01.py
python chapter_02_kv_cache_mla/test_chapter_02.py
python chapter_07_mega_moe_dualpipe/test_chapter_07.py
```

---

## Directory Structure and Technical Scope

| Chapter | Directory | Script | Description |
| :--- | :--- | :--- | :--- |
| **Ch 01** | `chapter_01_physics_and_roofline` | `roofline_analyzer.py` | Arithmetic intensity & compute vs memory bound classification (H100 vs B200) |
| | | `kv_cache_sizing.py` | Exact KV cache memory allocation equations for MHA, GQA, and MLA |
| | | `serving_simulator.py` | Discrete-event discrete simulation of prefill vs decode batching queues |
| | | `telemetry_tracker.py` | Token generation latency, TTFT, and inter-token arrival time tracking |
| **Ch 02** | `chapter_02_kv_cache_mla` | `radix_prefix_cache.py` | Production Radix-tree prefix cache with LRU eviction and memory recycling |
| | | `mla_simulation.py` | DeepSeek Multi-Head Latent Attention low-rank compression (94.7% VRAM reduction) |
| | | `kv_cache_benchmark.py` | Latency and throughput benchmark comparing Standard MHA vs Radix-cached MLA |
| **Ch 03** | `chapter_03_speculative_decoding` | `speculative_sampling.py` | Mathematical verification of lossless speculative rejection sampling |
| | | `speculative_tree.py` | EAGLE-2 dynamic speculative tree candidate expansion and verification |
| **Ch 04** | `chapter_04_quantization_fp4_fp8` | `mxfp4_microscaling.py` | Blackwell OCP MXFP4 (E2M1) block microscaling and quantization noise analysis |
| | | `kivi_quantizer.py` | KIVI asymmetric 2-bit per-channel Key and per-token Value cache quantization |
| | | `fast_rmsnorm.py` | Fast numerical RMSNorm layer implementation |
| **Ch 05** | `chapter_05_attention_kernels` | `flashattention3_pipeline.py` | Hopper TMA asynchronous copy and warp-specialized ping-pong GEMM simulation |
| | | `flashdecoding_split_kv.py` | Split-K FlashDecoding parallel KV reduction tree across GPU streaming multiprocessors |
| **Ch 06** | `chapter_06_disaggregated_serving` | `disaggregated_pd_simulator.py` | Prefill-Decode disaggregation queue simulator comparing coupled vs disaggregated P99 TTFT/ITL |
| | | `rdma_kv_transfer.py` | GPUDirect RDMA over RoCEv2 KV transfer throughput and network pipeline simulation |
| **Ch 07** | `chapter_07_mega_moe_dualpipe` | `dualpipe_schedule.py` | DeepSeek DualPipe bidirectional pipeline parallelism overlap schedule (forward & backward compute) |
| | | `expert_parallelism.py` | All-to-all expert token dispatch, routing capacity, and load balancing simulation |
| **Ch 08** | `chapter_08_test_time_compute` | `dynamic_budget_controller.py` | Dynamic reasoning budget allocation adjusting thinking depth based on prompt complexity |
| | | `reasoning_stream_demuxer.py` | Zero-allocation demultiplexer separating `<think>` tokens from final user responses |
| **Ch 09** | `chapter_09_long_context_dca_yarn` | `dual_chunk_attention.py` | Alibaba Qwen Dual-Chunk Attention (DCA) intra-chunk and inter-chunk receptive fields |
| | | `yarn_rope_scaling.py` | YaRN (Yet another RoPE extensioN) frequency interpolation and progressive wavelength modulation |
| **Ch 10** | `chapter_10_production_sre_xgrammar` | `xgrammar_fsm_evaluator.py` | Sub-microsecond Finite State Machine grammar parser for deterministic JSON schema masking |
| | | `cluster_sizing_tco.py` | Total Cost of Ownership (TCO) calculator sizing H100 vs B200 clusters under strict SLA constraints |
| **Ch 11** | `chapter_11_token_efficiency` | `factory_spend_decomposition.py` | Token expenditure decomposition across prefill, decode, and cache reuse layers |
| | | `gateway_affinity_router.py` | Consistent hashing and session-affinity load balancer maximizing prefix cache hit rates |
| | | `prompt_cache_multipliers.py` | Mathematical cost multiplier analysis for tiered multi-turn prompt caching |
| **Ch 12** | `chapter_12_agentic_inference` | `session_native_kv.py` | Multi-turn agentic session KV manager retaining tool execution context across turns |
| | | `streaming_tool_dispatcher.py` | Early tool call extraction and parallel execution engine for agent tool loops |
| | | `toolwait_offload_decision.py` | Dynamic offloading cost decision engine evaluating GPU VRAM vs host DRAM during tool wait states |

---

## Running Individual Scripts

You can run any individual script directly from the project root:

```bash
python chapter_01_physics_and_roofline/roofline_analyzer.py
python chapter_02_kv_cache_mla/radix_prefix_cache.py
python chapter_07_mega_moe_dualpipe/dualpipe_schedule.py
```

---

## Citation and References

If you use these implementations or architectures in your research or production serving infrastructure, please cite:

```bibtex
@book{huang2026frontier,
  title     = {The Frontier of LLM Inference: The Physics and Engineering of Frontier LLM Inference (2026 Edition)},
  author    = {Huang, Ken and Bhatt, Manish},
  publisher = {Packt Publishing},
  year      = {2026},
  isbn      = {978-1-83763-000-0}
}
```

---

## License

This companion code repository is distributed under the MIT License.
