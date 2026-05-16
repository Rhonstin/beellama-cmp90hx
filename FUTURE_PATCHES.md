# CMP 90HX Optimization Roadmap

**Fork**: beellama.cpp + CMP 90HX patches  
**Last updated**: 2026-05-16  
**Purpose**: Track completed and planned optimizations for NVIDIA CMP 90HX (GA102, sm_86).  
Check this file at the start of each work session and update it as tasks are completed.

---

## Hardware Context

CMP 90HX throttles certain instructions in firmware:

| Instruction | Latency | Factor |
|---|---|---|
| FFMA / FADD / FMUL | 17.8 ns | 14–15× slower |
| DP4A | 35.6 ns | 29× slower |
| IMAD / IADD | 1.4 ns | **unthrottled** |
| HFMA2 / HADD2 | 1.3–1.4 ns | **unthrottled** |
| Tensor cores (HMMA/IMMA) | 284.5 ns | severely throttled — DO NOT USE |

---

## Phase 0 — Setup ✅ DONE (2026-05-16)

- [x] Fork beellama.cpp into `/home/rhonstin/beellama-cmp90hx`
- [x] Port all CMP 90HX patches from llama.cpp fork
- [x] Set up `bench/cmp90hx/` benchmark directory with historical results
- [x] Update root README with CMP 90HX section
- [x] Initial build (pending GPU availability)

---

## Phase 1 — Core decode patches ✅ DONE

All applied to this fork in initial commit.

| Patch | File | Gain |
|---|---|---|
| DP4A → PTX IMAD (all quantizations) | `ggml/src/ggml-cuda/common.cuh` | +47.6% |
| HFMA2 dequant — Q4_K decode | `ggml/src/ggml-cuda/vecdotq.cuh` | +7.1% |
| HFMA2 dequant — Q5_K decode | `ggml/src/ggml-cuda/vecdotq.cuh` | included above |
| HFMA2 dequant — Q6_K decode | `ggml/src/ggml-cuda/vecdotq.cuh` | +4.2% |
| HFMA2 dequant — Q2_K decode | `ggml/src/ggml-cuda/vecdotq.cuh` | minor |
| fattn DKQ=512 crash fix | `ggml/src/ggml-cuda/fattn.cu` | crash fix |

**Baseline model**: Gemma-4 E4B Q4_K_XL  
**Cumulative vs unpatched beellama.cpp**: ~+58% on Q4_K, ~+87% on Q4_K_XL

---

## Phase 2 — Beellama-specific integration (TODO)

These tasks are specific to beellama.cpp features not present in upstream llama.cpp.

### 2a. Benchmark beellama.cpp features on CMP 90HX

- [ ] Run DFlash speculative decoding baseline (Qwen 3.6 27B + DFlash drafter)
  - Expected: CMP 90HX decode is now fast enough for DFlash to be meaningful
  - File results in `bench/cmp90hx/DFLASH_BENCH.md`
- [ ] Benchmark TurboQuant KV types (`turbo4`, `turbo3`, `turbo2`) on CMP 90HX
  - Previous tests on unpatched llama.cpp: turbo3 was net-negative (-6.8%)
  - With HFMA2 decode patches, the compute/memory balance may shift
  - File results in `bench/cmp90hx/BEELLAMA_TURBO_BENCH.md`

### 2b. DFlash hidden-state accumulation HFMA2

**File**: `src/llama-speculative.cpp` or DFlash CUDA kernel  
Check if DFlash draft cross-attention uses FP32 accumulation that could benefit from HFMA2.

### 2c. TurboQuant dequant → HFMA2

**File**: TurboQuant KV dequant kernel (exact file TBD)  
If TurboQuant dequant uses FP32 FFMA, replace with HFMA2 under `#if __CUDA_ARCH__ == 860`.

---

## Phase 3 — Prefill GEMM (largest remaining opportunity)

**Status**: Not started. Complex kernel engineering required.

CMP 90HX prefill is currently catastrophically slow because cuBLAS SGEMM uses FP32 FFMA (14× throttled) and tensor cores are severely throttled (200×). A custom tiled GEMM using HFMA2 in shared memory is the only path to reasonable prefill performance.

### 3a. Custom HFMA2 tiled GEMM for prefill

**Files**: new file `ggml/src/ggml-cuda/cmp90hx-gemm.cu`  
**Design**:
- 64×64 output tiles, 8×8 thread blocks
- Load A/B tiles into `__shared__` as `half2`
- Inner loop: `__hfma2` accumulation (unthrottled)
- Write back converting to float

**Entry point**: Replace `ggml_cuda_op_mul_mat_cublas` call for sm_86 only.

**Estimated gain**: 10–15× prefill speedup (from near-zero to usable).

### 3b. RMSNorm variance → half2

**File**: `ggml/src/ggml-cuda/norm.cu`  
Low priority for decode (single token), but matters for prefill and batch mode.

### 3c. ROPE application → half2

**File**: `ggml/src/ggml-cuda/rope.cu`  
`cosf`/`sinf` must stay FP32 (special function), but the rotation `x*cos - x_rot*sin`
could use HFMA2 if input activations are in fp16. Medium impact, low risk.

---

## Phase 4 — Long-context decode (Tier 2 KQ patches)

**Status**: Benchmarked on 50-token context — no gain. Worth revisiting for 2k+ token contexts.

### 4a. Flash attention KQ accumulator: `float[]` → `half2[]`

**File**: `ggml/src/ggml-cuda/fattn-tile.cuh` ~line 591  
At 2k+ tokens, KQ dot product becomes a larger fraction of compute, and replacing
2 FP32 FADDs with 1 HFMA2 per step would matter.

**Risk**: fp16 range overflow for long sequences or large head_dim. Requires softmax normalization audit.

### 4b. `ggml_cuda_mad` HADD2 optimization

**File**: `ggml/src/ggml-cuda/common.cuh` ~line 745  
Replace `float acc += tmp.x + tmp.y` with `HADD2 → float conversion → FP32 add`.

---

## Notes for Future Sessions

- Always run `bench/cmp90hx` benchmarks before and after each change
- Report format: `[change] before=XX tok/s after=YY tok/s delta=+ZZ%`
- Target arch: sm_86 only — guard all CMP patches with `#if __CUDA_ARCH__ == 860`
- Do NOT use tensor cores (HMMA/IMMA)
- Build command: `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)`
