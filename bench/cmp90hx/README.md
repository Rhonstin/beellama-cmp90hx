# llama.cpp — CMP 90HX Optimization

This fork patches llama.cpp to run efficiently on the **NVIDIA CMP 90HX** — a
GA102-based mining card that Nvidia artificially throttles to be useless for AI
inference. We bypass those throttles by replacing every slow instruction with an
unthrottled equivalent, recovering most of the card's real compute throughput.

Based on [AtomicBot-ai/atomic-llama-cpp-turboquant](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant).

---

## The Problem

The CMP 90HX is a GA102 chip (same die as RTX 3090) with 9877 MiB VRAM. Nvidia
sells it at GPU compute prices but firmware-gates the arithmetic units so that
most AI workloads run at 1/14th to 1/29th of their theoretical throughput:

| Instruction | Latency | Throttle |
|---|---:|---|
| FFMA (FP32 mul-add) | 17.8 ns | **14×** slowdown |
| FADD / FMUL | 17.8 ns | **15×** slowdown |
| DP4A (INT8 dot product) | 35.6 ns | **29×** slowdown |
| Tensor cores (HMMA/IMMA) | 284.5 ns | **~200×** slowdown |
| **IMAD** (INT multiply-add) | **1.4 ns** | **unthrottled** |
| **IADD** (INT add) | **0.51 ns** | **unthrottled** |
| **HFMA2** (FP16×2 mul-add) | **1.4 ns** | **unthrottled** |
| **HADD2 / HMUL2** | **1.3 ns** | **unthrottled** |

Stock llama.cpp uses DP4A for quantized dot products and FP32 FFMA for
dequantization — both heavily throttled. The card sits at full CUDA occupancy
while delivering a fraction of its rated throughput.

---

## The Fix

Two targeted PTX-level patches applied only for `__CUDA_ARCH__ == 860` (sm_86):

### Patch 1 — DP4A → IMAD (`common.cuh`)

`__dp4a(a, b, c)` expands to a single DP4A instruction (29× throttled). We
replace it with inline PTX that byte-extracts `a` and `b` and accumulates with
four `mad.lo.s32` instructions (IMAD — unthrottled):

```cuda
asm volatile(
    "{ .reg .s32 t0,t1,t2,t3, u0,u1,u2,u3;\n"
    "  bfe.s32 t0,%1, 0,8; bfe.s32 t1,%1, 8,8;\n"
    "  bfe.s32 t2,%1,16,8; bfe.s32 t3,%1,24,8;\n"
    "  bfe.s32 u0,%2, 0,8; bfe.s32 u1,%2, 8,8;\n"
    "  bfe.s32 u2,%2,16,8; bfe.s32 u3,%2,24,8;\n"
    "  mad.lo.s32 %0,t0,u0,%3;\n"
    "  mad.lo.s32 %0,t1,u1,%0;\n"
    "  mad.lo.s32 %0,t2,u2,%0;\n"
    "  mad.lo.s32 %0,t3,u3,%0; }"
    : "=r"(c) : "r"(a), "r"(b), "r"(c)
);
```

### Patch 2 — FP32 dequant accumulation → HFMA2 (`vecdotq.cuh`)

The MMVQ dequantization kernel accumulates scale-and-bias corrections in `float`
registers. We convert the accumulator to `__half2` and use `__hfma2` instead,
replacing throttled FFMA with unthrottled HFMA2 across `vec_dot_q4_K` and
`vec_dot_q5_K`:

```cuda
// before: float sums[2] = {0.f, 0.f};
//         sums[0] += scale * dot;

__half2 sums = __float2half2_rn(0.f);
sums = __hfma2(__float2half2_rn(scale), __float2half2_rn(dot), sums);
```

### Fix — Flash Attention DKQ=512 (`fattn.cu`)

Gemma 4's MTP assistant head uses `head_dim=512`. The MMA flash-attention path
aborts for DKQ > 256; the `case 512` branch fell through to that path. We route
it directly to the tile kernel, which supports D=512 natively:

```cpp
case 512:
    if (V->ne[0] != K->ne[0]) return BEST_FATTN_KERNEL_NONE;
    // MMA aborts for DKQ>256; tile kernel supports D=512
    return BEST_FATTN_KERNEL_TILE;
```

---

## Results

All benchmarks on CMP 90HX (sm_86, 9877 MiB), `-fa 1`, `tg50`, `-r 5`.

### Patch impact by model

| Model | Quant | GPU% | Clean tok/s | Patched tok/s | Speedup |
|---|---|---:|---:|---:|---:|
| **Qwen3.5-9B** (dense, 5.8 GiB) | Q4_K_XL | 100% | 30.40 | **56.92** | **+87%** |
| **gemma4 E4B** (dense, 6.2 GiB) | Q5_K | 100% | 42.27 | **66.57** | **+57%** |
| **Qwen3.6-35B-A3B** (MoE, ncmoe=26) | Q4_K_M | ~44% | 28.83 | **32.15** | **+11.5%** |

Speedup scales with the fraction of weights on GPU: patches only fire on
GPU-resident tensors. The 35B MoE model at `ncmoe=26` has ~56% of its expert
weights on CPU, so the effective gain is proportionally smaller.

The 35B Q4_K_M figure includes both IMAD+HFMA2 (Q4/Q5) and the Tier 1 Q6_K/Q2_K
HFMA2 patches — the Q4_K_M mix uses Q6_K for attention layers that remain
GPU-resident even when expert FFN layers are offloaded to CPU.

### TurboQuant KV cache — avoid on CMP 90HX

On normal GPUs, KV quantization trades compute for memory bandwidth. On CMP 90HX
the GPU is compute-bound (FFMA throttled 14×), so KV dequant in Flash Attention
costs more than the bandwidth it saves:

| Model | f16 tok/s | turbo3 tok/s | Delta |
|---|---:|---:|---:|
| gemma4 E4B | 66.15 | 61.68 | **−6.8%** |
| Qwen3.5-9B | 56.92 | 55.08 | **−3.2%** |
| Qwen3.6-35B (ncmoe=26) | 30.85 | 30.38 | **−1.5%** |

**Always use `-ctk f16 -ctv f16` on CMP 90HX.**

### NextN speculative decoding

| Model | Base tok/s | NextN tok/s | Accept rate |
|---|---:|---:|---:|
| Qwen3.5-9B dense | 56.9 | 56.3 | 86% — no gain |
| Qwen3.6-35B MoE (ncmoe=28) | 29.1 | **31.3** | **88% — +7.6%** |

Dense models are draft-compute-bound (draft ≈ verify cost) — NextN adds no
throughput. MoE models with heavy verify steps benefit modestly.

---

## Qwen3.6-35B-A3B — ncmoe Guide

The 35B MoE model (20–22 GiB) doesn't fit in 9877 MiB without CPU expert offload.
`-ncmoe N` keeps the first N layers' expert weights on CPU.

| Use case | ncmoe | tok/s | Notes |
|---|---:|---:|---|
| llama-bench | 24 | 30.81 | MTP GGUF; fits bench, not server |
| Server (no NextN) | 26 | ~29.5 | non-MTP; optimal throughput |
| Server + NextN | 28 | 31.3 | MTP GGUF; compute buffers need headroom |

---

## Quick-start

```bash
# Build (sm_86 only)
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j$(nproc) --target llama-bench llama-server

# Qwen3.5-9B — full GPU, best speedup
CUDA_VISIBLE_DEVICES=<cmp90hx_pci> llama-server \
  -m Qwen3.5-9B-UD-Q4_K_XL.gguf \
  -ngl 999 -fa 1 -ctk f16 -ctv f16

# Qwen3.6-35B — MoE with NextN
CUDA_VISIBLE_DEVICES=<cmp90hx_pci> llama-server \
  -m Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf \
  -md Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf \
  --spec-type nextn --draft-max 2 --draft-min 1 \
  -ngl 999 -ncmoe 28 -fa 1 -ctk f16 -ctv f16 -c 2048
```

---

## Benchmark files

| File | Contents |
|---|---|
| [GEMMA_E4B_BENCH.md](GEMMA_E4B_BENCH.md) | gemma4 E4B Q5_K-Medium — patch comparison + TurboQuant KV |
| [GEMMA_E4B_Q6K_BENCH.md](GEMMA_E4B_Q6K_BENCH.md) | gemma4 E4B Q6_K — Tier 1 Q6K patch isolated benchmark |
| [TURBOQUANT_BENCH.md](TURBOQUANT_BENCH.md) | TurboQuant KV sweep (f16 / q8_0 / turbo3) |
| [QWEN35_9B_BENCH.md](QWEN35_9B_BENCH.md) | Qwen3.5-9B — patches, TurboQuant, NextN |
| [QWEN36_35B_BENCH.md](QWEN36_35B_BENCH.md) | Qwen3.6-35B — ncmoe scan, patches, NextN |
| [NEXTN_BENCH.md](NEXTN_BENCH.md) | NextN speculative decoding deep-dive |
