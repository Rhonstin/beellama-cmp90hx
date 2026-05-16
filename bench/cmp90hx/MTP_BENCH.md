# MTP (Multi-Token Prediction) Benchmark — llama.cpp fork (CMP 90HX)

**Date**: 2026-05-16  
**GPU**: NVIDIA CMP 90HX (GA102, sm_86, 10 GB VRAM, `CUDA_VISIBLE_DEVICES=0`)  
**Build**: `/home/rhonstin/cmp90hx-patch/llama.cpp` — patched (IMAD/HFMA2)  
**Spec type**: `--spec-type mtp --mtp-head <path>`

---

## Setup

| Parameter | Value |
|---|---|
| Main model | `gemma-4-E4B-it-UD-Q4_K_XL.gguf` (6.18 GiB) |
| MTP head | `gemma-4-E4B-it-assistant.Q4_K_M.gguf` (166 MB) |
| MTP head arch | `gemma4_assistant` (supported in llama.cpp fork via commit `72f60cf85`) |
| Context | p=0, n=128, fa=1, f16 KV |
| Reps | 3 |
| Prompt | "Explain the history of artificial intelligence in detail." |

---

## Results

| Mode | tg t/s | Accept rate | vs baseline |
|---|---|---|---|
| **Baseline** | **65.32 ± 0.06** | — | — |
| MTP | 41.64 ± 0.14 | 72.8% | **-36.3%** |

---

## Аналіз

### Чому MTP net-negative на CMP 90HX?

**Висока acceptance rate (72.8%) але -36.3% throughput** — парадокс на перший погляд.

MTP (speculative decoding) має дві фази:
1. **Draft**: швидке генерування N токенів через MTP head (166MB — дешево)
2. **Verify**: пакетна верифікація N+1 токенів основною моделлю

Фаза верифікації = mini-prefill: основна модель обробляє батч з кількох токенів одночасно.  
На CMP 90HX mini-prefill використовує cuBLAS SGEMM → FP32 FFMA → throttled **14×**.

```
Decode (1 токен):   MMVQ kernel → unthrottled IMAD/HFMA2 ✅
Mini-prefill (N+1): GEMM kernel → FP32 FFMA → throttled 14× ❌
```

Навіть при 72.8% acceptance rate, затримка від throttled GEMM при кожному кроці верифікації
перекриває виграш від паралельного прийняття кількох токенів.

### Порівняння з A2000E

На A2000E FFMA не throttled → GEMM верифікація дешева → MTP має бути net-positive.  
Тести на A2000E заплановані окремо.

### Потенційне рішення (Phase 3)

Якщо реалізувати HFMA2 tiled GEMM (Phase 3 в FUTURE_PATCHES.md), то:
- GEMM-based верифікація буде unthrottled
- MTP може стати net-positive навіть на CMP 90HX

---

## Висновок

**На CMP 90HX MTP net-negative через throttled GEMM верифікацію.**  
MTP буде корисним лише після реалізації Phase 3 (custom HFMA2 tiled GEMM).  
До того — завжди використовувати звичайний decode без speculative decoding.
