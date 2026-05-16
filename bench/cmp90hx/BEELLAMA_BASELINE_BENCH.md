# beellama-cmp90hx — Baseline Benchmarks (CMP 90HX)

**Date**: 2026-05-16  
**GPU**: NVIDIA CMP 90HX (GA102, sm_86, 10 GB VRAM, CUDA device 0)  
**Build**: beellama.cpp + IMAD/HFMA2/fattn patches (commit 1905ce9e5)  
**Excluded**:
- Qwen3.5-9B — гібридна SSM/Mamba архітектура, не підтримується beellama.cpp
- 27B/35B MoE — не вміщуються в 10 GB VRAM або PCIe bandwidth bottleneck
- DFlash — потребує окремої draft-моделі (не завантажено)

---

## 1. Patch baseline — decode (p=0, n=128, fa=1, f16 KV)

| Модель | Розмір | Параметри | tg t/s | vs llama.cpp patched |
|---|---|---|---|---|
| Gemma-4 E4B Q4_K_XL | 6.18 GiB | 7.52B | **66.65 ± 0.00** | ~66.92 ✅ |
| Gemma-4 E4B Q6_K | 6.57 GiB | 7.52B | **62.19 ± 0.23** | — |

Патчі підтверджено — результати збігаються з llama.cpp fork.

---

## 2. TurboQuant KV cache — Gemma E4B Q4_K_XL (p=512, n=128, fa=1, r=3)

| ctk/ctv | tg t/s | vs f16 | Висновок |
|---|---|---|---|
| **f16**  | **66.45 ± 0.08** | baseline | ✅ найкраще |
| turbo2   | 55.79 ± 0.08 | -16.0% | net-negative |
| turbo4   | 53.62 ± 0.00 | -19.3% | net-negative |
| turbo3   | 53.43 ± 0.04 | -19.6% | net-negative |
| q8_0     | 49.60 ± 0.04 | -25.3% | net-negative |

---

## 3. TurboQuant KV cache — Gemma E4B Q6_K (p=512, n=128, fa=1, r=3)

| ctk/ctv | tg t/s | vs f16 | Висновок |
|---|---|---|---|
| **f16**  | **62.25 ± 0.20** | baseline | ✅ найкраще |
| turbo2   | 52.84 ± 0.14 | -15.1% | net-negative |
| turbo4   | 50.74 ± 0.05 | -18.5% | net-negative |
| turbo3   | 50.61 ± 0.04 | -18.7% | net-negative |
| q8_0     | 47.17 ± 0.04 | -24.2% | net-negative |

---

## Аналіз

### Чому TurboQuant net-negative на CMP 90HX?

CMP 90HX throttles FP32 FFMA 14×. KV cache дeквантизація (будь-який тип крім f16)
вимагає FP32 FFMA в dequant ядрі:

```
f16 KV:     load half → use directly (bandwidth bound, no dequant)
turboN KV:  load compressed → FFMA dequant (14× throttle penalty)
```

Bandwidth savings від менших KV блоків не компенсують 14× штраф на dequant compute.
Результат узгоджується з попередніми тестами на llama.cpp fork.

### Потенційне рішення (Phase 2c в FUTURE_PATCHES.md)

Якщо пропатчити TurboQuant dequant ядро → HFMA2 (аналогічно vecdotq патчу),
то dequant стає unthrottled і TurboQuant може стати net-positive на довгих контекстах.

### Висновок

**На CMP 90HX завжди використовувати `f16` KV cache.**  
TurboQuant корисний лише на картах без throttle (A2000E, RTX серія).

---

## DFlash (TODO)

Потребує завантаження DFlash draft-моделі. Candidate:
- `gemma-4-E4B-it-assistant.Q4_K_M.gguf` (166MB) — вбудований assistant head,
  перевірити чи він сумісний як DFlash drafter у beellama.cpp

Результати зберігати у `BEELLAMA_DFLASH_BENCH.md`.
