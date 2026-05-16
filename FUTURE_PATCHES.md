# CMP 90HX Optimization Roadmap

**Fork**: beellama.cpp + CMP 90HX patches  
**Last updated**: 2026-05-16 (night — Phase 3 investigated and abandoned; see findings)  
**VRAM**: CMP 90HX має 10 GB VRAM. Моделі >~9B Q4_K потребують перевірки на вміщення.  
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

## Phase 1b — Baseline benchmarks на beellama.cpp (TODO)

Перевірити всі моделі які тестувалися в llama.cpp fork, щоб отримати порівняльну базу.  
**Пропустити**: Qwen3.6-27B-UDT-Q3_K_XL_MTP.gguf — не вміщується у 10 GB VRAM.

| Модель | Файл | Раніше (llama.cpp) | Статус |
|---|---|---|---|
| ~~Qwen3.5-9B UD-Q4_K_XL~~ | ~~`Qwen3.5-9B-UD-Q4_K_XL.gguf`~~ | ~95 tok/s на llama.cpp | ❌ несумісний з beellama.cpp (гібридна SSM/Mamba архітектура, не підтримується) |
| Gemma-4 E4B Q4_K_XL | `gemma-4-E4B-it-UD-Q4_K_XL.gguf` | ~66.9 tok/s (patched) | ⬜ не тестовано |
| Gemma-4 E4B Q6_K | `gemma-4-E4B-it-Q6_K.gguf` | тестовано ізольовано | ⬜ не тестовано |
| Gemma-4 E4B Q4_K_M | `gemma-4-E4B-it-assistant.Q4_K_M.gguf` | — | ⬜ не тестовано |
| ~~Qwen3.6-35B-A3B Q4_K_M~~ | ~~`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`~~ | — | ❌ виключено (MoE, див. нижче) |
| ~~Qwen3.6-35B-A3B Q4_K_XL MTP~~ | ~~`Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf`~~ | — | ❌ виключено (MoE, див. нижче) |

**Виключені моделі**:
- `Qwen3.6-27B-*` — не вміщується у 10 GB VRAM
- `Qwen3.6-35B-A3B-*` — MoE: лише ~27% активних параметрів потрапляє на GPU, решта CPU-offload через PCIe 1 Гб/с. Пропускна здатність стає вузьким місцем раніше ніж патч встигає дати виграш. Залишити для тестування на A2000 (більший VRAM + кращий bandwidth).

**Команда для бенчмарку**:
```bash
/home/rhonstin/beellama-cmp90hx/build/bin/llama-bench \
  -m /home/rhonstin/models/<MODEL>.gguf \
  -ngl 99 -fa 1 -p 0 -n 50 2>&1 | grep "tok/s"
```

Результати зберігати у `bench/cmp90hx/BEELLAMA_BASELINE_BENCH.md`.

---

## Phase 2 — Beellama-specific integration (TODO)

These tasks are specific to beellama.cpp features not present in upstream llama.cpp.

### 2a. Benchmark beellama.cpp features on CMP 90HX

- [ ] Run DFlash speculative decoding baseline
  - `gemma-4-E4B-it-assistant.Q4_K_M.gguf` (166MB) — це **MTP head**, НЕ DFlash drafter
    - архітектура `gemma4_assistant` не підтримується beellama.cpp (тільки llama.cpp fork)
  - Для DFlash потрібна окрема натренована DFlash-модель для Gemma E4B (поки не знайдено публічно)
  - File results in `bench/cmp90hx/BEELLAMA_DFLASH_BENCH.md`

- [ ] Портувати `gemma4_assistant` архітектуру з llama.cpp → beellama.cpp
  - Дозволить використовувати MTP head в контексті beellama.cpp features
  - Джерело: `llama.cpp/src/llama-arch.cpp`, `llama-model.cpp`, `llama-context.h`
  - Commit у llama.cpp fork: `72f60cf85 feat: add Gemma 4 MTP assistant support`

- [x] MTP speculative decoding benchmark — llama.cpp fork ✅ DONE (2026-05-16)
  - **Результат**: baseline=65.32 tok/s, MTP=41.64 tok/s → **-36.3%** (net-negative)
  - Accept rate: 72.8% — accept rate відмінний, але throughput гірший
  - **Причина**: MTP verification = mini-prefill → cuBLAS SGEMM → FP32 FFMA → throttled 14×
    Кожен крок верифікації платить GEMM penalty, що перекриває виграш від паралельного decode
  - **Рішення**: MTP стане net-positive після Phase 3 (HFMA2 tiled GEMM для prefill/GEMM)
  - Деталі: `bench/cmp90hx/MTP_BENCH.md`

- [x] Benchmark TurboQuant KV types (`turbo4`, `turbo3`, `turbo2`) on CMP 90HX ✅ DONE
  - **Результат**: всі типи net-negative (f16 найкраще). turbo2=-16%, turbo3=-19.6%, q8_0=-25%
  - Причина: KV dequant використовує FP32 FFMA (throttled 14×), що переважає bandwidth savings
  - Рішення: пропатчити TurboQuant dequant → HFMA2 (Phase 2c)

### 2b. DFlash hidden-state accumulation HFMA2

**File**: `src/llama-speculative.cpp` or DFlash CUDA kernel  
Check if DFlash draft cross-attention uses FP32 accumulation that could benefit from HFMA2.

### 2c. TurboQuant dequant → HFMA2 ❌ ДОСЛІДЖЕННЯ ЗАВЕРШЕНО — PATCH НЕМОЖЛИВИЙ

**Status**: Investigated 2026-05-16. Abandoned.

#### Спроба: V_DOT2 (flash attention VKQ accumulation → half2)

Увімкнули `V_DOT2_F32_F16_AVAILABLE` для CUDA sm_86 у `common.cuh`, щоб змусити
`fattn-vec.cuh` використовувати `half2 VKQ[...]` замість `float2 VKQ[...]`.

**Результат**: нуль ефекту. Жодного поліпшення ні при 50-токенному, ні при 2048-токенному контексті.

#### Причина: реальний bottleneck — FWHT, не flash attention

Тест при різних контекстах показав: уповільнення TurboQuant (+19–25%) **однакове** при 50 і 2048
токенах. Flash attention dequant масштабується з контекстом — якщо б він був bottleneck,
уповільнення зростало б з контекстом. Висновок: bottleneck **контексто-незалежний**.

Справжній винуватець — `turbo_fwht_128_cuda` у `turbo-quant-cuda.cuh`:
- 7-стадійний Fast Walsh-Hadamard Transform: 896 FP32 FADDs + 128 FMULs на 128-елементну групу
- Викликається у `k_set_rows_turbo*` (кожен KV write) і `k_turbo*_dequant_f16_inv_fwht` (кожен decode step)
- Всі операції FP32 FFMA — throttled 14× на CMP 90HX
- ~2300 FP32 ops на групу, контексто-незалежний overhead

#### Чому FWHT half2 неможливий

FWHT акумулює похибки через 7 стадій butterfly. FP16 має range ±65504 і ~3 знаки точності.
Після 7 стадій додавань значення переповнюються або точність деградує — це зламає якість квантизації.
PTX inline `HADD`/`HSUB` теж не допоможе без аудиту числових меж по всіх 7 стадіях.

#### Висновок

TurboQuant конструктивно несумісний з CMP 90HX: алгоритм фундаментально вимагає FP32 FWHT,
а FP32 throttled 14× в firmware. Без переписування FWHT під INT-арифметику (що змінить математику)
net-positive результат неможливий. **Рекомендація**: використовувати `f16` KV, не TurboQuant.

---

## Phase 3 — Prefill GEMM ❌ ДОСЛІДЖЕННЯ ЗАВЕРШЕНО — PATCH НЕ ПОТРІБЕН

**Status**: Investigated 2026-05-16. Original assumption was wrong — prefill is already fast.

### Висновки (2026-05-16)

**Вихідна гіпотеза**: "prefill catastrophically slow because cuBLAS SGEMM uses FP32 FFMA (14×)"  
**Реальність**: Q4_K prefill іде через MMQ kernel (tensor cores), NOT cuBLAS SGEMM.

| Path | Batch | Kernel | Instructions | pp512 result |
|---|---|---|---|---|
| Decode | 1 | MMVQ | DP4A → **IMAD** (patched) | 66 tok/s ✅ |
| Prefill | >8 | MMQ | Tensor core HMMA | **503 tok/s** ✅ |

#### Чому tensor cores FAST для prefill, незважаючи на "200× throttle":

Метрика 200× — це **latency одного HMMA instruction** (284.5 ns). Але кожна HMMA інструкція
обчислює матрицю 16×8×16 = **2048 MAC** одночасно. Ефективна TOPS:

```
HMMA (throttled):  2048 MACs / 284.5 ns = 7.2 TOPS/warp — FAST for large GEMM
IMAD (unthrottled): 1 MAC    / 1.4 ns   = 0.7 TOPS/warp — slow for large GEMM
```

Tensor cores "throttled 200×" per instruction, але ~10× faster per MAC для великих GEMMs.

#### Що відбулося при спробі Phase 3:

Замінив tensor core MMA → IMAD/dp4a в MMQ kernel для sm_86 (через `CMP90HX_FORCE_DP4A`):
- **До**: pp512 = 503 tok/s (tensor cores)
- **Після**: pp512 = 195 tok/s (IMAD dp4a)
- **Результат**: -61% — НАБАГАТО ГІРШЕ

**Патч негайно відкинутий** (reverted via `git stash drop`).

#### Де IMAD/HFMA2 ДІЙСНО допомагають (decode):

- MMVQ (batch ≤ 8): пам'ять bandwidth-bound → compute майже "безкоштовний"
  → Перехід з DP4A (35.6 ns) на IMAD (1.4 ns) = +47.6% на decode ✅
- Tensor cores в MMVQ: занадто великий overhead для 1-8 токенів → IMAD краще

#### Можливості для prefill (якщо ПОТРІБНО покращення):

1. **Memory bandwidth**: CMP 90HX вже близька до bandwidth limit при pp512=503 tok/s
   Префікс на 7B-параметрній моделі: ~4GB weights read once → ~4GB/503tok/s*512tok ≈ 4GB/s
   CMP 90HX bandwidth: ~320 GB/s → є запас, але MMQ kernel overhead домінує
   
2. **Реальна проблема**: MMQ kernel overhead (quantize src1, shared mem latency) — не compute

**Висновок**: Phase 3 не потрібна. Prefill вже 503+ tok/s. Фокус на Phase 2c (TurboQuant dequant → HFMA2)
щоб розблокувати TurboQuant як net-positive для довгих контекстів.

### 3b. RMSNorm variance → half2

**File**: `ggml/src/ggml-cuda/norm.cu`  
Залишається low-priority. Norm FP32 FFMA throttled 14×, але норми — невеликий відсоток compute.

### 3c. ROPE application → half2

**File**: `ggml/src/ggml-cuda/rope.cu`  
`cosf`/`sinf` must stay FP32 (special function), але rotation `x*cos - x_rot*sin`
могла б використовувати HFMA2. Medium impact, low risk — розглянути разом з Phase 2c.

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
