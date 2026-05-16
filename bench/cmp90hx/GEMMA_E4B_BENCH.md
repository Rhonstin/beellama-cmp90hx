# Gemma 4 E4B Benchmark — CMP 90HX

**Date**: 2026-05-15  
**GPU**: NVIDIA CMP 90HX (GA102, sm_86, 9877 MiB)  
**Model**: gemma-4-E4B-it-UD-Q4_K_XL.gguf (gemma4 E4B, Q5_K-Medium, 6.18 GiB, 7.52B params)  
**Flags**: `-ngl 999 -fa 1 -p 0 -n 50 -r 5`  
**Branch**: `cmp90hx-optimizations` @ `a5df16302`

## Patch Comparison (IMAD + HFMA2 vs clean)

| Build | KV | tok/s | vs clean |
|---|---|---:|---:|
| Clean (`0a635dcd9`, no patches) | f16 | 42.27 ± 0.05 | — |
| **Patched (IMAD+HFMA2)** | **f16** | **66.57 ± 0.18** | **+57.5%** |

## TurboQuant KV Comparison (patched build, -r 3)

| type_k | type_v | tok/s | vs f16 |
|---|---|---:|---:|
| f16 | f16 | 66.15 ± 0.24 | — |
| q8_0 | q8_0 | 64.98 ± 0.23 | −1.7% |
| turbo3 | turbo3 | 61.68 ± 0.16 | −6.8% |

## Analysis

**+57.5% from IMAD+HFMA2 patches.** Gemma E4B is fully GPU-resident (6.18 GiB < 9877 MiB),
so patches apply to 100% of decode compute.

Gemma uses Q5_K for FFN and attention weights. The HFMA2 patch
(`vec_dot_q5_K_q8_1_impl_vmmq`) replaces FP32 accumulation with `__half2`/`__hfma2`,
eliminating throttled FFMA (17.8 ns/op) in favour of unthrottled HFMA2 (1.4 ns/op).

**TurboQuant KV is net-negative** (see TURBOQUANT_BENCH.md for full analysis).
The KV dequant path in Flash Attention uses FP32 FFMA — throttled on CMP 90HX.
Bandwidth savings do not compensate for the compute overhead.

## flash attention note

Gemma 4 uses head_dim=256 for the main model → MMA flash attention path (sm_86
supports DKQ≤256 via MMA). The MTP assistant head uses head_dim=512, which
previously triggered an abort in `ggml_cuda_get_best_fattn_kernel`; fixed in
commit `e7b5d6bed` (routes DKQ=512 to tile kernel).

## Recommendation

```bash
llama-server \
  -m gemma-4-E4B-it-UD-Q4_K_XL.gguf \
  -ngl 999 -fa 1 -ctk f16 -ctv f16
```
