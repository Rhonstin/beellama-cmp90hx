# TurboQuant KV Cache Benchmark — CMP 90HX

**Date**: 2026-05-15  
**GPU**: NVIDIA CMP 90HX (GA102, sm_86, 9877 MiB)  
**Model**: gemma-4-E4B-it-UD-Q4_K_XL.gguf (gemma4 E4B, Q5_K-Medium, 6.18 GiB, 7.52B params)  
**Flags**: `-ngl 999 -fa 1 -p 0 -n 50 -r 3`  
**Branch**: `cmp90hx-optimizations` @ `a5df16302` (atomic/feature/turboquant-kv-cache + IMAD + HFMA2)

## Results

| Scenario    | type_k | type_v | tg50 (tok/s)    | vs baseline |
|-------------|--------|--------|-----------------|-------------|
| No KV quant | f16    | f16    | **66.15 ± 0.24** | —           |
| KV q8_0     | q8_0   | q8_0   | 64.98 ± 0.23    | −1.7%       |
| KV turbo3   | turbo3 | turbo3 | 61.68 ± 0.16    | −6.8%       |

## Analysis

TurboQuant KV is **slower** on CMP 90HX — the opposite of what it achieves on normal GPUs.

**Why**: On a standard GPU, KV quantization trades compute for memory bandwidth (the bottleneck).
On CMP 90HX, FFMA/DP4A are throttled 14–29×, so dequantization during KV attention
lookups becomes expensive. Our IMAD+HFMA2 patches help the MMVQ matmul kernels, but the
KV dequant path in Flash Attention still uses throttled FP32 FFMA, making the overhead net-negative.

**Implication**: For CMP 90HX, keep KV cache in f16 (default). Do not use KV quantization.
TurboQuant KV is designed for bandwidth-bound inference; CMP 90HX is compute-bound (throttled).

## Recommendation

```
-ctk f16 -ctv f16   (default — best on CMP 90HX)
```
