# Gemma 4 E4B Q6_K Benchmark — CMP 90HX

**Date**: 2026-05-16  
**GPU**: NVIDIA CMP 90HX (GA102, sm_86, 9877 MiB)  
**Model**: gemma-4-E4B-it-Q6_K.gguf (gemma4 E4B, pure Q6_K, 6.57 GiB, 7.52B params)  
**Flags**: `-ngl 999 -fa 1 -p 0 -n 50 -r 5`

## Patch Comparison

| Build | tok/s | vs clean |
|---|---:|---:|
| Clean (no patches, `0a635dcd9`) | 49.03 ± 0.10 | — |
| IMAD only (DP4A→IMAD, no HFMA2) | 61.31 ± 0.12 | +25.1% |
| **All patches (IMAD + Q6K HFMA2)** | **62.10 ± 0.19** | **+26.6%** |

## Analysis

**+26.6% total from all patches.** Lower than Q5_K-Medium (+57.5%) for two reasons:

1. **IMAD dominates**: Q6_K has 1 DP4A per loop iteration (vs 2 for Q5_K, 4 for Q4_K).
   With fewer DP4A calls per element, the DP4A→IMAD replacement is proportionally
   smaller — but still the largest single improvement (+25.1%).

2. **Q6_K kernel is simpler**: Only one FP32 accumulator (`sumf`) vs the d/m dual-accumulator
   structure in Q4_K/Q5_K. After IMAD removes the DP4A bottleneck, the model becomes
   more memory-bandwidth limited and the HFMA2 contribution is smaller (+1.3%).

The Q6_K HFMA2 patch (Tier 1) uses QR6_K=2 — both iterations packed into two half2 lanes.
Block scale `d` is included in `h2_coeff = d * d8 * sc` to keep fp16 intermediates in
range (mirrors the dm4/dm5 approach in Q4K/Q5K). Returns `r.x + r.y` (d is already applied).

## Comparison with Q5_K model

| Model | Quant | Clean tok/s | Patched tok/s | Speedup |
|---|---|---:|---:|---:|
| gemma-4-E4B-it-UD-Q4_K_XL | Q5_K-Medium | 42.27 | 66.57 | +57.5% |
| gemma-4-E4B-it-Q6_K | Q6_K (pure) | 49.03 | 62.10 | +26.6% |

Q5_K-Medium benefits from both Q4_K and Q5_K HFMA2 patches across different layer types,
giving a larger cumulative gain than the pure-Q6_K model.

## Recommendation

```bash
# Q6_K gives better quality than Q5_K but lower throughput — use Q5_K for speed
llama-server \
  -m gemma-4-E4B-it-Q6_K.gguf \
  -ngl 999 -fa 1 -ctk f16 -ctv f16
```
