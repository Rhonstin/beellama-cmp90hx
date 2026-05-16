# Qwen3.5-9B Benchmark — CMP 90HX

**Date**: 2026-05-15  
**GPU**: NVIDIA CMP 90HX (GA102, sm_86, 9877 MiB)  
**Model**: Qwen3.5-9B-UD-Q4_K_XL.gguf (qwen35, 5.70 GiB, 9.20B params)  
**Source**: unsloth/Qwen3.5-9B-MTP-GGUF (contains NextN head, nextn_predict_layers=1)  
**Flags**: `-ngl 999 -fa 1 -p 0 -n 50 -r 5` (llama-bench) / `-c 4096 -np 1` (server)  
**Branch**: `cmp90hx-optimizations` @ `a5df16302` (IMAD + HFMA2 patches)

## llama-bench Results

| Build | KV | tok/s | vs clean |
|---|---|---:|---:|
| Clean (`0a635dcd9`, no patches) | f16 | 30.40 ± 0.02 | — |
| **Patched (IMAD+HFMA2)** | **f16** | **56.92 ± 0.07** | **+87.2%** |
| Patched (IMAD+HFMA2) | turbo3 | 55.08 ± 0.06 | +81.2% |

## NextN Speculative Decoding (server, -c 4096)

| Config | KV | tok/s | accept rate | vs patched-base |
|---|---|---:|---:|---:|
| Patched, no NextN | f16 | ~56.9 | — | — |
| Patched + NextN | f16 | 56.3 ± 0.2 | 86.3% | ≈ 0% |
| Patched + NextN | turbo3 | 55.7 ± 0.2 | 84.9% | ≈ 0% |

## Analysis

### Patch speedup: +87.2%

Largest gain observed across all tested models. Qwen3.5 dense architecture uses
Q4_K for all FFN weights (`ffn_gate`, `ffn_up`, `ffn_down`) with no MoE CPU offload —
100% of weights are GPU-resident, so IMAD and HFMA2 patches apply to every decode step.

Comparison across models (all patched, CMP 90HX):

| Model | GPU% | Patch speedup |
|---|---:|---:|
| Qwen3.5-9B Q4_K (100% GPU) | 100% | **+87.2%** |
| gemma E4B Q5_K (100% GPU) | 100% | +57.5% |
| Qwen3.6-35B A3B Q4_K (ncmoe=26, ~44% GPU) | ~44% | +7.0% |

### NextN: neutral for dense 9B

Draft cost ≈ verify cost: the NextN block is a full transformer block of the same
architecture; on a 9B dense model `t_draft ≈ t_verify`, so async overlap is minimal.
86% acceptance rate is high but cannot overcome the draft compute overhead.
Consistent with NEXTN.md §7: "Dense models are draft-compute-bound."

NextN is worth enabling only for larger MoE targets where verify is heavy enough
to fully overlap the draft (e.g. Qwen3.6-35B-A3B at ncmoe=28: +7.6%).

### TurboQuant KV: -3% on 9B

-3.2% (f16→turbo3) is consistent with other fully-GPU models on CMP 90HX.
Dequant in flash attention uses throttled FP32 FFMA; bandwidth savings don't
compensate on a compute-bound GPU. Use f16 KV.

## Recommendation

```bash
# Optimal config for Qwen3.5-9B on CMP 90HX:
llama-server \
  -m Qwen3.5-9B-UD-Q4_K_XL.gguf \
  -ngl 999 -fa 1 -ctk f16 -ctv f16 \
  -c 4096
```

NextN adds no throughput benefit for this model on CMP 90HX — omit `--spec-type nextn`.
