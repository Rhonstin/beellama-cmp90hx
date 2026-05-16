# NextN Speculative Decoding — CMP 90HX

**Date**: 2026-05-15  
**GPU**: NVIDIA CMP 90HX (GA102, sm_86, 9877 MiB)  
**Model**: Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf (qwen35moe, 20.65 GiB, 35.51B params)  
**Branch**: `cmp90hx-optimizations` @ `a5df16302` (IMAD + HFMA2 patches applied)  
**Flags**: `-ngl 999 -ncmoe 28 -fa 1 -c 2048 -np 1`  
**NextN**: `--spec-type nextn -md <same_file> --draft-max 2 --draft-min 1` (shared-model path)

## Results

| Scenario | KV | tok/s | vs f16-base | accept rate |
|---|---|---:|---:|---:|
| Base (no NextN) | f16 | **29.1 ± 0.4** | — | — |
| **NextN** | **f16** | **31.3 ± 0.2** | **+7.6%** | **87.7%** |
| NextN | turbo3 | 30.4 ± 0.0 | +4.5% | 83.4% |

## VRAM budget at ncmoe=28

| Component | Size |
|---|---|
| Model weights (GPU) | 7745.85 MiB |
| KV cache — target (f16, c=2048) | 40.00 MiB |
| Compute buffer — target | 501.00 MiB |
| KV cache — NextN draft (1 layer) | 4.00 MiB |
| Compute buffer — NextN draft | 497.00 MiB |
| **Total** | **~8788 MiB / 9877 MiB** |

**ncmoe=24 is too tight for the server** (compute buffers push over 9877 MiB).  
Use `ncmoe=28` for NextN server; `ncmoe=24` only works for bare `llama-bench` (no compute buffers).

## Analysis

NextN gives a **modest but consistent +7.6%** on CMP 90HX vs the **+35% on M4 Max** (NEXTN.md §7).

**Why the gap:**
- On M4 Max, all 40 expert layers are on GPU → verify is fast, draft fully overlaps
- On CMP 90HX at ncmoe=28, only 12 expert layers are on GPU → verify is slower (more CPU work), so the draft has less room to hide its compute cost
- CMP 90HX throttles FFMA/DP4A 14–29×; draft computation (NextN block) also uses these throttled paths for non-Q4_K tensors
- Our IMAD + HFMA2 patches help the NextN block's Q4_K attention weights, partially offsetting the throttle

**TurboQuant with NextN:**  
turbo3 KV reduces acceptance (83.4% vs 87.7%) — compressed KV introduces attention noise that degrades draft quality. The bandwidth savings don't compensate on CMP 90HX (compute-bound, not bandwidth-bound). Use f16 KV.

## Recommendation

```
# Optimal NextN config for CMP 90HX:
llama-server \
  -m Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf \
  -md Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf \
  --spec-type nextn --draft-max 2 --draft-min 1 \
  -ngl 999 -ncmoe 28 -fa 1 -ctk f16 -ctv f16 \
  -c 2048
```

## Cross-model summary (IMAD+HFMA2 patched, CMP 90HX)

| Model | Mode | tok/s | vs clean baseline |
|---|---|---:|---:|
| gemma E4B Q5_K (100% GPU) | base | 66.57 | +57.5% vs clean |
| Qwen3.6-35B Q4_K ncmoe=26 | base | 30.85 | +7.0% vs clean |
| Qwen3.6-35B Q4_K ncmoe=28 | base (server) | 29.1 | — |
| Qwen3.6-35B Q4_K ncmoe=28 | **NextN** | **31.3** | **+7.6% vs base** |
