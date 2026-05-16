# Qwen3.6-35B-A3B Benchmark — CMP 90HX

**Date**: 2026-05-15  
**GPU**: NVIDIA CMP 90HX (GA102, sm_86, 9877 MiB)  
**Models**:
- `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (qwen35moe, 21.10 GiB, 35.51B) — patch comparison
- `Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf` (qwen35moe, 20.65 GiB, 35.51B) — ncmoe scan + NextN  
**Branch**: `cmp90hx-optimizations` @ `a5df16302`

## ncmoe Scan — Q4_K_M (non-MTP)

`-ngl 999 -fa 1 -p 0 -n 10 -r 1`

| ncmoe | GPU expert layers | tok/s |
|---:|---:|---:|
| 40 | 0 | 22.01 |
| 36 | 4 | 23.90 |
| 32 | 8 | 25.79 |
| 28 | 12 | 27.40 |
| **26** | **14** | **29.48** ← optimal |
| 24 | 16 | OOM (9877 MiB exceeded) |

## ncmoe Scan — Q4_K_XL_MTP (with NextN head, block_count=41)

`-ngl 999 -fa 1 -p 0 -n 10 -r 1`

| ncmoe | tok/s | note |
|---:|---:|---|
| 40 | 11.75 | NextN head on GPU, all MoE experts on CPU |
| 36 | 22.17 | |
| 32 | 17.67 | VRAM pressure, throughput dip |
| 28 | 29.18 | |
| 26 | 29.89 | |
| 25 | 30.77 | |
| **24** | **30.81** ← optimal (llama-bench) | |
| 22 | OOM | |

> Note: ncmoe=24 fits for `llama-bench` but not for `llama-server` (server compute
> buffers require ~998 MiB extra VRAM). Use ncmoe=28 for server deployments.

## Patch Comparison — Q4_K_M, ncmoe=26

`-ngl 999 -ncmoe 26 -fa 1 -p 0 -n 50 -r 3` (r=5 for Tier 1)

| Build | KV | tok/s | vs clean |
|---|---|---:|---:|
| Clean (`0a635dcd9`) | f16 | 28.83 ± 0.11 | — |
| Patched (IMAD+HFMA2) | f16 | 30.85 ± 0.22 | +7.0% |
| **Patched + Q6_K/Q2_K HFMA2** | **f16** | **32.15 ± 0.57** | **+11.5%** |
| Patched | turbo3 | 30.38 ± 0.21 | +5.4% |

## NextN Speculative Decoding — Q4_K_XL_MTP, ncmoe=28 (server)

See `NEXTN_BENCH.md` for full analysis. Summary:

| Config | KV | tok/s | accept rate |
|---|---|---:|---:|
| Base (no NextN) | f16 | 29.1 ± 0.4 | — |
| **NextN** | **f16** | **31.3 ± 0.2** | **87.7%** |
| NextN | turbo3 | 30.4 ± 0.0 | 83.4% |

## Analysis

**+7.0% from patches** — much less than fully-GPU models (+57–87%) because only
~44% of weights (14 of 40 expert layers + attention) are GPU-resident at ncmoe=26.
The remaining 56% (CPU expert layers) are unaffected by IMAD/HFMA2.

Patch speedup scales proportionally with GPU weight fraction:

| GPU% of weights | Patch speedup |
|---:|---:|
| 100% (9B / E4B models) | +57–87% |
| ~44% (35B MoE, ncmoe=26) | +7.0% |

**TurboQuant KV**: -1.5% (f16 → turbo3) for 35B — less harmful than on fully-GPU
models because the GPU is already less saturated (more CPU work per step).
Still not recommended; f16 KV is optimal.

**NextN +7.6%**: modest win from speculative decoding. MoE verify is heavy enough
to partially overlap the draft, unlike dense models where draft ≈ verify cost.

## Recommendation

```bash
# llama-bench / offline evaluation:
llama-bench -m Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -ngl 999 -ncmoe 26 -fa 1

# Server (no NextN):
llama-server -m Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf \
  -ngl 999 -ncmoe 28 -fa 1 -ctk f16 -ctv f16 -c 2048

# Server + NextN (+7.6%):
llama-server \
  -m Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf \
  -md Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf \
  --spec-type nextn --draft-max 2 --draft-min 1 \
  -ngl 999 -ncmoe 28 -fa 1 -ctk f16 -ctv f16 -c 2048
```
