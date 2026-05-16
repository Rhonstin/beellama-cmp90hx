# Future Patch Opportunities — CMP 90HX

Analysis of remaining throttled paths after current patches (IMAD + HFMA2 + fattn fix).
Ranked by estimated impact and implementation difficulty.

## Current patch coverage

| Path | Status |
|---|---|
| DP4A → IMAD, decode (MMVQ) | ✅ done (`common.cuh`) |
| DP4A → IMAD, prefill (MMQ) | ✅ done (same `ggml_cuda_dp4a` macro) |
| FP32 dequant → HFMA2, Q4_K decode | ✅ done (`vecdotq.cuh`) |
| FP32 dequant → HFMA2, Q5_K decode | ✅ done (`vecdotq.cuh`) |
| Flash attention DKQ=512 crash | ✅ fixed (`fattn.cu`) |
| FP32 dequant → HFMA2, Q6_K decode | ✅ done (`vecdotq.cuh`) |
| FP32 dequant → HFMA2, Q2_K decode | ✅ done (`vecdotq.cuh`) |

**Tier 1 result**: +4.2% on Qwen3.6-35B Q4_K_M (ncmoe=26): 30.85 → 32.15 tok/s.
Negligible on Q4_K_XL or Q5_K-heavy models (those layers already patched).

---

## Tier 1 — Easy (same pattern as existing patches, ~30–60 min each) ✅ DONE

### 1a. Q6_K decode HFMA2 — `vec_dot_q6_K_q8_1_impl_mmvq` ✅

**File**: `ggml/src/ggml-cuda/vecdotq.cuh`  
**Implementation notes**: QR6_K=2 — manually unrolled both iterations into two half2 lanes.
Includes `d` (block scale) in the `h2_coeff = d * d8 * sc` product to keep fp16 intermediates
in range (mirrors dm4/dm5 in Q4K/Q5K). Returns `r.x + r.y` — `d` is already accumulated.

### 1b. Q2_K decode HFMA2 — `vec_dot_q2_K_q8_1_impl_mmvq` ✅

**File**: `ggml/src/ggml-cuda/vecdotq.cuh`  
Followed exact Q4K/Q5K pattern: `sumf_d` → lane 0, `sumf_m` → lane 1.
`sc_m` (min scale) is already baked into `dot_m = dp4a(m_broadcast, u, 0)`, so
lane 1 coefficient is `dm2.y * 1 * d8[i]` (no extra scale factor for lane 1).

---

## Tier 2 — Medium complexity, high impact (2–8 hours)

> **Status**: Implemented and benchmarked — no measurable gain at tg50 (50-token context).
> Flash attention KQ accumulation is <1% of decode compute at short contexts, so replacing
> 2 throttled FP32 FADDs with 1 HFMA2 cannot move tok/s. Both 2a and 2b reverted.
> Worth revisiting for long-context inference (pp or tg with 2k+ tokens in KV cache).

### 2a. `ggml_cuda_mad(float, half2, half2)` — eliminate final FP32 FFMA

**File**: `ggml/src/ggml-cuda/common.cuh` ~line 745  
**Impact**: flash attention KQ dot product when KV cache is f16

When K and Q are both `half2`, the current implementation does:
```cpp
// v*u → HMUL2 (free), then:
const float2 tmp = __half22float2(v*u);
acc += tmp.x + tmp.y;  // ← FP32 FFMA (throttled!)
```

Patch: use `__hadd` to sum the two half lanes before converting to float:
```cpp
#if __CUDA_ARCH__ == 860
    acc += __half2float(__hadd(__low2half(v * u), __high2half(v * u)));
    // HMUL2 (free) → HADD2 (free) → one float conversion → one FP32 add
#else
    // original
#endif
```
Net: 2 FP32 FMAs → 1 HMUL2 + 1 HADD2 + 1 float add. Eliminates one throttled FFMA per KQ pair.

### 2b. Flash attention KQ accumulator: `float[]` → `half2[]`

**File**: `ggml/src/ggml-cuda/fattn-tile.cuh` ~line 591  
**Impact**: removes all throttled FP32 from attention score computation;
**would make turbo3 KV net-positive on CMP 90HX**

```cpp
// Current:
float KQ_acc[nbatch_fa/(np*warp_size) * cpw] = {0.0f};

// Patch:
#if __CUDA_ARCH__ == 860
    half2 KQ_acc[...] = {__float2half2_rn(0.0f)};
    // ggml_cuda_mad(half2 &, half2, half2) already exists — uses pure HFMA2
#endif
```

**Risk**: fp16 range ≈ ±65504; pre-softmax attention scores may overflow for long sequences
or large head_dim. Requires careful validation. Softmax KQ_max normalization helps but
may not be sufficient for all cases.

**If this works**: turbo3 KV dequant overhead (which is also FP32) becomes the next
bottleneck — patch 2c would follow.

### 2c. Turbo3 KV dequant → FP16 (follow-up to 2b)

**File**: `ggml/src/ggml-cuda/fattn-tile.cuh`, load_tile functions  
Currently dequantizes KV blocks to `float` before attention. If KQ_acc is half2 (2b),
dequanting to `half2` directly removes the float intermediate entirely.

---

## Tier 3 — Complex / research-level (days to weeks)

### 3a. Prefill GEMM: cuBLAS SGEMM → custom HFMA2 tiled GEMM

**Impact**: prefill (pp) speed — currently near-zero performance
**Why it's broken**: cuBLAS SGEMM uses FP32 FFMA (throttled 14×); tensor cores throttled ~200×.
A custom tiled GEMM in shared memory using HFMA2 would be ~12× faster on compute.

```
Standard approach:
  - 64×64 output tiles, 8×8 thread blocks
  - Load A/B tiles into __shared__ as half2
  - Inner loop: __hfma2 accumulation
  - Write back converting to float
```

This is the largest remaining opportunity (prefill is currently catastrophically slow)
but requires significant kernel engineering.

### 3b. RMSNorm variance accumulation → half2

**File**: `ggml/src/ggml-cuda/norm.cu`  
Low priority for decode (single token = cheap), but matters for prefill.
Risk: squaring small values in fp16 loses precision.

### 3c. ROPE application → half2

**File**: `ggml/src/ggml-cuda/rope.cu`  
`cosf`/`sinf` computed in FP32 (necessary, special functions), but the rotation
application `x_new = x*cos - x_rot*sin` could use HFMA2 if activations are in fp16.
Medium impact, low risk.

---

## Recommended next steps

1. ~~**Q6_K HFMA2** (1a)~~ — ✅ done
2. ~~**`ggml_cuda_mad` patch** (2a)~~ — attempted, negligible at short context
3. ~~**KQ_acc float→half2** (2b)~~ — attempted, -1.3% regression at tg50; <1% of compute
4. **Prefill GEMM** (3a) — largest remaining opportunity; CMP 90HX prefill is currently catastrophically slow
5. **ROPE half2** (3c) — straightforward, medium impact on prefill

For decode, Tier 1 patches captured the bulk of the available gains. Further decode improvements
require either long-context workloads (where Tier 2 KQ patches would matter) or new quantization
types with untouched FP32 paths.
