# 02 — Precision Tiers

Demonstrates barraCuda's 15-tier precision continuum with the three
compile-path tiers (F32, F64, DF64) that represent the primary runtime
distinction on current GPU hardware.

## What It Shows

- Identical mathematical operation computed at three compile-path levels
- Automatic precision routing based on GPU hardware capabilities via `PrecisionBrain`
- Error analysis: relative error vs CPU f64 reference
- DF64 (double-float): ~48-bit mantissa on consumer GPUs with 1:64 f64 rate
- Side-by-side comparison table

## The 15-Tier Precision Continuum

| Tier | Mantissa | Use Case | Hardware |
|------|----------|----------|----------|
| Binary | 1-bit | Hashing, masks | All GPUs (f32 core) |
| Int2 | 2-bit | Extreme quantization | All GPUs (f32 core) |
| Q4 | ~3.5-bit | Inference (GGML-class) | All GPUs (f32 core) |
| Q8 | ~7-bit | Training quantization | All GPUs (f32 core) |
| FP8 E5M2 | 2-bit | Gradient exchange | All GPUs (f32 core) |
| FP8 E4M3 | 3-bit | Inference activations | All GPUs (f32 core) |
| BF16 | 7-bit | Training (brain float) | All GPUs (f32 core) |
| F16 | 10-bit | Inference, mobile | Native f16 or f32 core |
| TF32 | 10-bit | Tensor core training | Tensor cores (internal) |
| F32 | 23-bit | Universal baseline | All GPUs |
| DF64 | ~48-bit | Scientific bulk math | Consumer GPUs (f32 pairs) |
| F64 | 52-bit | Reference precision | Compute GPUs (V100, MI250) |
| F64Precise | 52-bit | FMA-separate precision | Compute GPUs |
| QF128 | ~96-bit | Extended (no f64 needed) | All GPUs (f32 quad-double) |
| DF128 | ~104-bit | Maximum precision | Compute GPUs (f64 double-double) |

This demo exercises the three primary compile paths (F32/DF64/F64). The full
15-tier routing is defined in `PrecisionTier`, with coralReef handling
compilation at each tier level.

## Run

```bash
cargo run --release
```
