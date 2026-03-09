# 02 — Precision Tiers

Demonstrates barraCuda's 3-tier precision model: F32, F64, and DF64.

## What It Shows

- Identical mathematical operation computed at three precision levels
- Automatic precision routing based on GPU hardware capabilities
- Error analysis: relative error vs CPU f64 reference
- DF64 (double-float): ~48-bit mantissa on consumer GPUs with 1:64 f64 rate
- Side-by-side comparison table

## The 3-Tier Model

| Tier | Mantissa | Use Case | Hardware |
|------|----------|----------|----------|
| F32 | 23-bit | Graphics, preview | All GPUs |
| DF64 | ~48-bit | Scientific bulk math | Consumer GPUs (RTX 40xx) |
| F64 | 52-bit | Reference precision | Compute GPUs (V100, MI250) |

## Run

```bash
cargo run --release
```
