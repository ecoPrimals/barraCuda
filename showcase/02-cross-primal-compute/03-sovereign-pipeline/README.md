# 03 — Sovereign Pipeline

The capstone demo: demonstrates the full sovereign compute pipeline from
hardware discovery through shader compilation to GPU dispatch and validation.

## What It Shows

The complete compute triangle operating as a pipeline:

```
  toadStool (hardware)
       |
       | GPU capabilities, precision profile
       v
  barraCuda (math)
       |
       | WGSL shader + arch target
       v
  coralReef (compiler)
       |
       | native binary (SASS, GFX)
       v
  barraCuda (dispatch)
       |
       | GPU execution + readback
       v
  Validated result
```

## Graceful Degradation

Each layer falls back independently:

| Layer | Available | Fallback |
|-------|-----------|----------|
| toadStool | Rich hardware profile | barraCuda local wgpu discovery |
| coralReef | Native binary compilation | wgpu SPIR-V path |
| GPU | Hardware dispatch | llvmpipe software rendering |

## Run

```bash
cargo run --release
```
