# GPU Silicon Capability Matrix

**Version**: 1.0.0
**Date**: April 10, 2026
**Status**: Living specification — updated as hardware probes report new data
**Authority**: barraCuda (Layer 1) + toadStool (Layer 0) + coralReef (Layer 0)

---

## 1. Purpose

This document maps GPU silicon capabilities across consumer and HPC generations,
guiding three routing decisions:

1. **Precision routing**: native f64 vs DF64 vs coralReef sovereign lowering
2. **Execution unit routing**: shader cores vs tensor cores vs fixed-function
3. **Polyfill strategy**: what software must replace what hardware cannot do

---

## 2. FP64 Rate by Generation

The ratio of FP64:FP32 throughput determines whether native f64 is viable or
whether DF64/polyfill is the better path.

### NVIDIA

| Generation     | Arch   | SM   | FP64:FP32  | DFMA | Tensor Cores   | Memory    |
|----------------|--------|------|------------|------|----------------|-----------|
| Kepler (HPC)   | GK110  | 3.5  | 1:3        | Yes  | —              | GDDR5     |
| Maxwell        | GM2xx  | 5.x  | 1:32       | Yes  | —              | GDDR5     |
| Pascal (consumer) | GP10x | 6.1 | 1:32      | Yes  | —              | GDDR5X    |
| Pascal (HPC)   | GP100  | 6.0  | 1:2        | Yes  | —              | HBM2      |
| Volta (HPC)    | GV100  | 7.0  | **1:2**    | Yes  | Gen 1 (FP16)   | HBM2      |
| Turing         | TU1xx  | 7.5  | 1:32       | Yes  | Gen 2 (FP16/INT8) | GDDR6  |
| Ampere (HPC)   | GA100  | 8.0  | **1:2**    | Yes  | Gen 3 (FP16/BF16/TF32/INT8) | HBM2e |
| Ampere (consumer) | GA10x | 8.6 | 1:64      | Yes  | Gen 3          | GDDR6X    |
| Ada Lovelace   | AD10x  | 8.9  | 1:64       | Yes  | Gen 4 (+ FP8)  | GDDR6X    |
| Hopper (HPC)   | GH100  | 9.0  | **1:2**    | Yes  | Gen 4 (+ FP8)  | HBM3      |
| Blackwell (HPC)| GB100  | 10.0 | **1:2**    | Yes  | Gen 5 (+ FP4)  | HBM3e     |
| Blackwell Ultra| GB300  | 10.x | 1:64(!)    | Yes  | Gen 5 (AI opt) | HBM3e     |

**Critical insight**: Blackwell Ultra (B300) drops to **1:64** even on datacenter
silicon — NVIDIA is deprioritizing FP64 in favor of low-precision AI. The HPC
FP64 story may not survive past Hopper for new architectures. NVIDIA has stated
future architectures will add "additional FP64 capabilities" but this is not
guaranteed.

### AMD

| Generation     | Arch   | FP64:FP32 | Matrix Cores | Memory    |
|----------------|--------|-----------|--------------|-----------|
| GCN 5 (Vega)   | GCN5   | 1:4       | —            | HBM2      |
| RDNA 2         | RDNA2  | 1:16      | —            | GDDR6     |
| RDNA 3         | RDNA3  | 1:32      | —            | GDDR6     |
| RDNA 4         | RDNA4  | 1:32(est) | —            | GDDR6     |
| CDNA 2 (MI250X)| CDNA2  | **1:2**   | Matrix FP64  | HBM2e     |
| CDNA 3 (MI300X)| CDNA3  | **1:2**   | Matrix FP64 (163 TFLOPS) | HBM3  |
| CDNA 4 (MI350X)| CDNA4  | ~1:4(est) | Matrix FP64 (72 TFLOPS, halved) | HBM3e |

**Critical insight**: AMD is also cutting FP64 matrix throughput — MI350X has
half the FP64 matrix TFLOPS of MI300X. The industry trend is clear.

### Intel

| Generation     | Arch       | FP64    | XMX Engines   | Memory    |
|----------------|------------|---------|---------------|-----------|
| Xe-HPG (Arc)   | Alchemist  | None    | INT8/FP16     | GDDR6     |
| Xe-HPG (Arc)   | Battlemage | None    | INT8/FP16     | GDDR6     |
| Xe-HPC         | Ponte Vecchio | 1:2  | XMX (FP64)    | HBM2e     |
| Xe-HPC         | Falcon Shores | TBD  | TBD           | HBM3e     |

---

## 3. DF64 Decomposition Strategy

### Can f64 math be decomposed into DF64?

**Yes, completely.** barraCuda already has the full library:

#### Implemented DF64 Operations (df64_core.wgsl + df64_transcendentals.wgsl)

| Category        | Functions                                                        |
|-----------------|------------------------------------------------------------------|
| Arithmetic      | add, sub, mul, div, neg, scale, abs                              |
| Comparison      | gt, lt                                                           |
| Conversion      | from_f32, from_f64, to_f64, zero                                 |
| Elementary      | sqrt (Newton-Raphson, 2 iterations)                              |
| Exponential     | exp (Cody-Waite + degree-6 Horner), pow                          |
| Logarithmic     | log (atanh series + degree-5 Horner)                             |
| Trigonometric   | sin, cos (Cody-Waite π/2 reduction + minimax kernels)            |
| Inverse trig    | atan (Taylor + argument reduction), asin, acos, atan2            |
| Hyperbolic      | tanh, sinh, cosh                                                 |
| Special         | gamma (Lanczos 9-term), erf (Abramowitz & Stegun)               |

**Precision**: ~48-bit mantissa (~14 decimal digits) vs f64's 52-bit (~15.9).
The 4-bit gap is acceptable for Krylov solvers, MD forces, lattice QCD.

**Performance**: DF64 runs on f32 cores at ~0.4× f32 throughput. On consumer
GPUs where f64 is 1:64 throttled, DF64 is **~16× faster** than native f64.

### The Naga Blocker

DF64 transcendentals are **blocked by naga** (wgpu's WGSL→SPIR-V compiler).
Naga's SPIR-V codegen zeroes DF64 transcendental outputs — every function above
`sqrt_df64` returns all zeros when compiled through naga.

**Status**: `df64_transcendentals_safe` is forced `false` in probe_f64_builtins.
`compile_shader_df64` strips df64_transcendentals.wgsl when poisoning is detected.
DF64 arithmetic (add, mul, div, reduce) works correctly through naga.

### The coralReef Bypass

coralReef compiles WGSL → native GPU ISA **without naga**. This means:

1. DF64 transcendentals through coralReef should work on all hardware
2. coralReef also has its own native f64 lowering (Newton-Raphson + polynomial)
3. On SM≥70 NVIDIA, coralReef emits DFMA/DMul/DAdd sequences (≤1 ULP for sqrt/rcp)
4. On AMD, coralReef passes through native v_sqrt_f64 etc.

### Decision Matrix

| Hardware            | Best f64 Path                        | Why                        |
|---------------------|--------------------------------------|----------------------------|
| HPC (V100/A100/H100)| Native f64 (probed)                 | 1:2 rate, hardware works   |
| CDNA (MI250X/MI300X)| Native f64 (probed)                 | 1:2 rate, matrix f64       |
| Consumer + naga     | DF64 arithmetic only                 | Naga poisons transcendentals |
| Consumer + coralReef| DF64 full library                    | Bypasses naga, 16× faster than native |
| Consumer + coralReef| coralReef native f64 lowering        | ≤4 ULP, uses DFMA hardware |
| Any + broken f64    | coralReef f64 lowering + DF64 fallback | Multi-tier resilience    |

---

## 4. GPU Silicon Beyond Compute Shaders

### What toadStool Exposes

toadStool provides VFIO-based direct silicon access:

| Access Layer          | What It Gives Us                                  |
|-----------------------|---------------------------------------------------|
| BAR0 MMIO             | Direct register read/write (PMC, PFIFO, PBUS, FB) |
| Userspace DMA         | IOMMU-mapped host↔GPU memory transfers            |
| PCI capability scan   | Device ID → silicon capability lookup              |

### Silicon Units Available

toadStool models these execution units (per `SiliconUnit` enum):

| Unit           | Available via wgpu? | Available via VFIO+coralReef? | Use Cases                    |
|----------------|--------------------|-----------------------------|------------------------------|
| ShaderCore     | Yes (compute)      | Yes (native ISA)            | General compute              |
| TensorCore     | No                 | Yes (HMMA/WGMMA instrs)    | Matrix multiply, GEMM        |
| RT Core        | No                 | Yes (BVH traversal)         | Spatial queries              |
| TMU            | Partial (texture)  | Yes                         | Interpolation, sampling      |
| ROP            | No                 | Yes                         | Blend operations             |
| Video Encoder  | No                 | Yes (NVENC/VCN)             | Media encoding               |
| Video Decoder  | No                 | Yes (NVDEC/VCN)             | Media decoding               |
| Copy Engine    | Implicit (DMA)     | Yes (explicit DMA)          | Async data movement          |

### Tensor Core Generations (via toadStool silicon tables)

| Generation | Data Types                     | Instructions     | Peak Ops/Cycle |
|------------|--------------------------------|------------------|----------------|
| Volta      | FP16                           | HMMA              | 64 FMA/cycle   |
| Turing     | FP16, INT8                     | HMMA, IMMA        | 64 FMA/cycle   |
| Ampere     | FP16, BF16, TF32, INT8        | HMMA              | 256 FMA/cycle  |
| Ada        | FP16, BF16, TF32, INT8, FP8   | HMMA              | 256 FMA/cycle  |
| Hopper     | FP16, BF16, TF32, INT8, FP8   | WGMMA (warp-group)| 512 FMA/cycle |

**Key opportunity**: Tensor cores could accelerate batched linear algebra
(eigensolvers, matrix-matrix products) at reduced precision. For scientific
computing, the TF32 path (10-bit mantissa, sufficient for preconditioners)
gives tensor core throughput with acceptable precision for iterative methods.

### The Sovereign Pipeline

```
barraCuda (WGSL)
    → coralReef (IR → native ISA, selects execution units)
        → toadStool (VFIO submission via PFIFO/QMD, BAR0 setup)
            → GPU silicon (shader cores, tensor cores, etc.)
```

When toadStool holds the VFIO lease and coralReef compiles to native ISA,
we can target ANY silicon unit the hardware provides — not just what the
Vulkan compute shader API exposes. This is the sovereignty path.

---

## 5. Implications for barraCuda

### Near-term (current sprint)

1. **DF64 transcendentals**: Blocked by naga. Arithmetic works. Tests gated.
2. **f64 probes**: Multi-tier (individual + composite). Working correctly.
3. **Precision routing**: PrecisionBrain selects F64/DF64/F32 per physics domain.

### Medium-term (coralReef integration)

1. **DF64 transcendentals via coralReef**: When sovereign pipeline is active,
   compile DF64 shaders through coralReef to bypass naga poisoning.
2. **Native f64 lowering**: Use coralReef's polynomial lowering (≤4 ULP trig,
   ≤2 ULP exp/log, ≤1 ULP sqrt) when native f64 hardware probes fail.
3. **Capability advertising**: coralReef reports per-op polyfill availability
   via `shader.compile.capabilities` JSON-RPC endpoint.

### Long-term (full silicon access)

1. **Tensor core GEMM**: Route matrix operations through tensor cores for
   eigensolvers, preconditioners, attention mechanisms.
2. **Mixed precision iterative refinement**: Use tensor cores for approximate
   solve, shader cores for f64/DF64 residual correction.
3. **Spatial queries via RT cores**: BVH acceleration for neighbor lists in MD.
4. **Per-device silicon profile**: toadStool enumerates available units,
   barraCuda's PrecisionBrain routes work to the optimal unit.

---

## 6. Industry Trend Summary

Both NVIDIA and AMD are **deprioritizing FP64** in favor of low-precision AI
formats (FP8, FP4, MXFP). The implications:

- Native f64 at 1:2 rate may not survive beyond current HPC generations
- DF64 (f32-pair) becomes the **universal f64 substrate** — f32 cores are
  the one thing every GPU generation will always have
- coralReef's sovereign compilation is essential for accessing DF64
  transcendentals (naga blocks them) and tensor cores (Vulkan doesn't expose them)
- toadStool's VFIO access is essential for reaching silicon units that the
  standard graphics API stack doesn't surface

**The f32 core is the universal constant. Everything else is optional.**
