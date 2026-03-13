# BarraCuda WGSL Shader Library

**731 Production WGSL Shaders** (806 total including ops/) | 3-Tier Precision (f32/f64/df64) | Cross-Vendor Compatible

---

## Directory Structure

All shaders are organized by function category for easy discovery and reuse.

```
src/shaders/
├── activation/       40  Non-linear activations (ReLU, GELU, Swish, Sigmoid, etc.)
├── attention/        19  Attention mechanisms (MHA, GQA, Flash, Causal, etc.)
├── audio/             9  Audio/signal processing (STFT, MFCC, Mel, Griffin-Lim, etc.)
├── augmentation/     10  Data augmentation (CutMix, Mixup, Random transforms, etc.)
├── bio/              39  Bioinformatics (Smith-Waterman, HMM, phylogenetics, etc.)
├── conv/             11  Convolution operations (1D/2D/3D, Depthwise, Dilated, etc.)
├── detection/         5  Object detection primitives (NMS, Anchors, BBox, IoU, etc.)
├── dropout/           2  Regularization (Dropout, Spatial dropout)
├── folding/          15  Protein folding (AlphaFold2 components)
├── gnn/               6  Graph neural networks (GCN, GAT, GIN, SAGE, etc.)
├── gradient/          1  Gradient manipulation (Clip norm, clip value)
├── grid/              9  Grid/mesh operations
├── interpolation/     3  Interpolation kernels (RBF, LOO-CV)
├── lattice/          36  Lattice QCD (SU(3), Wilson, Dirac, HMC)
├── linalg/           32  Linear algebra (Cholesky, Eigh, LinSolve, Inverse, etc.)
├── loss/             34  Loss functions (Focal, Dice, IoU, BCE, MSE, etc.)
├── math/            106  Element-wise math (Trig, Exp, Log, Floor, Sqrt, etc.)
├── md/                5  Molecular dynamics (forces, integrators, PBC)
├── misc/             74  Utilities (MatMul, Embedding, Quantize, etc.)
├── ml/                8  Machine learning (ESN, reservoir, etc.)
├── norm/             28  Normalization (BatchNorm, LayerNorm, GroupNorm, etc.)
├── numerical/         7  Numerical methods (ODE, integration)
├── optimizer/        18  Weight update rules (Adam, SGD, RMSProp, etc.)
├── pooling/          17  Spatial reduction (MaxPool, AvgPool, Adaptive, etc.)
├── reduce/           28  Tensor reduction (Sum, Mean, Argmax, Logsumexp, etc.)
├── rnn/               4  Recurrent networks (LSTM, GRU, BiLSTM)
├── sample/            4  Sampling methods (LHS, Sobol, etc.)
├── science/          16  Scientific compute (batched elementwise, hydrology, etc.)
├── sparse/            5  Sparse linear algebra (SpMV, CG)
├── special/          40  Special functions (Bessel, Gamma, Erf, etc.)
├── spectral/          3  Spectral analysis (Anderson, Lanczos)
├── stats/             5  Statistical operations (bootstrap, diversity, etc.)
└── tensor/           43  Shape manipulation (Concat, Slice, Reshape, Transpose, etc.)
```

**2 additional tooling directories** (`precision/`, `sovereign/`)
contain precision infrastructure and DF64 rewrite tooling rather than standalone shaders.

---

## Quick Reference

### By Use Case

**Deep Learning Training:**
- **Activation**: `activation/gelu.wgsl`, `activation/silu.wgsl`, `activation/relu.wgsl`
- **Loss**: `loss/focal_loss.wgsl`, `loss/cross_entropy.wgsl`, `loss/mse_loss.wgsl`
- **Optimizer**: `optimizer/adam.wgsl`, `optimizer/adamw.wgsl`, `optimizer/sgd.wgsl`
- **Norm**: `norm/batch_norm.wgsl`, `norm/layer_norm.wgsl`, `norm/group_norm.wgsl`

**Vision/CNN:**
- **Conv**: `conv/conv2d.wgsl`, `conv/depthwise_conv2d.wgsl`, `conv/grouped_conv2d.wgsl`
- **Pooling**: `pooling/maxpool2d.wgsl`, `pooling/avgpool2d.wgsl`, `pooling/adaptive_avgpool2d.wgsl`
- **Detection**: `detection/nms.wgsl`, `detection/anchor_generator.wgsl`, `detection/box_iou.wgsl`
- **Augment**: `augmentation/cutmix.wgsl`, `augmentation/mixup.wgsl`, `augmentation/color_jitter.wgsl`

**NLP/Transformers:**
- **Attention**: `attention/attention_matmul.wgsl`, `attention/flash_attention.wgsl`, `attention/gqa_matmul.wgsl`
- **RNN**: `rnn/lstm_cell.wgsl`, `rnn/gru_cell.wgsl`, `rnn/bi_lstm.wgsl`
- **Embedding**: `misc/embedding.wgsl`

**Scientific Computing:**
- **LinAlg**: `linalg/cholesky.wgsl`, `linalg/eigh.wgsl`, `linalg/linsolve.wgsl`, `linalg/triangular_solve.wgsl`
- **Special**: `special/bessel_j0.wgsl`, `special/bessel_i0.wgsl`, `special/spherical_harmonics.wgsl`
- **MD Forces**: `../ops/md/forces/coulomb.wgsl`, `../ops/md/forces/lennard_jones.wgsl`
- **MD Integrators**: `../ops/md/integrators/velocity_verlet.wgsl`, `../ops/md/integrators/rk4.wgsl`
- **Interpolation**: `interpolation/rbf_kernel.wgsl`, `interpolation/loo_cv.wgsl`

**Audio/Signal:**
- **Audio**: `audio/stft.wgsl`, `audio/istft.wgsl`, `audio/mfcc.wgsl`, `audio/mel_scale.wgsl`
- **FFT**: `../ops/fft/fft_1d.wgsl`

**Graph Neural Networks:**
- **GNN**: `gnn/gcn_conv.wgsl`, `gnn/gat_conv.wgsl`, `gnn/gin_conv.wgsl`, `gnn/edge_conv.wgsl`

**Cryptography:**
- **FHE**: `../ops/fhe_ntt.wgsl`, `../ops/fhe_poly_add.wgsl`, `../ops/fhe_and.wgsl`
- **Complex**: `../ops/complex/add.wgsl`, `../ops/complex/mul.wgsl`, `../ops/complex/exp.wgsl`

---

## Usage Patterns

### Including Shaders in Rust

**From ops at `src/ops/{name}.rs`:**
```rust
const SHADER: &str = include_str!("../shaders/{category}/{name}.wgsl");
```

**From ops subdirectory at `src/ops/{subdir}/mod.rs`:**
```rust
const SHADER: &str = include_str!("../../shaders/{category}/{name}.wgsl");
```

**Example:**
```rust
// src/ops/gelu.rs
const SHADER: &str = include_str!("../shaders/activation/gelu.wgsl");

// src/ops/attention/mod.rs
const ATTENTION_MATMUL: &str = include_str!("../../shaders/attention/attention_matmul.wgsl");
```

### Common Shader Patterns

**Point-wise operations (element-wise):**
- Input: `@binding(0) var<storage, read> input: array<f32>`
- Output: `@binding(1) var<storage, read_write> output: array<f32>`
- Workgroup: `@workgroup_size(256)` typically

**Reduction operations:**
- Shared memory: `var<workgroup> shared: array<f32, 256>`
- Synchronization: `workgroupBarrier()`
- Tree reduction pattern

**2D spatial operations (conv, pool):**
- Input shape: `(batch, channels, height, width)`
- Output shape calculated from input + kernel + stride + padding

**Attention patterns:**
- Q, K, V matrices
- Softmax normalization
- Causal masking for autoregressive models

---

## Adding New Shaders

### 1. Determine Category

Choose the most specific category:
- **Activation** if it's a non-linear point-wise function
- **Loss** if it's a training objective
- **Math** if it's a general mathematical operation
- **Tensor** if it's primarily about shape manipulation
- **Misc** if it doesn't fit elsewhere

### 2. Create Shader File

```bash
# Example: adding a new activation
cd crates/barracuda/src/shaders/activation
vim my_new_activation.wgsl
```

### 3. Create Rust Wrapper

```bash
cd crates/barracuda/src/ops
vim my_new_activation.rs
```

### 4. Template

```rust
use crate::prelude::*;

const SHADER: &str = include_str!("../shaders/activation/my_new_activation.wgsl");

pub struct MyNewActivation {
    input: Tensor,
}

impl MyNewActivation {
    pub fn new(input: Tensor) -> BarracudaResult<Self> {
        Ok(Self { input })
    }

    pub fn execute(self) -> BarracudaResult<Vec<f32>> {
        let device = &self.input.device;
        let size = self.input.total_size();
        
        device.execute_shader(
            SHADER,
            "MyNewActivation",
            &[&self.input.buffer],
            size,
            256,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_my_new_activation() {
        // Test implementation
    }
}
```

### 5. Register in `mod.rs`

```rust
// In src/ops/mod.rs
pub mod my_new_activation;
pub use my_new_activation::MyNewActivation;
```

---

## Shader Standards

All shaders follow these conventions:

### Naming
- **File**: `snake_case.wgsl`
- **Entry point**: Usually `@compute` with function name `main`

### Bindings
- **Read-only inputs**: `@binding(N) var<storage, read>`
- **Read-write outputs**: `@binding(N) var<storage, read_write>`
- **Uniforms**: `@binding(N) var<uniform>`

### Workgroup Size
- **1D ops**: `@workgroup_size(256)` (or 128, 512)
- **2D ops**: `@workgroup_size(16, 16)` typically
- **3D ops**: `@workgroup_size(8, 8, 8)` typically

### Precision (3-Tier — Aligned with coralReef `Fp64Strategy`)
- **Canonical**: All shaders are f64-canonical WGSL — pure math, conceptually infinite precision.
- **3 tiers**: `compile_shader()` (f32), `compile_shader_f64()` (f64), `compile_shader_df64()` (df64/fp48).
- **Routing**: `fp64_strategy()` selects f64-native vs df64 based on hardware capabilities.
- **DF64**: `compile_shader_df64()` auto-injects `df64_core.wgsl` + `df64_transcendentals.wgsl`.
- **FHE**: `u32` for modular arithmetic
- **Quantization**: `u8`, `i8`, `i4` via bit packing

---

## Performance Tips

### Choosing the Right Shader

| Operation Type | Prefer | Avoid |
|----------------|--------|-------|
| Element-wise math | `math/` shaders | Custom implementations |
| Matrix multiply | `misc/matmul_tiled.wgsl` | Naive matmul |
| Convolution | `conv/conv2d.wgsl` | Im2col + matmul |
| Attention (large) | `attention/flash_attention.wgsl` | Naive attention |
| Reduction | `reduce/` shaders with tree reduction | Sequential reduction |

### Memory Access Patterns
- **Coalesced**: Access consecutive elements in a warp
- **Shared memory**: For data reuse within workgroup
- **Uniform buffers**: For small constant data

### Workgroup Size
- **NVIDIA**: Multiples of 32 (warp size)
- **AMD**: Multiples of 64 (wavefront size)
- **Intel**: Multiples of 16 (SIMD width)
- **Safe default**: 256 for 1D, (16,16) for 2D

---

## Shader Categories Explained

### activation/ (40 shaders)
Non-linear activation functions. All are element-wise operations.

**Common**: ReLU, GELU, Swish, Sigmoid, Tanh  
**Advanced**: Mish, CELU, Hardswish, PReLU, RReLU  
**Pattern**: `f(x) → y` where `x` and `y` are same shape

### attention/ (19 shaders)
Attention mechanism components for transformers.

**Core**: Attention matmul, softmax, apply  
**Variants**: Causal, cross, local, sparse  
**Advanced**: Flash attention, GQA (grouped query attention)  
**Pattern**: Q, K, V → Attention(Q,K,V)

### linalg/ (32 shaders)
Linear algebra operations for scientific computing.

**Decompositions**: Cholesky, Eigh (eigenvalue), QR, LU, SVD  
**Solvers**: LinSolve (Ax=b), triangular solve  
**Operations**: Inverse, determinant, trace  
**Pattern**: Dense matrices → decomposition or solution

### loss/ (34 shaders)
Training objectives and distance metrics.

**Classification**: Cross-entropy, focal loss, NLL  
**Detection**: IoU loss, GIoU loss, dice loss  
**Contrastive**: Triplet loss, contrastive loss, center loss  
**Regression**: MSE, MAE, Huber, smooth L1  
**Pattern**: (predictions, targets) → scalar loss

### tensor/ (43 shaders)
Shape manipulation and indexing operations.

**Combine**: Concat, stack, tile, repeat  
**Split**: Chunk, slice, split, narrow  
**Reshape**: Reshape, view, flatten, squeeze  
**Index**: Gather, scatter, index_select, masked_fill  
**Pattern**: Shape transformation without computation

---

## Finding Similar Shaders

**Need activation?** → `activation/`  
**Need loss function?** → `loss/`  
**Need matrix operation?** → `linalg/` or `misc/matmul*`  
**Need convolution?** → `conv/`  
**Need pooling?** → `pooling/`  
**Need attention?** → `attention/`  
**Need normalization?** → `norm/`  
**Need reduction?** → `reduce/`  
**Need element-wise math?** → `math/`  
**Need shape manipulation?** → `tensor/`  
**Need optimizer?** → `optimizer/`  

When in doubt, check `misc/` for general utilities.

---

## Maintenance

### Testing New Shaders
```bash
cargo test -p barracuda --lib {test_name}
```

### Benchmarking
```bash
cargo bench -p barracuda --bench {shader_category}
```

### Cross-Vendor Validation
```bash
# Run on all available GPUs
cargo test -p barracuda --lib --release
```

---

**Last Updated**: March 13, 2026  
**Shader Count**: 731 in shaders/ (806 total with ops/) — organized by domain, all f64 canonical. DF64 emulated double-precision on hardware without native f64.  
**Categories**: 41 directories  
**Status**: Production — dual-layer universal precision operational, 15-function DF64 transcendental suite complete
