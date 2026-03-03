# BarraCuda WGSL Shader Library

**700 Production WGSL Shaders** | Universal Precision (f16/f32/f64/DF64) | Cross-Vendor Compatible

---

## Directory Structure

All shaders are organized by function category for easy discovery and reuse.

```
src/shaders/
├── activation/       37  Non-linear activations (ReLU, GELU, Swish, Sigmoid, etc.)
├── attention/         8  Attention mechanisms (MHA, GQA, Flash, Causal, etc.)
├── audio/             9  Audio/signal processing (STFT, MFCC, Mel, Griffin-Lim, etc.)
├── augmentation/     10  Data augmentation (CutMix, Mixup, Random transforms, etc.)
├── conv/             11  Convolution operations (1D/2D/3D, Depthwise, Dilated, etc.)
├── detection/         5  Object detection primitives (NMS, Anchors, BBox, IoU, etc.)
├── dropout/           2  Regularization (Dropout, Spatial dropout)
├── gnn/               6  Graph neural networks (GCN, GAT, GIN, SAGE, etc.)
├── gradient/          1  Gradient manipulation (Clip norm, clip value)
├── interpolation/     2  Interpolation kernels (RBF, LOO-CV)
├── linalg/           11  Linear algebra (Cholesky, Eigh, LinSolve, Inverse, etc.)
├── loss/             31  Loss functions (Focal, Dice, IoU, BCE, MSE, etc.)
├── math/             68  Element-wise math (Trig, Exp, Log, Floor, Sqrt, etc.)
├── misc/             56  Utilities (MatMul, Embedding, Quantize, etc.)
├── norm/             27  Normalization (BatchNorm, LayerNorm, GroupNorm, etc.)
├── optimizer/        13  Weight update rules (Adam, SGD, RMSProp, etc.)
├── pooling/          17  Spatial reduction (MaxPool, AvgPool, Adaptive, etc.)
├── reduce/           14  Tensor reduction (Sum, Mean, Argmax, Logsumexp, etc.)
├── rnn/               4  Recurrent networks (LSTM, GRU, BiLSTM)
├── special/           5  Special functions (Bessel J0/J1/I0/K0, Spherical harmonics)
└── tensor/           41  Shape manipulation (Concat, Slice, Reshape, Transpose, etc.)
```

**Plus specialized subdirectories:**
- `complex/` (10 shaders in `src/ops/complex/`) -- Complex number arithmetic
- `fft/` (2 shaders in `src/ops/fft/`) -- Fast Fourier transforms
- `fhe/` (13 shaders in `src/ops/`) -- Fully homomorphic encryption
- `md/` (9 shaders in `src/ops/md/`) -- Molecular dynamics (forces, integrators, PBC)

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

### Precision (Universal — Math Is Universal, Precision Is Silicon)
- **Canonical**: All shaders are f64 canonical. f32 generated at runtime via `LazyLock<String>` + `downcast_f64_to_f32()`.
- **Pipeline**: `compile_shader_universal(source, precision)` routes to f16/f32/f64/DF64.
- **Abstract ops**: New shaders can use `op_add`/`op_mul` via `compile_op_shader()` for precision-agnostic code.
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

### activation/ (37 shaders)
Non-linear activation functions. All are element-wise operations.

**Common**: ReLU, GELU, Swish, Sigmoid, Tanh  
**Advanced**: Mish, CELU, Hardswish, PReLU, RReLU  
**Pattern**: `f(x) → y` where `x` and `y` are same shape

### attention/ (8 shaders)
Attention mechanism components for transformers.

**Core**: Attention matmul, softmax, apply  
**Variants**: Causal, cross, local, sparse  
**Advanced**: Flash attention, GQA (grouped query attention)  
**Pattern**: Q, K, V → Attention(Q,K,V)

### linalg/ (11 shaders)
Linear algebra operations for scientific computing.

**Decompositions**: Cholesky, Eigh (eigenvalue), QR, LU, SVD  
**Solvers**: LinSolve (Ax=b), triangular solve  
**Operations**: Inverse, determinant, trace  
**Pattern**: Dense matrices → decomposition or solution

### loss/ (31 shaders)
Training objectives and distance metrics.

**Classification**: Cross-entropy, focal loss, NLL  
**Detection**: IoU loss, GIoU loss, dice loss  
**Contrastive**: Triplet loss, contrastive loss, center loss  
**Regression**: MSE, MAE, Huber, smooth L1  
**Pattern**: (predictions, targets) → scalar loss

### tensor/ (41 shaders)
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

**Last Updated**: March 2, 2026 — Session 86  
**Shader Count**: 671 (21 DF64, 577 f64, 2 f32-named — zero f32-only, all f64 canonical). 4 additional DF64 force-field shaders in `ops/md/forces/` (25 DF64 workspace-wide).  
**Categories**: 41 directories  
**Status**: Production — dual-layer universal precision operational, 15-function DF64 transcendental suite complete
