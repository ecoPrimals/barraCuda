// SPDX-License-Identifier: AGPL-3.0-or-later
//! # barraCuda: Hardware-Agnostic Tensor Compute
//!
//! **Deep Debt Excellence**: Zero duplication, pure capability-based compute
//!
//! ## Philosophy
//!
//! - ✅ **Hardware-Agnostic**: One API, works on any device (GPU/CPU/NPU/TPU)
//! - ✅ **Pure WGSL**: WGSL shaders ONLY (wgpu handles all backends)
//! - ✅ **Automatic Fallback**: wgpu uses CPU when GPU unavailable
//! - ✅ **Zero Duplication**: Single WGSL implementation per operation
//! - ✅ **Runtime Discovery**: wgpu selects best available backend
//! - ✅ **Simple**: No separate CPU code, no trait abstractions
//! - **Zero unsafe**: `#![forbid(unsafe_code)]` — all former FFI evolved to safe Rust
//!
//! ## Architecture
//!
//! ```text
//! User Code: Tensor<f32>
//!     ↓
//! Operation (WGSL shader)
//!     ↓
//! WgpuDevice
//!     ↓
//! wgpu Backend Selection (automatic):
//! ├── Vulkan (NVIDIA, AMD, Intel GPU)
//! ├── Metal (Apple GPU)
//! ├── DX12 (Windows GPU)
//! └── Software Rasterizer (CPU fallback)
//!
//! Same WGSL code runs on ALL backends!
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use barracuda::prelude::*;
//!
//! // Auto-discovers best device (GPU if available, CPU fallback)
//! let x = Tensor::randn([128, 256])?;
//! let y = Tensor::randn([256, 512])?;
//!
//! // Operations execute on discovered device (WGSL on GPU, Rayon on CPU)
//! let z = x.matmul(&y)?;
//! let activated = z.relu()?;
//! let normalized = activated.softmax(0)?;
//!
//! println!("Device: {}", x.device().name());
//! // "NVIDIA GeForce RTX 4090" or "AMD Radeon RX 7900" or "CPU (16 cores)"
//! ```
//!
//! ## Deep Debt Elimination
//!
//! **Before** (architectural debt):
//! - Separate CPU and GPU implementations
//! - User must choose backend explicitly
//! - WGSL shaders existed but weren't used by operations
//! - Duplication: Same logic in CPU and WGSL
//!
//! **After** (unified):
//! - Single Tensor API works everywhere
//! - Automatic device discovery and fallback
//! - All WGSL shaders properly utilized
//! - Zero duplication: One implementation per op

#![forbid(unsafe_code)]
// ── Domain-specific expectations ────────────────────────────────────────────
// Compile-time verified: #[expect] warns if the suppression becomes unnecessary.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    reason = "GPU APIs require u32 for dimensions/indices; scientific computing uses f64/f32 for counts/sizes"
)]
#![expect(
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::unreadable_literal,
    reason = "scientific/physics code uses single-char variables (x, y, z, r, θ), similar bindings (pos_x vs pos_y), and physical constants"
)]
#![expect(
    clippy::items_after_statements,
    clippy::float_cmp,
    reason = "compute kernels declare buffers close to use; float equality is intentional for exact comparisons (0.0, NaN, sentinels)"
)]
#![expect(
    clippy::module_name_repetitions,
    reason = "intentional for clarity in a large API (e.g. tensor::TensorError, device::DeviceSelection)"
)]
#![expect(
    clippy::default_trait_access,
    reason = "wgpu's Default::default() pattern is idiomatic when type is inferred (ComputePassDescriptor, CommandEncoderDescriptor)"
)]
#![expect(
    clippy::unnecessary_wraps,
    reason = "GPU ops return Result for future error paths (device lost, validation) without breaking callers"
)]
#![expect(
    clippy::match_same_arms,
    reason = "GPU dispatch routes multiple precision variants to the same kernel"
)]
#![expect(
    clippy::missing_fields_in_debug,
    reason = "Debug impls intentionally omit large buffer fields"
)]
#![expect(
    clippy::match_wildcard_for_single_variants,
    reason = "wildcard matching is intentional for forward compatibility — new variants caught at compile time"
)]
#![expect(
    clippy::pub_underscore_fields,
    reason = "GPU uniform structs use pub _padding fields for repr(C) alignment with bytemuck::Pod"
)]
#![expect(
    clippy::unused_async,
    reason = "async signatures required by trait even when implementation is synchronous"
)]
#![expect(
    clippy::too_many_lines,
    reason = "GPU pipeline functions set up multiple bind groups, encoders, and staging buffers"
)]
#![expect(
    clippy::unused_self,
    reason = "trait implementations require &self even when unused"
)]
#![expect(
    clippy::struct_field_names,
    reason = "struct field name patterns are intentional for domain clarity"
)]
#![expect(
    clippy::needless_pass_by_value,
    reason = "GPU ops take tensors/configs by value for ownership transfer"
)]
#![expect(
    clippy::struct_excessive_bools,
    reason = "booleans in config structs are intentional (enable_bias, use_cache)"
)]
#![expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "passing single bytes by reference is acceptable for trait consistency"
)]
#![cfg_attr(
    test,
    expect(clippy::unwrap_used, reason = "test code uses unwrap for brevity")
)]
#![expect(
    rustdoc::broken_intra_doc_links,
    reason = "cross-crate links to barracuda-core types"
)]

// ── CPU-only modules (always available, no GPU dependency) ────────────────────
pub mod activations;
pub mod cast;
pub mod discovery;
pub mod error;
pub mod health;
pub mod linalg;
pub mod nautilus;
pub mod numerical;
pub mod rng;
pub mod special;
pub mod tolerances;
pub mod validation;

// ── GPU-dependent modules (require the "gpu" feature) ────────────────────────
#[cfg(feature = "gpu")]
pub mod auto_tensor;
#[cfg(feature = "gpu")]
pub mod benchmarks;
#[cfg(feature = "gpu")]
pub mod compute_graph;
#[cfg(feature = "gpu")]
pub mod cpu_conv_pool;
#[cfg(feature = "gpu")]
pub mod cpu_executor;
#[cfg(feature = "gpu")]
pub mod device;
#[cfg(feature = "gpu")]
pub mod dispatch;
#[cfg(all(feature = "gpu", feature = "domain-esn"))]
pub mod esn_v2;
#[cfg(all(feature = "gpu", feature = "domain-genomics"))]
pub mod genomics;
#[cfg(feature = "gpu")]
pub mod gpu_executor;
#[cfg(feature = "gpu")]
pub mod interpolate;
#[cfg(feature = "gpu")]
pub mod multi_gpu;
#[cfg(all(feature = "gpu", feature = "domain-nn"))]
pub mod nn;
#[cfg(feature = "gpu")]
pub mod npu;
#[cfg(feature = "gpu")]
pub mod npu_executor;
#[cfg(feature = "gpu")]
pub mod ops;
#[cfg(feature = "gpu")]
pub mod optimize;
#[cfg(all(feature = "gpu", feature = "domain-pde"))]
pub mod pde;
#[cfg(feature = "gpu")]
pub mod pipeline;
#[cfg(feature = "gpu")]
pub mod provenance;
#[cfg(feature = "gpu")]
pub mod resource_quota;
pub mod sample;
#[cfg(feature = "gpu")]
pub mod scheduler;
#[cfg(feature = "gpu")]
pub mod session;
#[cfg(feature = "gpu")]
pub mod shaders;
#[cfg(all(feature = "gpu", feature = "domain-snn"))]
pub mod snn;
pub mod spectral;
#[cfg(feature = "gpu")]
pub mod tensor;
#[cfg(all(feature = "gpu", feature = "domain-timeseries"))]
pub mod timeseries;
#[cfg(feature = "gpu")]
pub mod utils;
#[cfg(all(feature = "gpu", feature = "domain-vision"))]
pub mod vision;
#[cfg(feature = "gpu")]
pub mod workload;

/// CPU-only math functions (convenience alias for `special`).
///
/// Re-exports the most commonly needed special functions so Springs
/// can `use barracuda::math::{erf, ln_gamma, regularized_gamma_p}`.
pub mod math {
    pub use crate::special::erf::{erf_batch, erfc_batch};
    pub use crate::special::{
        beta, digamma, erf, erfc, gamma, ln_beta, ln_gamma, lower_incomplete_gamma,
        regularized_gamma_p, regularized_gamma_q,
    };
    pub use crate::stats::metrics::{dot, l2_norm};
    pub use crate::stats::normal::norm_cdf;
}
pub mod stats;

#[cfg(feature = "gpu")]
pub mod staging;
#[cfg(feature = "gpu")]
pub mod surrogate;
#[cfg(feature = "gpu")]
pub mod unified_hardware;
#[cfg(feature = "gpu")]
pub mod unified_math;

#[cfg(feature = "gpu")]
pub use ops::sparse_matmul_quantized::sparse_matmul_quantized;

#[cfg(all(feature = "gpu", feature = "domain-genomics"))]
pub use ops::bio::{
    AniBatchF64, BatchFitnessGpu, Dada2Buffers, Dada2Dimensions, Dada2DispatchArgs, Dada2EStepGpu,
    DnDsBatchF64, FelsensteinGpu, FelsensteinResult, FlatForest, FlatTree, GillespieConfig,
    GillespieGpu, GillespieModel, GillespieResult, HillGateGpu, HmmBatchForwardF64, HmmForwardArgs,
    KmerHistogramGpu, LocusVarianceGpu, MultiObjFitnessGpu, PairwiseHammingGpu, PairwiseJaccardGpu,
    PairwiseL2Gpu, PangenomeClassifyGpu, PhyloTree, QualityConfig, QualityFilterGpu,
    RfBatchInferenceGpu, SmithWatermanGpu, SnpCallingF64, SpatialPayoffGpu, SwConfig, SwResult,
    SwarmNnGpu, TaxonomyFcGpu, TreeInferenceGpu, UniFracConfig, UniFracPropagateGpu,
};

#[cfg(feature = "gpu")]
pub use device::driver_profile::PrecisionRoutingAdvice;

#[cfg(feature = "gpu")]
pub use ops::rk45_adaptive::Rk45DispatchArgs;

/// Prelude: Common imports for using barracuda.
///
/// `use barracuda::prelude::*` brings in `Tensor`, `Device`, `Result`, and domain-specific
/// types (ESN, genomics, NN, SNN) when their features are enabled. Use this for application
/// code; use explicit paths when you need to avoid name collisions.
pub mod prelude {
    pub use crate::error::{BarracudaError, Result};

    #[cfg(feature = "gpu")]
    pub use crate::compute_graph::ComputeGraph;
    #[cfg(feature = "gpu")]
    pub use crate::device::{
        Auto, AutoTuner, Capability, Device, DeviceCapabilities, DeviceContext, DeviceInfo,
        GLOBAL_TUNER, GpuCalibration, WgpuDevice, WorkloadHint,
    };
    #[cfg(feature = "gpu")]
    pub use crate::dispatch::{
        DispatchConfig, DispatchTarget, batch_fitness_substrate, dispatch_for,
        dispatch_with_transfer_cost, hmm_substrate, ode_substrate, pairwise_substrate,
        spatial_substrate,
    };
    #[cfg(all(feature = "gpu", feature = "domain-esn"))]
    pub use crate::esn_v2::{ESN, ESNConfig};
    #[cfg(all(feature = "gpu", feature = "domain-genomics"))]
    pub use crate::genomics::{
        CompositionReport, MotifMatch, QualityReport, SequenceAnalyzer, SequenceConfig,
    };
    #[cfg(feature = "gpu")]
    pub use crate::multi_gpu::DeviceInfo as GpuDeviceInfo;
    #[cfg(feature = "gpu")]
    pub use crate::multi_gpu::{
        DeviceClass, DeviceLease, DeviceRequirements, GpuPool, MultiDevicePool, WorkloadConfig,
    };
    #[cfg(all(feature = "gpu", feature = "domain-nn"))]
    pub use crate::nn::{Layer, LossFunction, Optimizer};
    #[cfg(feature = "gpu")]
    pub use crate::npu::EventCodec;
    #[cfg(feature = "gpu")]
    pub use crate::resource_quota::{QuotaTracker, ResourceQuota, presets as quota_presets};
    #[cfg(feature = "gpu")]
    pub use crate::session::{AttentionDims, SessionTensor, TensorSession};
    #[cfg(all(feature = "gpu", feature = "domain-snn"))]
    pub use crate::snn::{SNNConfig, SNNLayer, SpikingNetwork};
    #[cfg(feature = "gpu")]
    pub use crate::staging::{
        BufferDirection, GpuRingBuffer, PipelineBuilder, PipelineStats, RingBufferConfig, Stage,
        StageLink, StreamingPipeline, UnidirectionalConfig, UnidirectionalPipeline, WorkHandle,
        WriteHandle,
    };
    #[cfg(feature = "gpu")]
    pub use crate::tensor::Tensor;
    #[cfg(feature = "gpu")]
    pub use crate::workload::{
        ComputeDevice, DeviceHint, DeviceSelector, Priority, SparsityAnalyzer, WorkloadClassifier,
        WorkloadType,
    };
}
