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
//! - ✅ **Minimal unsafe**: Exactly 2 wgpu FFI calls (pipeline cache, SPIR-V passthrough); no C FFI
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

#![deny(unsafe_code)]
#![warn(clippy::pedantic)]
// ── Domain-specific allows ──────────────────────────────────────────────────
// GPU APIs require u32 for dimensions/indices; scientific computing uses
// f64/f32 for counts/sizes.  These casts are intentional and pervasive.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap
)]
// Scientific/physics code uses single-char variables (x, y, z, r, θ),
// similar binding names (pos_x vs pos_y), and long numeric literals
// (physical constants).
#![allow(
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::unreadable_literal
)]
// Items-after-statements is common Rust style in compute kernels where
// buffers are declared close to their use.  Float equality is intentional
// in exact comparisons (0.0, NaN checks, sentinel values).
#![allow(clippy::items_after_statements, clippy::float_cmp)]
// Module-name repetitions are intentional for clarity in a large API
// (e.g., tensor::TensorError, device::DeviceSelection).
#![allow(clippy::module_name_repetitions)]
// wgpu's Default::default() pattern is idiomatic when the type is
// inferred from context (e.g., ComputePassDescriptor, CommandEncoderDescriptor).
#![allow(clippy::default_trait_access)]
// GPU ops legitimately return Result even when the current implementation
// is infallible — the contract allows future error paths (device lost,
// validation failure) without breaking callers.
#![allow(clippy::unnecessary_wraps)]
// GPU dispatch code often handles multiple match variants identically
// (e.g., different precisions routed to the same kernel).
#![allow(clippy::match_same_arms)]
// Async signatures on trait methods / framework callbacks are required by
// the trait even when the implementation is synchronous.
#![allow(clippy::unused_async)]
// GPU pipeline functions legitimately exceed 100 lines — they set up
// multiple bind groups, encoders, and staging buffers.
#![allow(clippy::too_many_lines)]
// Trait implementations require `&self` even when unused.
#![allow(clippy::unused_self)]
// Struct field name patterns are intentional for domain clarity.
#![allow(clippy::struct_field_names)]
// GPU ops take tensors/configs by value for ownership transfer; many public
// functions intentionally consume their arguments.
#![allow(clippy::needless_pass_by_value)]
// Booleans in config structs are intentional (e.g., enable_bias, use_cache).
#![allow(clippy::struct_excessive_bools)]
// Public fields prefixed with _ are used in tests or reserved for future use.
#![allow(clippy::used_underscore_items)]
// Passing single bytes by reference is acceptable for trait consistency.
#![allow(clippy::trivially_copy_pass_by_ref)]
// Debug impls intentionally omit large buffer fields.
#![allow(clippy::missing_fields_in_debug)]
// Wildcard matching against single-variant enums is intentional for forward
// compatibility — new variants will be caught at compile time.
#![allow(clippy::match_wildcard_for_single_variants)]
// inline(always) is used for hot GPU dispatch paths where the cost model
// is known.
#![allow(clippy::inline_always)]
// GPU uniform structs use `pub _padding` fields for repr(C) alignment;
// these are intentionally unused but must be pub for bytemuck::Pod derive.
#![allow(clippy::pub_underscore_fields)]
#![cfg_attr(
    test,
    expect(clippy::unwrap_used, reason = "test code uses unwrap for brevity"),
    allow(clippy::large_stack_arrays)
)]
#![expect(
    rustdoc::broken_intra_doc_links,
    reason = "cross-crate links to barracuda-core types"
)]

// ── CPU-only modules (always available, no GPU dependency) ────────────────────
pub mod error;
pub mod linalg;
pub mod nautilus;
pub mod numerical;
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

#[cfg(feature = "gpu")]
pub use ops::bio::{
    AniBatchF64, BatchFitnessGpu, Dada2EStepGpu, DnDsBatchF64, FelsensteinGpu, FelsensteinResult,
    FlatForest, FlatTree, GillespieConfig, GillespieGpu, GillespieResult, HillGateGpu,
    HmmBatchForwardF64, KmerHistogramGpu, LocusVarianceGpu, MultiObjFitnessGpu, PairwiseHammingGpu,
    PairwiseJaccardGpu, PairwiseL2Gpu, PangenomeClassifyGpu, PhyloTree, QualityConfig,
    QualityFilterGpu, RfBatchInferenceGpu, SmithWatermanGpu, SnpCallingF64, SpatialPayoffGpu,
    SwConfig, SwResult, SwarmNnGpu, TaxonomyFcGpu, TreeInferenceGpu, UniFracConfig,
    UniFracPropagateGpu,
};

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
        DeviceLease, DeviceRequirements, GpuPool, GpuVendor, MultiDevicePool, WorkloadConfig,
    };
    #[cfg(all(feature = "gpu", feature = "domain-nn"))]
    pub use crate::nn::{Layer, LossFunction, Optimizer};
    #[cfg(feature = "gpu")]
    pub use crate::npu::EventCodec;
    #[cfg(feature = "gpu")]
    pub use crate::resource_quota::{QuotaTracker, ResourceQuota, presets as quota_presets};
    #[cfg(feature = "gpu")]
    pub use crate::session::{SessionTensor, TensorSession};
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
