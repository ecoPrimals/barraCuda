// SPDX-License-Identifier: AGPL-3.0-or-later
//! DF64 (double-float, f32-pair) universal math shaders.
//!
//! These shaders use only f32 hardware but achieve ~48-bit mantissa (~14 decimal
//! digits) at up to 9.9× throughput of native f64. Use [`crate::device::WgpuDevice::compile_shader_df64`]
//! to compile (auto-injects df64_core.wgsl + df64_transcendentals.wgsl).
//!
//! DF64 values are stored as `vec2<f32>` in GPU buffers (hi in .x, lo in .y).

/// Elementwise addition at DF64 precision.
pub const WGSL_ELEMENTWISE_ADD_DF64: &str =
    include_str!("../shaders/math/elementwise_add_df64.wgsl");

/// Elementwise multiplication at DF64 precision.
pub const WGSL_ELEMENTWISE_MUL_DF64: &str =
    include_str!("../shaders/math/elementwise_mul_df64.wgsl");

/// Elementwise subtraction at DF64 precision.
pub const WGSL_ELEMENTWISE_SUB_DF64: &str =
    include_str!("../shaders/math/elementwise_sub_df64.wgsl");

/// Elementwise fused multiply-add at DF64 precision: result[i] = a[i] * b[i] + c[i].
pub const WGSL_ELEMENTWISE_FMA_DF64: &str =
    include_str!("../shaders/math/elementwise_fma_df64.wgsl");

/// Parallel sum reduction at DF64 precision (tree reduction, workgroup_size 256).
pub const WGSL_SUM_REDUCE_DF64: &str = include_str!("../shaders/reduce/sum_reduce_df64.wgsl");

/// Parallel mean reduction at DF64 precision (sum / n per workgroup).
pub const WGSL_MEAN_REDUCE_DF64: &str = include_str!("../shaders/reduce/mean_reduce_df64.wgsl");
