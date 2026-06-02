// SPDX-License-Identifier: AGPL-3.0-or-later

//! Semi-empirical mass formula (SEMF) GPU shaders for nuclear EOS L1.

/// Batched SEMF: one thread per nucleus, evaluates Bethe-Weizsäcker formula.
pub const WGSL_SEMF_BATCH: &str = include_str!("../../shaders/physics/semf_batch_f64.wgsl");

/// Pure-GPU SEMF using `math_f64` library — no CPU precomputation.
pub const WGSL_SEMF_PURE_GPU: &str = include_str!("../../shaders/physics/semf_pure_gpu_f64.wgsl");

/// Batched chi-squared: per-nucleus squared residual.
pub const WGSL_CHI2_BATCH: &str = include_str!("../../shaders/physics/chi2_batch_f64.wgsl");
