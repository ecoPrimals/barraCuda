// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated hydrological pipelines.
//!
//! Batch ET₀ computation, fused seasonal pipelines, and Monte Carlo
//! uncertainty propagation — all executed on GPU via WGSL shaders.

pub use super::hargreaves_gpu::HargreavesBatchGpu;
pub use super::mc_et0_gpu::{Fao56BaseInputs, Fao56Uncertainties, McEt0PropagateGpu};
pub use super::seasonal_gpu::{
    SeasonalGpuParams, SeasonalGpuParamsBuilder, SeasonalOutput, SeasonalPipelineF64,
};
