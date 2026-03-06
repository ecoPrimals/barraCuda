// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hardware-Agnostic Echo State Network (ESN) API
//!
//! **EVOLVED v2**: Uses `BarraCuda` Tensors - Works on ANY hardware!
//!
//! This module provides a production-ready interface for training and using
//! Echo State Networks using `BarraCuda`'s universal Tensor operations.
//!
//! # Philosophy
//!
//! ESN operations ARE `BarraCuda` operations! Instead of CPU-specific code,
//! we use universal tensor operations that work on CPU, GPU, and NPU.
//!
//! # Deep Debt Compliance
//!
//! - ✅ **Hardware agnostic**: Uses Tensor operations (CPU/GPU/NPU)
//! - ✅ **Pure Rust**: `BarraCuda` is 100% Rust
//! - ✅ **Fast**: Leverages best device for workload
//! - ✅ **Safe**: Zero unsafe code
//! - ✅ **Capability-based**: Runtime device discovery
//! - ✅ **No hardcoding**: User can specify device

mod config;
mod model;
mod multi_head;
mod npu;

pub use config::{ESNConfig, expect_size, validate_config};
pub use model::{ESN, ExportedWeights};
pub use multi_head::{HeadConfig, HeadGroup, MultiHeadEsn};
pub use npu::{NpuReadoutWeights, dequantize_affine_i8_f64, quantize_affine_i8_f64};

/// GPU shader for fused reservoir update: `W_in·input` + `W_res·state` → leaky tanh → new state.
#[must_use]
pub fn wgsl_reservoir_update() -> &'static str {
    static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
            "../shaders/ml/esn_reservoir_update_f64.wgsl"
        ))
    });
    std::sync::LazyLock::force(&SHADER).as_str()
}

/// GPU shader for readout: output[i] = `W_out`[i,:] · state (matrix-vector product).
pub const WGSL_READOUT: &str = include_str!("../shaders/ml/esn_readout.wgsl");

/// GPU shader for readout (alias).
pub const WGSL_ESN_READOUT: &str = include_str!("../shaders/ml/esn_readout.wgsl");

/// f64 version of the reservoir update for universal math library portability.
pub const WGSL_ESN_RESERVOIR_UPDATE_F64: &str =
    include_str!("../shaders/esn/esn_reservoir_update_f64.wgsl");

/// f64 version of the readout for universal math library portability.
pub const WGSL_ESN_READOUT_F64: &str = include_str!("../shaders/ml/esn_readout_f64.wgsl");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn esn_reservoir_update_f64_shader_contains_main() {
        assert!(WGSL_ESN_RESERVOIR_UPDATE_F64.contains("fn main"));
        assert!(WGSL_ESN_RESERVOIR_UPDATE_F64.contains("f64"));
    }

    #[test]
    fn esn_reservoir_update_f64_shader_compiles_via_naga() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        let _ = device.compile_shader_f64(
            WGSL_ESN_RESERVOIR_UPDATE_F64,
            Some("esn_reservoir_update_f64"),
        );
    }
}
