// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hill Gate — Two-input Hill AND gate (f64 pipeline).
//!
//! Computes f(a, b) = vmax × H(a, `K_a`, `n_a`) × H(b, `K_b`, `n_b`) where
//! H(x, K, n) = x^n / (K^n + x^n) is the Hill function. Used for regulatory
//! network signal integration.
//!
//! Provenance: neuralSpring metalForge → toadStool absorption

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;

/// WGSL source for f32 Hill gate (paired or grid mode).
pub const WGSL_HILL_GATE: &str = include_str!("../../shaders/bio/hill_gate.wgsl");

/// f64 version for universal math library portability.
pub const WGSL_HILL_GATE_F64: &str = include_str!("../../shaders/bio/hill_gate_f64.wgsl");

/// Parameters for Hill gate GPU kernel.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HillGateParams {
    /// Size of input A (or rows in grid mode)
    pub n_a: u32,
    /// Size of input B (or cols in grid mode)
    pub n_b: u32,
    /// 0 = paired (output[i] = f(a[i], b[i])), 1 = grid
    pub mode: u32,
    /// Padding for alignment
    pub _pad: u32,
    /// Hill coefficient `K_a` for input A
    pub k_a: f64,
    /// Hill coefficient `K_b` for input B
    pub k_b: f64,
    /// Precomputed `n_a^exponent`
    pub n_a_exp: f64,
    /// Precomputed `n_b^exponent`
    pub n_b_exp: f64,
    /// Maximum output value
    pub vmax: f64,
    /// Padding for alignment
    pub _pad2: f64,
}

/// Hill gate GPU kernel (f64 pipeline).
pub struct HillGateGpu {
    device: Arc<WgpuDevice>,
}

impl HillGateGpu {
    /// Create new Hill gate GPU kernel.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Compute Hill gate. Mode 0: paired (output[i] = f(a[i], b[i])).
    /// Mode 1: grid (output[ix*`n_b` + iy] = f(a[ix], b[iy])).
    #[expect(clippy::missing_panics_doc, reason = "dispatch submit is infallible on valid device")]
    pub fn dispatch(
        &self,
        input_a: &wgpu::Buffer,
        input_b: &wgpu::Buffer,
        output: &wgpu::Buffer,
        params: &HillGateParams,
    ) {
        let d = self.device.device();

        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("HillGate Params"),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let element_count = if params.mode == 0 {
            params.n_a
        } else {
            params.n_a * params.n_b
        };

        ComputeDispatch::new(&self.device, "HillGate")
            .shader(WGSL_HILL_GATE_F64, "main")
            .f64()
            .storage_read(0, input_a)
            .storage_read(1, input_b)
            .storage_rw(2, output)
            .uniform(3, &params_buf)
            .dispatch_1d(element_count)
            .submit()
            .expect("HillGate GPU dispatch failed");
    }
}

#[cfg(test)]
mod tests {
    use super::{HillGateGpu, HillGateParams, WGSL_HILL_GATE, WGSL_HILL_GATE_F64};

    #[test]
    fn sanity_constants_exported() {
        assert!(!WGSL_HILL_GATE.is_empty());
        assert!(WGSL_HILL_GATE.contains("fn main"));
        assert!(WGSL_HILL_GATE.contains("HillGateParams"));
        assert!(std::any::type_name::<HillGateGpu>().contains("HillGateGpu"));
        assert!(std::any::type_name::<HillGateParams>().contains("HillGateParams"));
    }

    #[test]
    fn f64_shader_contains_main() {
        assert!(WGSL_HILL_GATE_F64.contains("fn main"));
        assert!(WGSL_HILL_GATE_F64.contains("f64"));
    }

    #[test]
    fn f64_shader_compiles_via_naga() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        let _ = device.compile_shader_f64(WGSL_HILL_GATE_F64, Some("hill_gate_f64"));
    }
}
