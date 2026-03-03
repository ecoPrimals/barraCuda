//! Fused KL Divergence (f64) — neuralSpring V24
//!
//! D_KL(P||Q) = Σ p_i * log(p_i / q_i) computed entirely on GPU.

use std::sync::Arc;

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use bytemuck::{Pod, Zeroable};

/// WGSL shader for fused KL divergence (element-wise terms).
pub const WGSL_FUSED_KL_DIVERGENCE_F64: &str =
    include_str!("../shaders/special/fused_kl_divergence_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuParams {
    n: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

/// GPU-backed fused KL divergence.
pub struct FusedKlDivergenceGpu;

impl FusedKlDivergenceGpu {
    /// Execute KL divergence D_KL(P||Q) on GPU.
    ///
    /// Handles numerical stability: clamps to epsilon to avoid log(0).
    pub fn execute(device: Arc<WgpuDevice>, p: &[f64], q: &[f64]) -> Result<f64> {
        if p.len() != q.len() {
            return Err(BarracudaError::InvalidInput {
                message: format!("P and Q must have same length: {} vs {}", p.len(), q.len()),
            });
        }
        let n = p.len();
        if n == 0 {
            return Err(BarracudaError::InvalidInput {
                message: "P and Q cannot be empty".to_string(),
            });
        }

        let wg_size = 256u32;
        let n_workgroups = n.div_ceil(wg_size as usize) as u32;

        let p_buf = device.create_buffer_f64_init("fused_kl:p", p);
        let q_buf = device.create_buffer_f64_init("fused_kl:q", q);
        let partial_buf = device.create_buffer_f64(n_workgroups as usize)?;

        let params = GpuParams {
            n: n as u32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        let params_buf = device.create_uniform_buffer("fused_kl:params", &params);

        ComputeDispatch::new(&device, "fused_kl_divergence_f64")
            .shader(WGSL_FUSED_KL_DIVERGENCE_F64, "main")
            .f64()
            .storage_read(0, &p_buf)
            .storage_read(1, &q_buf)
            .storage_rw(2, &partial_buf)
            .uniform(3, &params_buf)
            .dispatch(n_workgroups, 1, 1)
            .submit();

        let partials = device.read_f64_buffer(&partial_buf, n_workgroups as usize)?;
        Ok(partials.iter().sum::<f64>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_f64_gpu_available;

    #[tokio::test]
    async fn test_fused_kl_construction() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let p = vec![0.25, 0.25, 0.25, 0.25];
        let q = vec![0.25, 0.25, 0.25, 0.25];

        let kl = FusedKlDivergenceGpu::execute(device, &p, &q).unwrap();
        assert!((kl - 0.0).abs() < 1e-10, "KL(P||P) should be 0, got {}", kl);
    }

    #[tokio::test]
    async fn test_fused_kl_dimension_check() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let p = vec![0.5, 0.5];
        let q = vec![0.33, 0.33, 0.34];

        let result = FusedKlDivergenceGpu::execute(device, &p, &q);
        assert!(result.is_err());
    }
}
