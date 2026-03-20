// SPDX-License-Identifier: AGPL-3.0-or-later
//! Atanh — inverse hyperbolic tangent, GPU-resident, pipeline-cached, batchable
//!
//! `atanh(x) = 0.5 · ln((1+x)/(1−x))`, valid for x ∈ (−1, 1)

use crate::device::pipeline_cache::{BindGroupLayoutSignature, GLOBAL_CACHE};
use crate::device::tensor_context::get_device_context;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// f64 is the canonical source — math is universal, precision is silicon.
const SHADER_F64: &str = include_str!("../shaders/activation/atanh_f64.wgsl");
static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| SHADER_F64.to_string());

/// Element-wise inverse hyperbolic tangent: atanh(x), valid for x ∈ (-1, 1).
pub struct Atanh {
    input: Tensor,
}

impl Atanh {
    /// Create an atanh operation.
    #[must_use]
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }

    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Execute atanh on GPU.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size = self.input.len();
        let ctx = get_device_context(device);
        let caps = DeviceCapabilities::from_device(device);
        let wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
        let workgroups = (size as u32).div_ceil(wg_size);
        let adapter_info = device.adapter_info();

        let output_buffer = ctx.acquire_pooled_output(size);

        // elementwise_unary: 1 read-only + 1 read-write storage, no uniform.
        let layout_sig = BindGroupLayoutSignature::elementwise_unary();
        let bind_group = ctx.get_or_create_bind_group(
            layout_sig,
            &[self.input.buffer(), &output_buffer],
            Some("Atanh BG"),
        );

        let pipeline = GLOBAL_CACHE.get_or_create_pipeline(
            device.device(),
            adapter_info,
            Self::wgsl_shader(),
            layout_sig,
            "main",
            Some("Atanh Pipeline"),
        );

        ctx.record_operation(move |encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Atanh Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&*bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        })?;

        Ok(Tensor::from_pooled_buffer(
            output_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Compute element-wise atanh(x).
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn atanh(self) -> Result<Self> {
        Atanh::new(self).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn get_test_device() -> Option<std::sync::Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_atanh_finite() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let input = Tensor::new(vec![-0.9, -0.5, 0.0, 0.5, 0.9], vec![5], device);
        let output = input.atanh().unwrap();
        let result = output.to_vec().unwrap();
        assert!(
            result.iter().all(|&x| x.is_finite()),
            "atanh produced non-finite: {result:?}"
        );
        assert!(
            result[2].abs() < 1e-5,
            "atanh(0) should be 0, got {}",
            result[2]
        );
    }
}
