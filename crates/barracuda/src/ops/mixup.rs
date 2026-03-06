// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mixup data augmentation
//!
//! **Pure WGSL**: Single implementation via WebGPU shader
//! Mixes two training examples and their labels

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

const SHADER_F64: &str = include_str!("../shaders/augmentation/mixup_f64.wgsl");
static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| crate::shaders::precision::downcast_f64_to_f32(SHADER_F64));

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MixupParams {
    batch_size: u32,
    feature_size: u32,
    lambda: f32,
    mix_idx: u32,
}

/// Mixup augmentation: mixes two training examples with weight lambda.
pub struct Mixup {
    input: Tensor,
    lambda: f32,
    mix_idx: u32,
}

impl Mixup {
    /// Create Mixup operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(input: Tensor, lambda: f32, mix_idx: u32) -> Result<Self> {
        if !(0.0..=1.0).contains(&lambda) {
            return Err(BarracudaError::invalid_op(
                "Mixup",
                format!("lambda must be in [0, 1], got {lambda}"),
            ));
        }

        Ok(Self {
            input,
            lambda,
            mix_idx,
        })
    }

    /// WGSL shader source (embedded at compile time)
    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Execute Mixup on tensor
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_shape = self.input.shape();

        if input_shape.len() != 2 {
            return Err(BarracudaError::invalid_op(
                "Mixup",
                format!("input must be 2D [batch_size, feature_size], got shape {input_shape:?}"),
            ));
        }

        let batch_size = input_shape[0];
        let feature_size = input_shape[1];

        // Create output buffer: [batch_size, feature_size]
        let output_size = batch_size * feature_size;
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = MixupParams {
            batch_size: batch_size as u32,
            feature_size: feature_size as u32,
            lambda: self.lambda,
            mix_idx: self.mix_idx,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Mixup Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let caps = DeviceCapabilities::from_device(device);
        let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
        let total_elements = batch_size * feature_size;
        let workgroups = (total_elements as u32).div_ceil(optimal_wg_size);

        ComputeDispatch::new(device, "mixup")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch(workgroups, 1, 1)
            .submit()?;

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            input_shape.to_vec(),
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_mixup_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let batch_size = 4;
        let feature_size = 3;

        let input = Tensor::from_vec_on(
            vec![1.0; batch_size * feature_size],
            vec![batch_size, feature_size],
            device.clone(),
        )
        .await
        .unwrap();

        let result = Mixup::new(input, 0.5, 1).unwrap().execute().unwrap();

        assert_eq!(result.shape(), &[batch_size, feature_size]);
    }
}
