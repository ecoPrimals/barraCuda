// SPDX-License-Identifier: AGPL-3.0-or-later
//! Label Smoothing for classification
//!
//! **Pure WGSL**: Single implementation via WebGPU shader
//! Prevents overconfidence by smoothing hard labels

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LabelSmoothingParams {
    batch_size: u32,
    num_classes: u32,
    smoothing: f32,
    _padding: u32,
}

/// Label smoothing: prevents overconfidence by smoothing hard one-hot labels.
pub struct LabelSmoothing {
    labels: Tensor,
    num_classes: u32,
    smoothing: f32,
}

impl LabelSmoothing {
    /// Create `LabelSmoothing` operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(labels: Tensor, num_classes: u32, smoothing: f32) -> Result<Self> {
        if !(0.0..=1.0).contains(&smoothing) {
            return Err(BarracudaError::invalid_op(
                "LabelSmoothing",
                format!("smoothing must be in [0, 1], got {smoothing}"),
            ));
        }

        Ok(Self {
            labels,
            num_classes,
            smoothing,
        })
    }

    /// WGSL shader source (embedded at compile time)
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                "../shaders/loss/label_smoothing_f64.wgsl"
            ))
        });
        &SHADER
    }

    /// Execute `LabelSmoothing` on tensor
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.labels.device();
        let labels_shape = self.labels.shape();

        if labels_shape.len() != 1 {
            return Err(BarracudaError::invalid_op(
                "LabelSmoothing",
                format!("labels must be 1D [batch_size], got shape {labels_shape:?}"),
            ));
        }

        let batch_size = labels_shape[0];

        // Create output buffer: [batch_size, num_classes]
        let output_size = batch_size * self.num_classes as usize;
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = LabelSmoothingParams {
            batch_size: batch_size as u32,
            num_classes: self.num_classes,
            smoothing: self.smoothing,
            _padding: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LabelSmoothing Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let caps = DeviceCapabilities::from_device(device.as_ref());
        let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
        let workgroups = (batch_size as u32).div_ceil(optimal_wg_size);

        ComputeDispatch::new(device.as_ref(), "LabelSmoothing")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.labels.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch(workgroups, 1, 1)
            .submit()?;

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, self.num_classes as usize],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_label_smoothing_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let batch_size = 4;
        let num_classes = 3;

        let labels =
            Tensor::from_vec_on(vec![0.0, 1.0, 2.0, 0.0], vec![batch_size], device.clone())
                .await
                .unwrap();

        let result = LabelSmoothing::new(labels, num_classes, 0.1)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(result.shape(), &[batch_size, num_classes as usize]);
    }
}
