// SPDX-License-Identifier: AGPL-3.0-only
//! `IoULoss` - Intersection over Union loss
//!
//! **Canonical `BarraCuda` Pattern**: Struct with new/execute
//!
//! Direct optimization of `IoU` metric.
//! Used in segmentation and object detection.

use crate::device::DeviceCapabilities;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// `IoU` Loss operation
pub struct IoULoss {
    predictions: Tensor,
    targets: Tensor,
    smooth: f32,
}

impl IoULoss {
    /// Create a new `IoU` loss operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(predictions: Tensor, targets: Tensor, smooth: f32) -> Result<Self> {
        // Validate shapes match
        if predictions.shape() != targets.shape() {
            return Err(BarracudaError::shape_mismatch(
                predictions.shape().to_vec(),
                targets.shape().to_vec(),
            ));
        }

        Ok(Self {
            predictions,
            targets,
            smooth,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!("../shaders/loss/iou_loss_f64.wgsl").to_string()
        });
        &SHADER
    }

    /// Execute the `IoU` loss operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.predictions.device();
        let size = self.predictions.len();

        // Number of workgroups for pass 1 (workgroup_size 256)
        let num_workgroups = (size as u32).div_ceil(crate::device::capabilities::WORKGROUP_SIZE_1D);

        // Create reduction buffers - one slot per workgroup for partial sums
        let intersection_buffer = device.create_buffer_f32(num_workgroups as usize)?;
        let union_buffer = device.create_buffer_f32(num_workgroups as usize)?;
        let output_buffer = device.create_buffer_f32(1)?;

        // Zero-initialize partial sum buffers
        device.write_buffer_f32(&intersection_buffer, &vec![0.0; num_workgroups as usize])?;
        device.write_buffer_f32(&union_buffer, &vec![0.0; num_workgroups as usize])?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            smooth_val: f32,
            num_partials: u32,
            _pad1: u32,
        }

        let params = Params {
            size: size as u32,
            smooth_val: self.smooth,
            num_partials: num_workgroups,
            _pad1: 0,
        };
        let params_buffer = device.create_uniform_buffer("IoU Loss Params", &params);

        let caps = DeviceCapabilities::from_device(device);
        let workgroups = caps.dispatch_1d(size as u32);

        ComputeDispatch::new(device, "iou_loss_pass1")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.predictions.buffer())
            .storage_read(1, self.targets.buffer())
            .storage_rw(2, &intersection_buffer)
            .storage_rw(3, &union_buffer)
            .storage_rw(4, &output_buffer)
            .uniform(5, &params_buffer)
            .dispatch(workgroups, 1, 1)
            .submit()?;

        ComputeDispatch::new(device, "iou_loss_pass2")
            .shader(Self::wgsl_shader(), "compute_loss")
            .storage_read(0, self.predictions.buffer())
            .storage_read(1, self.targets.buffer())
            .storage_rw(2, &intersection_buffer)
            .storage_rw(3, &union_buffer)
            .storage_rw(4, &output_buffer)
            .uniform(5, &params_buffer)
            .dispatch(1, 1, 1)
            .submit()?;

        let output_data = crate::utils::read_buffer(device, &output_buffer, 1)?;
        Ok(Tensor::new(output_data, vec![1], device.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_iou_loss() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let predictions = Tensor::from_vec_on(vec![0.8; 500], vec![500], device.clone())
            .await
            .unwrap();
        let targets = Tensor::from_vec_on(vec![1.0; 500], vec![500], device.clone())
            .await
            .unwrap();
        let loss = IoULoss::new(predictions, targets, 1e-6)
            .unwrap()
            .execute()
            .unwrap();
        let result = loss.to_vec().unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0] > 0.0 && result[0] < 1.0);
    }
}
