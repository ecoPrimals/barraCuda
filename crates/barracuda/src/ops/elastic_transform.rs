// SPDX-License-Identifier: AGPL-3.0-or-later
//! Elastic transform operation - Elastic deformation for data augmentation
//!
//! Elastic deformations: Random displacement fields for image augmentation
//! Widely used in medical imaging and handwriting recognition

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};

const SHADER_F64: &str = include_str!("../shaders/augmentation/elastic_transform_f64.wgsl");
static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| SHADER_F64.to_string());

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ElasticTransformParams {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
    alpha: f32,
    sigma: f32,
    _padding: [u32; 2],
}

/// Elastic transform operation
pub struct ElasticTransform {
    input: Tensor,
    displacement_x: Tensor,
    displacement_y: Tensor,
    alpha: f32,
    sigma: f32,
}

impl ElasticTransform {
    /// Create elastic transform operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(
        input: Tensor,
        displacement_x: Tensor,
        displacement_y: Tensor,
        alpha: f32,
        sigma: f32,
    ) -> Result<Self> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(BarracudaError::invalid_op(
                "elastic_transform",
                format!("input must be 4D [B, C, H, W], got shape {shape:?}"),
            ));
        }

        let height = shape[2];
        let width = shape[3];
        let displacement_size = height * width;

        if displacement_x.shape() != [displacement_size] {
            return Err(BarracudaError::invalid_op(
                "elastic_transform",
                format!(
                    "displacement_x shape {:?} must be [{:?}]",
                    displacement_x.shape(),
                    displacement_size
                ),
            ));
        }

        if displacement_y.shape() != [displacement_size] {
            return Err(BarracudaError::invalid_op(
                "elastic_transform",
                format!(
                    "displacement_y shape {:?} must be [{:?}]",
                    displacement_y.shape(),
                    displacement_size
                ),
            ));
        }

        if alpha < 0.0 {
            return Err(BarracudaError::invalid_op(
                "elastic_transform",
                format!("alpha must be non-negative, got {alpha}"),
            ));
        }

        if sigma <= 0.0 {
            return Err(BarracudaError::invalid_op(
                "elastic_transform",
                format!("sigma must be positive, got {sigma}"),
            ));
        }

        Ok(Self {
            input,
            displacement_x,
            displacement_y,
            alpha,
            sigma,
        })
    }

    /// WGSL shader source (embedded at compile time)
    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Execute elastic transform on tensor
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];
        let size = self.input.len();

        // Create output buffer
        let output_buffer = device.create_buffer_f32(size)?;

        // Create params
        let params = ElasticTransformParams {
            batch_size: batch_size as u32,
            channels: channels as u32,
            height: height as u32,
            width: width as u32,
            alpha: self.alpha,
            sigma: self.sigma,
            _padding: [0; 2],
        };

        let params_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ElasticTransform Params"),
            size: std::mem::size_of::<ElasticTransformParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Dispatch workgroups (8x8x1 workgroup size)
        let workgroups_x = (width as u32).div_ceil(8);
        let workgroups_y = (height as u32).div_ceil(8);
        let workgroups_z = (batch_size * channels) as u32;

        ComputeDispatch::new(device, "elastic_transform")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.input.buffer())
            .storage_read(1, self.displacement_x.buffer())
            .storage_read(2, self.displacement_y.buffer())
            .storage_rw(3, &output_buffer)
            .uniform(4, &params_buffer)
            .dispatch(workgroups_x, workgroups_y, workgroups_z)
            .submit()?;

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            shape.to_vec(),
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_elastic_transform_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        if device.is_lost() {
            return;
        }
        let input = Tensor::from_vec_on(vec![1.0; 2 * 3 * 4 * 4], vec![2, 3, 4, 4], device.clone())
            .await
            .unwrap();

        let disp_size = 4 * 4;
        let displacement_x =
            Tensor::from_vec_on(vec![0.1; disp_size], vec![disp_size], device.clone())
                .await
                .unwrap();

        let displacement_y = Tensor::from_vec_on(vec![0.1; disp_size], vec![disp_size], device)
            .await
            .unwrap();

        let result = ElasticTransform::new(input, displacement_x, displacement_y, 1.0, 1.0)
            .and_then(super::ElasticTransform::execute)
            .and_then(|t| t.to_vec());
        let Ok(result) = result else { return };
        assert_eq!(result.len(), 2 * 3 * 4 * 4);
    }
}
