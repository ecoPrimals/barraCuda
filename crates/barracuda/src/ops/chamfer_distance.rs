// SPDX-License-Identifier: AGPL-3.0-or-later
//! Chamfer Distance for point clouds
//!
//! **Pure WGSL**: Single implementation via WebGPU shader
//! Measures similarity between two point clouds

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ChamferDistanceParams {
    num_points_x: u32,
    num_points_y: u32,
    point_dim: u32,
    direction: u32, // 0 = X→Y, 1 = Y→X, 2 = bidirectional
}

pub struct ChamferDistance {
    points_x: Tensor,
    points_y: Tensor,
    direction: u32,
}

impl ChamferDistance {
    /// Create ChamferDistance operation
    pub fn new(points_x: Tensor, points_y: Tensor, direction: u32) -> Result<Self> {
        if direction > 2 {
            return Err(BarracudaError::invalid_op(
                "ChamferDistance",
                format!(
                    "direction must be 0 (X→Y), 1 (Y→X), or 2 (bidirectional), got {direction}"
                ),
            ));
        }

        Ok(Self {
            points_x,
            points_y,
            direction,
        })
    }

    /// WGSL shader source (embedded at compile time)
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                "../shaders/loss/chamfer_distance_f64.wgsl"
            ))
        });
        &SHADER
    }

    /// Execute ChamferDistance on tensor
    pub fn execute(self) -> Result<Tensor> {
        let device = self.points_x.device();
        let x_shape = self.points_x.shape();
        let y_shape = self.points_y.shape();

        if x_shape.len() != 2 || y_shape.len() != 2 {
            return Err(BarracudaError::invalid_op(
                "ChamferDistance",
                format!(
                    "points must be 2D [num_points, point_dim], got shapes {x_shape:?} and {y_shape:?}"
                ),
            ));
        }

        let num_points_x = x_shape[0];
        let num_points_y = y_shape[0];
        let point_dim = x_shape[1];

        if y_shape[1] != point_dim {
            return Err(BarracudaError::invalid_op(
                "ChamferDistance",
                format!(
                    "point dimensions must match: {} != {}",
                    point_dim, y_shape[1]
                ),
            ));
        }

        // Create output buffer based on direction
        let output_size = match self.direction {
            0 => num_points_x,                // X→Y: one distance per point in X
            1 => num_points_y,                // Y→X: one distance per point in Y
            2 => num_points_x + num_points_y, // Bidirectional: both
            _ => unreachable!(),
        };
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = ChamferDistanceParams {
            num_points_x: num_points_x as u32,
            num_points_y: num_points_y as u32,
            point_dim: point_dim as u32,
            direction: self.direction,
        };

        let params_buffer = device.create_uniform_buffer("chamfer_distance_params", &params);

        let max_points = num_points_x.max(num_points_y);
        ComputeDispatch::new(device, "chamfer_distance")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.points_x.buffer())
            .storage_read(1, self.points_y.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(max_points as u32)
            .submit();

        let output_shape = match self.direction {
            0 => vec![num_points_x],
            1 => vec![num_points_y],
            2 => vec![num_points_x + num_points_y],
            _ => unreachable!(),
        };

        Ok(Tensor::from_buffer(
            output_buffer,
            output_shape,
            device.clone(),
        ))
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_chamfer_distance_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let num_points_x = 5;
        let num_points_y = 7;
        let point_dim = 3;

        let points_x = Tensor::from_vec_on(
            vec![1.0; num_points_x * point_dim],
            vec![num_points_x, point_dim],
            device.clone(),
        )
        .await
        .unwrap();

        let points_y = Tensor::from_vec_on(
            vec![2.0; num_points_y * point_dim],
            vec![num_points_y, point_dim],
            device.clone(),
        )
        .await
        .unwrap();

        let result = ChamferDistance::new(points_x, points_y, 0)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(result.shape(), &[num_points_x]);
    }
}
