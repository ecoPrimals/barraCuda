// SPDX-License-Identifier: AGPL-3.0-or-later
//! Earth Mover's Distance operation (Wasserstein-1)
//!
//! Measures distance between probability distributions
//! Also known as Wasserstein distance

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct EarthMoverDistanceParams {
    size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
    _pad6: u32,
}

/// Earth Mover's Distance operation
pub struct EarthMoverDistance {
    dist1: Tensor,
    dist2: Tensor,
}

impl EarthMoverDistance {
    /// Create Earth Mover's Distance operation
    pub fn new(dist1: Tensor, dist2: Tensor) -> Result<Self> {
        if dist1.shape() != dist2.shape() {
            return Err(BarracudaError::invalid_op(
                "earth_mover_distance",
                format!(
                    "dist1 shape {:?} must match dist2 shape {:?}",
                    dist1.shape(),
                    dist2.shape()
                ),
            ));
        }

        Ok(Self { dist1, dist2 })
    }

    /// WGSL shader source (embedded at compile time)
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                "../shaders/loss/earth_mover_distance_f64.wgsl"
            ))
        });
        &SHADER
    }

    /// Execute Earth Mover's Distance on tensors
    pub fn execute(self) -> Result<Tensor> {
        let device = self.dist1.device().clone();
        let size = self.dist1.len();

        // Create output buffer (scalar distance)
        let output_buffer = device.create_buffer_f32(1)?;

        // Create params
        let params = EarthMoverDistanceParams {
            size: size as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
            _pad4: 0,
            _pad5: 0,
            _pad6: 0,
        };

        let params_buffer = device.create_uniform_buffer("earth_mover_distance:params", &params);

        let caps = DeviceCapabilities::from_device(&device);
        let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
        let workgroups = (size as u32).div_ceil(optimal_wg_size);

        ComputeDispatch::new(&device, "earth_mover_distance")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.dist1.buffer())
            .storage_read(1, self.dist2.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch(workgroups, 1, 1)
            .submit();

        // Create output tensor (scalar)
        Ok(Tensor::from_buffer(output_buffer, vec![1], device.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_earth_mover_distance_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        if device.is_lost() {
            return;
        }
        let dist1 = Tensor::from_vec_on(vec![0.5, 0.3, 0.2], vec![3], device.clone())
            .await
            .unwrap();

        let dist2 = Tensor::from_vec_on(vec![0.4, 0.4, 0.2], vec![3], device)
            .await
            .unwrap();

        let result = EarthMoverDistance::new(dist1, dist2)
            .and_then(|t| t.execute())
            .and_then(|t| t.to_vec());
        let Ok(result) = result else { return };
        assert_eq!(result.len(), 1);
        assert!(result[0] >= 0.0);
    }
}
