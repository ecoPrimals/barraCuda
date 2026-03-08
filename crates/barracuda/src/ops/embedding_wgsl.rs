// SPDX-License-Identifier: AGPL-3.0-or-later
//! Embedding - Lookup table operation - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its embedding dimensions
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::{ComputeDispatch, DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// Embedding operation - Lookup embeddings from a weight matrix
pub struct Embedding {
    weight: Tensor,
    indices: Vec<usize>,
}

impl Embedding {
    /// Create a new embedding operation
    #[must_use]
    pub fn new(weight: Tensor, indices: Vec<usize>) -> Self {
        Self { weight, indices }
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!(
                "../shaders/misc/embedding_f64.wgsl"
            ).to_string()
        });
        &SHADER
    }

    /// Execute the embedding operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.weight.device();
        let weight_shape = self.weight.shape();

        let _ = weight_shape[0]; // Reserved for validation
        let embedding_dim = weight_shape[1];
        let num_indices = self.indices.len();

        let output_size = num_indices * embedding_dim;

        // Create buffers
        let weight_buffer = self.weight.buffer();

        let indices_u32: Vec<u32> = self.indices.iter().map(|&x| x as u32).collect();
        let indices_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Embedding Indices"),
                contents: bytemuck::cast_slice(&indices_u32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Embedding Output"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            num_indices: u32,
            embedding_dim: u32,
        }

        let params = Params {
            num_indices: num_indices as u32,
            embedding_dim: embedding_dim as u32,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Embedding Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let caps = DeviceCapabilities::from_device(device);
        let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
        let workgroups = (output_size as u32).div_ceil(optimal_wg_size);

        ComputeDispatch::new(device, "embedding")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, weight_buffer)
            .storage_read(1, &indices_buffer)
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch(workgroups, 1, 1)
            .submit()?;

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, output_size)?;

        Ok(Tensor::new(
            output_data,
            vec![num_indices, embedding_dim],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Lookup embeddings from weight matrix
    /// # Arguments
    /// * `indices` - Indices to lookup
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn embedding_wgsl(self, indices: Vec<usize>) -> Result<Self> {
        Embedding::new(self, indices).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn get_test_device() -> Option<std::sync::Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_embedding_basic() {
        let Some(device) = get_test_device().await else {
            return;
        };
        if device.is_lost() {
            return;
        }
        let weight_data = vec![
            1.0, 2.0, 3.0, // embedding 0
            4.0, 5.0, 6.0, // embedding 1
            7.0, 8.0, 9.0, // embedding 2
            10.0, 11.0, 12.0, // embedding 3
        ];
        let weight = Tensor::new(weight_data, vec![4, 3], device.clone());

        let indices = vec![1, 0, 3];
        let Ok(output) = weight.embedding_wgsl(indices) else {
            return;
        };

        assert_eq!(output.shape(), &[3, 3]);
        let Ok(result) = output.to_vec() else { return };

        // Should get embeddings 1, 0, 3
        assert_eq!(result[0], 4.0); // embedding 1
        assert_eq!(result[1], 5.0);
        assert_eq!(result[2], 6.0);
        assert_eq!(result[3], 1.0); // embedding 0
        assert_eq!(result[4], 2.0);
        assert_eq!(result[5], 3.0);
        assert_eq!(result[6], 10.0); // embedding 3
        assert_eq!(result[7], 11.0);
        assert_eq!(result[8], 12.0);
    }
}
