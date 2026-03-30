// SPDX-License-Identifier: AGPL-3.0-or-later
//! `RandomPerspective` - Random perspective transformation
//!
//! Applies random perspective distortion.
//! Simulates different camera viewpoints.
//!
//! Deep Debt Principles:
//! - Pure GPU/WGSL execution
//! - Safe Rust wrappers
//! - Hardware-agnostic via WebGPU
//! - Runtime device discovery
//! - Zero CPU fallbacks in execution
//!
//! Shader: f64 canonical (downcast to f32 at compile)

const SHADER_F64: &str = include_str!("../shaders/augmentation/random_perspective_f64.wgsl");

use crate::device::DeviceCapabilities;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// `RandomPerspective` operation
pub struct RandomPerspective {
    input: Tensor,
    distortion_scale: f32,
    seed: u64,
}

impl RandomPerspective {
    /// Create a new random perspective operation
    /// # Errors
    /// Returns [`Err`] if input is not 3D (C, H, W).
    pub fn new(input: Tensor, distortion_scale: f32, seed: u64) -> Result<Self> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(crate::error::BarracudaError::invalid_op(
                "RandomPerspective",
                format!("Expected 3D tensor (C, H, W), got {}D", shape.len()),
            ));
        }

        Ok(Self {
            input,
            distortion_scale,
            seed,
        })
    }

    /// Get the WGSL shader source (f64 canonical, downcast to f32 at compile)
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> =
            std::sync::LazyLock::new(|| SHADER_F64.to_string());
        SHADER.as_str()
    }

    /// Execute the random perspective operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer readback fails (e.g. device lost).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();

        let channels = shape[0];
        let height = shape[1];
        let width = shape[2];

        // Generate random corner displacements from seed (CPU-side, deterministic)
        let mut rng = self.seed;
        let mut rand = || {
            rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((rng % 2000) as f32 / 1000.0 - 1.0) * self.distortion_scale
        };

        // Source corners
        let src_corners = [
            (0.0, 0.0),
            (width as f32, 0.0),
            (width as f32, height as f32),
            (0.0, height as f32),
        ];

        // Destination corners with random displacement
        let dst_corners = [
            (
                src_corners[0].0 + rand() * width as f32,
                src_corners[0].1 + rand() * height as f32,
            ),
            (
                src_corners[1].0 + rand() * width as f32,
                src_corners[1].1 + rand() * height as f32,
            ),
            (
                src_corners[2].0 + rand() * width as f32,
                src_corners[2].1 + rand() * height as f32,
            ),
            (
                src_corners[3].0 + rand() * width as f32,
                src_corners[3].1 + rand() * height as f32,
            ),
        ];

        let output_size = channels * height * width;

        // Create buffers
        let input_buffer = self.input.buffer();

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RandomPerspective Output"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            channels: u32,
            height: u32,
            width: u32,
            _pad: u32,
            dst_corner0: [f32; 2],
            dst_corner1: [f32; 2],
            dst_corner2: [f32; 2],
            dst_corner3: [f32; 2],
        }

        let params = Params {
            channels: channels as u32,
            height: height as u32,
            width: width as u32,
            _pad: 0,
            dst_corner0: [dst_corners[0].0, dst_corners[0].1],
            dst_corner1: [dst_corners[1].0, dst_corners[1].1],
            dst_corner2: [dst_corners[2].0, dst_corners[2].1],
            dst_corner3: [dst_corners[3].0, dst_corners[3].1],
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RandomPerspective Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let caps = DeviceCapabilities::from_device(device);
        let (workgroups_x, workgroups_y) = caps.dispatch_2d(width as u32, height as u32);

        ComputeDispatch::new(device, "random_perspective")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch(workgroups_x, workgroups_y, 1)
            .submit()?;

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, output_size)?;

        Ok(Tensor::new(
            output_data,
            vec![channels, height, width],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply random perspective transformation
    /// # Arguments
    /// * `distortion_scale` - Scale of perspective distortion
    /// * `seed` - Random seed for deterministic transformation
    /// # Errors
    /// Returns [`Err`] if input is not 3D, or buffer allocation/GPU dispatch/readback fails (e.g. device lost).
    pub fn random_perspective(self, distortion_scale: f32, seed: u64) -> Result<Self> {
        RandomPerspective::new(self, distortion_scale, seed)?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_random_perspective() {
        let device = crate::device::test_pool::get_test_device().await;
        let image_data = vec![1.0; 3 * 100 * 100];
        let tensor = Tensor::from_vec_on(image_data.clone(), vec![3, 100, 100], device)
            .await
            .unwrap();
        let transformed_tensor = tensor.random_perspective(0.2, 33_333).unwrap();
        let transformed = transformed_tensor.to_vec().unwrap();
        assert_eq!(transformed.len(), image_data.len());
    }
}
