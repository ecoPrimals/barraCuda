// SPDX-License-Identifier: AGPL-3.0-or-later
//! `GridMask` - Grid-based masking augmentation (Chen et al.)
//!
//! Masks structured grid regions in images.
//! Prevents overfitting to spatial structures.
//!
//! Deep Debt Principles:
//! - Pure GPU/WGSL execution
//! - Safe Rust wrappers
//! - Hardware-agnostic via WebGPU
//! - Runtime device discovery
//! - Zero CPU fallbacks in execution

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

const SHADER_F64: &str = include_str!("../shaders/augmentation/grid_mask_f64.wgsl");

/// `GridMask` operation
pub struct GridMask {
    input: Tensor,
    ratio: f32,
    rotate: f32,
    grid_size: usize,
    seed: u64,
}

impl GridMask {
    /// Create a new grid mask operation
    /// # Errors
    /// Returns [`Err`] if input is not 3D (C, H, W), or if ratio is not in [0, 1].
    pub fn new(
        input: Tensor,
        ratio: f32,
        rotate: f32,
        grid_size: usize,
        seed: u64,
    ) -> Result<Self> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(crate::error::BarracudaError::invalid_op(
                "GridMask",
                format!("Expected 3D tensor (C, H, W), got {}D", shape.len()),
            ));
        }

        if !(0.0..=1.0).contains(&ratio) {
            return Err(crate::error::BarracudaError::invalid_op(
                "GridMask",
                format!("Ratio must be in [0, 1], got {ratio}"),
            ));
        }

        Ok(Self {
            input,
            ratio,
            rotate,
            grid_size,
            seed,
        })
    }

    /// Execute the grid mask operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, buffer readback fails
    /// (e.g. device lost).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();

        let channels = shape[0];
        let height = shape[1];
        let width = shape[2];

        // Compute random offsets from seed (CPU-side, deterministic)
        let offset_x = ((self.seed * 1_103_515_245) % self.grid_size as u64) as usize;
        let offset_y = ((self.seed * 22_695_477) % self.grid_size as u64) as usize;

        let mask_size = (self.grid_size as f32 * self.ratio) as usize;
        let angle_rad = self.rotate.to_radians();
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();

        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;

        let output_size = channels * height * width;

        // Create buffers
        let input_buffer = self.input.buffer();

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GridMask Output"),
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
            ratio: f32,
            rotate: f32,
            grid_size: u32,
            offset_x: u32,
            offset_y: u32,
            mask_size: u32,
            cos_a: f32,
            sin_a: f32,
            cx: f32,
            cy: f32,
        }

        let params = Params {
            channels: channels as u32,
            height: height as u32,
            width: width as u32,
            ratio: self.ratio,
            rotate: self.rotate,
            grid_size: self.grid_size as u32,
            offset_x: offset_x as u32,
            offset_y: offset_y as u32,
            mask_size: mask_size as u32,
            cos_a,
            sin_a,
            cx,
            cy,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GridMask Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let workgroups_x = (width as u32).div_ceil(16);
        let workgroups_y = (height as u32).div_ceil(16);

        ComputeDispatch::new(device, "GridMask")
            .shader(SHADER_F64, "main")
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
    /// Apply grid mask augmentation
    /// # Arguments
    /// * `ratio` - Mask ratio (0.0 to 1.0)
    /// * `rotate` - Rotation angle in degrees
    /// * `grid_size` - Size of grid cells
    /// * `seed` - Random seed for deterministic masking
    /// # Errors
    /// Returns [`Err`] if input is not 3D, ratio is not in [0, 1], buffer allocation fails,
    /// GPU dispatch fails, or buffer readback fails (e.g. device lost).
    pub fn grid_mask(self, ratio: f32, rotate: f32, grid_size: usize, seed: u64) -> Result<Self> {
        GridMask::new(self, ratio, rotate, grid_size, seed)?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_grid_mask_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let image_data = vec![1.0; 3 * 224 * 224];
        let tensor = Tensor::from_vec_on(image_data.clone(), vec![3, 224, 224], device)
            .await
            .unwrap();
        let masked_tensor = tensor.grid_mask(0.6, 15.0, 96, 11_111).unwrap();
        let masked = masked_tensor.to_vec().unwrap();
        assert_eq!(masked.len(), image_data.len());
        // Some pixels should be masked (set to 0)
        assert!(masked.contains(&0.0));
        assert!(masked.iter().any(|&x| x > 0.0));
    }

    #[tokio::test]
    async fn test_grid_mask_edge_cases() {
        let device = crate::device::test_pool::get_test_device().await;
        // Ratio = 0 (no masking)
        let image_data = vec![1.0; 32 * 32];
        let tensor = Tensor::from_vec_on(image_data.clone(), vec![1, 32, 32], device.clone())
            .await
            .unwrap();
        let masked_tensor = tensor.grid_mask(0.0, 0.0, 16, 12_345).unwrap();
        let masked = masked_tensor.to_vec().unwrap();
        assert_eq!(masked, image_data); // No masking applied

        // Small image
        let small_image_data = vec![1.0; 8 * 8];
        let tensor = Tensor::from_vec_on(small_image_data.clone(), vec![1, 8, 8], device)
            .await
            .unwrap();
        let masked_tensor = tensor.grid_mask(0.5, 0.0, 4, 99_999).unwrap();
        let masked = masked_tensor.to_vec().unwrap();
        assert_eq!(masked.len(), 64);
    }

    #[tokio::test]
    async fn test_grid_mask_boundary() {
        let device = crate::device::test_pool::get_test_device().await;
        // Ratio = 1.0 (maximum masking)
        let image_data = vec![1.0; 64 * 64];
        let tensor = Tensor::from_vec_on(image_data.clone(), vec![1, 64, 64], device.clone())
            .await
            .unwrap();
        let masked_tensor = tensor.grid_mask(1.0, 0.0, 32, 77_777).unwrap();
        let masked = masked_tensor.to_vec().unwrap();
        assert_eq!(masked.len(), image_data.len());
        assert!(masked.contains(&0.0));

        // With rotation
        let image_data = vec![1.0; 64 * 64];
        let tensor = Tensor::from_vec_on(image_data.clone(), vec![1, 64, 64], device)
            .await
            .unwrap();
        let masked_tensor = tensor.grid_mask(0.5, 45.0, 16, 55_555).unwrap();
        let masked = masked_tensor.to_vec().unwrap();
        assert_eq!(masked.len(), image_data.len());
    }

    #[tokio::test]
    async fn test_grid_mask_large_batch() {
        let device = crate::device::test_pool::get_test_device().await;
        // RGB image (3 channels)
        let channels = 3;
        let height = 128;
        let width = 128;
        let image_data = vec![1.0; channels * height * width];
        let tensor = Tensor::from_vec_on(image_data.clone(), vec![channels, height, width], device)
            .await
            .unwrap();
        let masked_tensor = tensor.grid_mask(0.6, 30.0, 48, 88_888).unwrap();
        let masked = masked_tensor.to_vec().unwrap();
        assert_eq!(masked.len(), image_data.len());
        assert!(masked.contains(&0.0));
        assert!(masked.iter().any(|&x| x > 0.0));
    }

    #[tokio::test]
    async fn test_grid_mask_precision() {
        let device = crate::device::test_pool::get_test_device().await;
        // Deterministic with same seed
        let image_data = vec![1.0; 32 * 32];
        let tensor1 = Tensor::from_vec_on(image_data.clone(), vec![1, 32, 32], device.clone())
            .await
            .unwrap();
        let masked_tensor1 = tensor1.grid_mask(0.5, 0.0, 16, 12_345).unwrap();
        let masked1 = masked_tensor1.to_vec().unwrap();

        let tensor2 = Tensor::from_vec_on(image_data.clone(), vec![1, 32, 32], device.clone())
            .await
            .unwrap();
        let masked_tensor2 = tensor2.grid_mask(0.5, 0.0, 16, 12_345).unwrap();
        let masked2 = masked_tensor2.to_vec().unwrap();

        // Same seed should produce same mask
        assert_eq!(masked1, masked2);

        // Different seed should produce different mask
        let tensor3 = Tensor::from_vec_on(image_data.clone(), vec![1, 32, 32], device)
            .await
            .unwrap();
        let masked_tensor3 = tensor3.grid_mask(0.5, 0.0, 16, 99_999).unwrap();
        let masked3 = masked_tensor3.to_vec().unwrap();
        let different = masked1
            .iter()
            .zip(masked3.iter())
            .any(|(a, b)| (a - b).abs() > 0.1);
        assert!(different);
    }
}
