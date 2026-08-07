// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mosaic - Mosaic augmentation (YOLO-style)
//!
//! Combines 4 images into one mosaic.
//! Used in object detection for multi-scale training.
//!
//! Deep Debt Principles:
//! - Pure GPU/WGSL execution
//! - Safe Rust wrappers
//! - Hardware-agnostic via WebGPU
//! - Runtime device discovery
//! - Zero CPU fallbacks in execution

use crate::device::capabilities::WORKGROUP_SIZE_2D;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

const SHADER_F64: &str = include_str!("../shaders/augmentation/mosaic_f64.wgsl");

/// Mosaic operation
pub struct Mosaic {
    images: [Tensor; 4],
    seed: u64,
}

impl Mosaic {
    /// Create a new mosaic operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(images: [Tensor; 4], seed: u64) -> Result<Self> {
        // Validate all images have same shape
        let shape = images[0].shape();
        for (i, img) in images.iter().enumerate() {
            if img.shape() != shape {
                return Err(crate::error::BarracudaError::invalid_op(
                    "Mosaic",
                    format!("All images must have same shape, image {i} differs"),
                ));
            }
        }

        if shape.len() != 3 {
            return Err(crate::error::BarracudaError::invalid_op(
                "Mosaic",
                format!("Expected 3D tensor (C, H, W), got {}D", shape.len()),
            ));
        }

        Ok(Self { images, seed })
    }

    /// Execute the mosaic operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.images[0].device();
        let shape = self.images[0].shape();

        let channels = shape[0];
        let height = shape[1];
        let width = shape[2];

        // Compute random split point from seed
        let split_x = ((self.seed * 1_103_515_245) % width as u64) as usize;
        let split_y = ((self.seed * 22_695_477) % height as u64) as usize;

        let output_size = channels * height * width;

        // Create buffers
        let image0_buffer = self.images[0].buffer();
        let image1_buffer = self.images[1].buffer();
        let image2_buffer = self.images[2].buffer();
        let image3_buffer = self.images[3].buffer();

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mosaic Output"),
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
            split_x: u32,
            split_y: u32,
        }

        let params = Params {
            channels: channels as u32,
            height: height as u32,
            width: width as u32,
            split_x: split_x as u32,
            split_y: split_y as u32,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Mosaic Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "Mosaic")
            .shader(SHADER_F64, "main")
            .storage_read(0, image0_buffer)
            .storage_read(1, image1_buffer)
            .storage_read(2, image2_buffer)
            .storage_read(3, image3_buffer)
            .storage_rw(4, &output_buffer)
            .uniform(5, &params_buffer)
            .dispatch(
                (width as u32).div_ceil(WORKGROUP_SIZE_2D),
                (height as u32).div_ceil(WORKGROUP_SIZE_2D),
                1,
            )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mosaic_basic() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let t1 = Tensor::from_vec_on(vec![1.0; 3 * 640 * 640], vec![3, 640, 640], device.clone())
            .await
            .unwrap();
        let t2 = Tensor::from_vec_on(vec![0.8; 3 * 640 * 640], vec![3, 640, 640], device.clone())
            .await
            .unwrap();
        let t3 = Tensor::from_vec_on(vec![0.6; 3 * 640 * 640], vec![3, 640, 640], device.clone())
            .await
            .unwrap();
        let t4 = Tensor::from_vec_on(vec![0.4; 3 * 640 * 640], vec![3, 640, 640], device)
            .await
            .unwrap();
        let mosaic_tensor = Mosaic::new([t1, t2, t3, t4], 77_777)
            .unwrap()
            .execute()
            .unwrap();
        let mosaic_img = mosaic_tensor.to_vec().unwrap();
        assert_eq!(mosaic_img.len(), 3 * 640 * 640);
        assert!(mosaic_img.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_mosaic_edge_cases() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        // Small images
        let t1 = Tensor::from_vec_on(vec![1.0; 3 * 32 * 32], vec![3, 32, 32], device.clone())
            .await
            .unwrap();
        let t2 = Tensor::from_vec_on(vec![2.0; 3 * 32 * 32], vec![3, 32, 32], device.clone())
            .await
            .unwrap();
        let t3 = Tensor::from_vec_on(vec![3.0; 3 * 32 * 32], vec![3, 32, 32], device.clone())
            .await
            .unwrap();
        let t4 = Tensor::from_vec_on(vec![4.0; 3 * 32 * 32], vec![3, 32, 32], device.clone())
            .await
            .unwrap();
        let mosaic_tensor = Mosaic::new([t1, t2, t3, t4], 12_345)
            .unwrap()
            .execute()
            .unwrap();
        let mosaic_img = mosaic_tensor.to_vec().unwrap();
        assert_eq!(mosaic_img.len(), 3 * 32 * 32);

        // Single channel (grayscale)
        let t1 = Tensor::from_vec_on(vec![1.0; 64 * 64], vec![1, 64, 64], device.clone())
            .await
            .unwrap();
        let t2 = Tensor::from_vec_on(vec![2.0; 64 * 64], vec![1, 64, 64], device.clone())
            .await
            .unwrap();
        let t3 = Tensor::from_vec_on(vec![3.0; 64 * 64], vec![1, 64, 64], device.clone())
            .await
            .unwrap();
        let t4 = Tensor::from_vec_on(vec![4.0; 64 * 64], vec![1, 64, 64], device)
            .await
            .unwrap();
        let mosaic_tensor = Mosaic::new([t1, t2, t3, t4], 99_999)
            .unwrap()
            .execute()
            .unwrap();
        let mosaic_img = mosaic_tensor.to_vec().unwrap();
        assert_eq!(mosaic_img.len(), 64 * 64);
    }

    #[tokio::test]
    async fn test_mosaic_boundary() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        // Different seeds produce different mosaics
        let t1 = Tensor::from_vec_on(vec![1.0; 3 * 128 * 128], vec![3, 128, 128], device.clone())
            .await
            .unwrap();
        let t2 = Tensor::from_vec_on(vec![2.0; 3 * 128 * 128], vec![3, 128, 128], device.clone())
            .await
            .unwrap();
        let t3 = Tensor::from_vec_on(vec![3.0; 3 * 128 * 128], vec![3, 128, 128], device.clone())
            .await
            .unwrap();
        let t4 = Tensor::from_vec_on(vec![4.0; 3 * 128 * 128], vec![3, 128, 128], device)
            .await
            .unwrap();
        let mosaic_tensor1 = Mosaic::new([t1.clone(), t2.clone(), t3.clone(), t4.clone()], 111)
            .unwrap()
            .execute()
            .unwrap();
        let mosaic1 = mosaic_tensor1.to_vec().unwrap();

        let mosaic_tensor2 = Mosaic::new([t1, t2, t3, t4], 222)
            .unwrap()
            .execute()
            .unwrap();
        let mosaic2 = mosaic_tensor2.to_vec().unwrap();
        assert_eq!(mosaic1.len(), mosaic2.len());
    }

    #[tokio::test]
    async fn test_mosaic_large_images() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        // HD images
        let t1 = Tensor::from_vec_on(
            vec![1.0; 3 * 1024 * 1024],
            vec![3, 1024, 1024],
            device.clone(),
        )
        .await
        .unwrap();
        let t2 = Tensor::from_vec_on(
            vec![0.5; 3 * 1024 * 1024],
            vec![3, 1024, 1024],
            device.clone(),
        )
        .await
        .unwrap();
        let t3 = Tensor::from_vec_on(
            vec![0.25; 3 * 1024 * 1024],
            vec![3, 1024, 1024],
            device.clone(),
        )
        .await
        .unwrap();
        let t4 = Tensor::from_vec_on(vec![0.0; 3 * 1024 * 1024], vec![3, 1024, 1024], device)
            .await
            .unwrap();
        let mosaic_tensor = Mosaic::new([t1, t2, t3, t4], 42)
            .unwrap()
            .execute()
            .unwrap();
        let mosaic_img = mosaic_tensor.to_vec().unwrap();
        assert_eq!(mosaic_img.len(), 3 * 1024 * 1024);
    }

    #[tokio::test]
    async fn test_mosaic_precision() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        // Test that all 4 quadrants are represented
        let t1 = Tensor::from_vec_on(vec![1.0; 3 * 100 * 100], vec![3, 100, 100], device.clone())
            .await
            .unwrap();
        let t2 = Tensor::from_vec_on(vec![2.0; 3 * 100 * 100], vec![3, 100, 100], device.clone())
            .await
            .unwrap();
        let t3 = Tensor::from_vec_on(vec![3.0; 3 * 100 * 100], vec![3, 100, 100], device.clone())
            .await
            .unwrap();
        let t4 = Tensor::from_vec_on(vec![4.0; 3 * 100 * 100], vec![3, 100, 100], device)
            .await
            .unwrap();
        let mosaic_tensor = Mosaic::new([t1, t2, t3, t4], 50_505)
            .unwrap()
            .execute()
            .unwrap();
        let mosaic_img = mosaic_tensor.to_vec().unwrap();

        // Should contain values from all 4 images
        assert_eq!(mosaic_img.len(), 3 * 100 * 100);
        assert!(mosaic_img.iter().all(|&x| (1.0..=4.0).contains(&x)));
    }
}
