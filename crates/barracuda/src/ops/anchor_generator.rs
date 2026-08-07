// SPDX-License-Identifier: AGPL-3.0-or-later
//! Anchor Generator - Generate anchor boxes
//!
//! Creates anchor boxes for object detection.
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

/// `AnchorGenerator` operation
pub struct AnchorGenerator {
    feature_h: usize,
    feature_w: usize,
    stride: usize,
    sizes: Vec<f32>,
    aspect_ratios: Vec<f32>,
    device: std::sync::Arc<crate::device::WgpuDevice>,
}

impl AnchorGenerator {
    /// Create a new anchor generator operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if `sizes` or `aspect_ratios` is empty.
    pub fn new(
        feature_h: usize,
        feature_w: usize,
        stride: usize,
        sizes: Vec<f32>,
        aspect_ratios: Vec<f32>,
        device: std::sync::Arc<crate::device::WgpuDevice>,
    ) -> Result<Self> {
        if sizes.is_empty() || aspect_ratios.is_empty() {
            return Err(crate::error::BarracudaError::invalid_op(
                "AnchorGenerator",
                "Sizes and aspect_ratios must not be empty",
            ));
        }

        Ok(Self {
            feature_h,
            feature_w,
            stride,
            sizes,
            aspect_ratios,
            device,
        })
    }

    /// Execute the anchor generator operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = &self.device;
        let num_anchors = self.sizes.len() * self.aspect_ratios.len();
        let total_anchors = self.feature_h * self.feature_w * num_anchors;
        let output_size = total_anchors * 4;

        // Create output buffer
        let anchors_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("AnchorGenerator Output"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create buffers for sizes and aspect_ratios
        let sizes_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("AnchorGenerator Sizes"),
                contents: bytemuck::cast_slice(&self.sizes),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let ratios_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("AnchorGenerator Ratios"),
                contents: bytemuck::cast_slice(&self.aspect_ratios),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Create uniform buffer for parameters

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct ParamsInner {
            feature_h: u32,
            feature_w: u32,
            stride: u32,
            num_sizes: u32,
            num_ratios: u32,
            _pad0: u32,
            _pad1: u32,
            _pad2: u32,
        }

        let params = ParamsInner {
            feature_h: self.feature_h as u32,
            feature_w: self.feature_w as u32,
            stride: self.stride as u32,
            num_sizes: self.sizes.len() as u32,
            num_ratios: self.aspect_ratios.len() as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("AnchorGenerator Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "AnchorGenerator")
            .shader(
                include_str!("../shaders/detection/anchor_generator_f64.wgsl"),
                "main",
            )
            .storage_rw(0, &anchors_buffer)
            .uniform(1, &params_buffer)
            .storage_read(2, &sizes_buffer)
            .storage_read(3, &ratios_buffer)
            .dispatch(
                (self.feature_w as u32).div_ceil(16),
                (self.feature_h as u32).div_ceil(16),
                1,
            )
            .submit()?;

        // Read back results
        let output_data = crate::utils::read_buffer(device, &anchors_buffer, output_size)?;

        Ok(Tensor::new(
            output_data,
            vec![total_anchors, 4],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // No longer needed - using Tensor method API

    #[tokio::test]
    async fn test_anchor_generator_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let op =
            AnchorGenerator::new(4, 4, 16, vec![32.0, 64.0], vec![0.5, 1.0, 2.0], device).unwrap();
        let anchors = op.execute().unwrap();
        assert_eq!(anchors.shape(), &[4 * 4 * 2 * 3, 4]); // h*w*sizes*ratios, 4
    }

    #[tokio::test]
    async fn test_anchor_generator_edge_cases() {
        let device = crate::device::test_pool::get_test_device().await;

        // Single feature map location
        let op1 = AnchorGenerator::new(1, 1, 8, vec![16.0], vec![1.0], device.clone()).unwrap();
        let result1 = op1.execute().unwrap();
        let anchors1 = result1.to_vec().unwrap();
        assert_eq!(anchors1.len(), 4); // 4 coordinates
        assert!(anchors1.iter().all(|&x| x.is_finite()));

        // Test with single aspect ratio
        let op2 = AnchorGenerator::new(2, 2, 16, vec![32.0], vec![1.0], device).unwrap();
        let result2 = op2.execute().unwrap();
        let anchors2 = result2.to_vec().unwrap();
        assert_eq!(anchors2.len(), 2 * 2 * 4);
    }

    #[tokio::test]
    async fn test_anchor_generator_boundary() {
        let device = crate::device::test_pool::get_test_device().await;

        // Test with different strides
        let op1 = AnchorGenerator::new(3, 3, 8, vec![16.0], vec![1.0], device.clone()).unwrap();
        let result1 = op1.execute().unwrap();
        let anchors1 = result1.to_vec().unwrap();

        let op2 = AnchorGenerator::new(3, 3, 16, vec![16.0], vec![1.0], device).unwrap();
        let result2 = op2.execute().unwrap();
        let anchors2 = result2.to_vec().unwrap();

        assert!(anchors1.iter().all(|&x| x.is_finite()));
        assert!(anchors2.iter().all(|&x| x.is_finite()));
        // Different strides should produce different anchor positions
        assert_ne!(anchors1, anchors2);

        // Larger stride should produce larger coordinate values
        assert!(anchors2.iter().sum::<f32>() > anchors1.iter().sum::<f32>());
    }

    #[tokio::test]
    async fn test_anchor_generator_large_batch() {
        let device = crate::device::test_pool::get_test_device().await;

        // Large feature map with multiple scales and ratios
        let feature_h = 16;
        let feature_w = 16;
        let sizes = vec![32.0, 64.0, 128.0];
        let ratios = vec![0.5, 1.0, 2.0];

        let op = AnchorGenerator::new(
            feature_h,
            feature_w,
            16,
            sizes.clone(),
            ratios.clone(),
            device,
        )
        .unwrap();
        let result = op.execute().unwrap();
        let anchors = result.to_vec().unwrap();

        assert_eq!(
            anchors.len(),
            feature_h * feature_w * sizes.len() * ratios.len() * 4
        );
        assert!(anchors.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_anchor_generator_precision() {
        let device = crate::device::test_pool::get_test_device().await;

        // Test with known values - single anchor at (0,0)
        let op = AnchorGenerator::new(1, 1, 16, vec![32.0], vec![1.0], device).unwrap();
        let result = op.execute().unwrap();
        let anchors = result.to_vec().unwrap();

        // Center should be at (8, 8) - stride/2
        // Size=32, ratio=1.0 → w=h=32
        // Anchor box: [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
        // = [8-16, 8-16, 8+16, 8+16] = [-8, -8, 24, 24]
        assert!((anchors[0] + 8.0).abs() < 1e-5); // x1
        assert!((anchors[1] + 8.0).abs() < 1e-5); // y1
        assert!((anchors[2] - 24.0).abs() < 1e-5); // x2
        assert!((anchors[3] - 24.0).abs() < 1e-5); // y2
    }
}
