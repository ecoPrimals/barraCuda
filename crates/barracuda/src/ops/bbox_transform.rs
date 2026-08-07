// SPDX-License-Identifier: AGPL-3.0-or-later
//! `BBox` Transform - Transform bounding boxes
//!
//! Applies deltas to anchor boxes (object detection).
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

/// `BBoxTransform` operation
pub struct BBoxTransform {
    anchors: Tensor,
    deltas: Tensor,
}

impl BBoxTransform {
    /// Create a new bbox transform operation
    /// # Errors
    /// Returns [`Err`] if anchors is not [N, 4] or deltas shape does not match anchors.
    pub fn new(anchors: Tensor, deltas: Tensor) -> Result<Self> {
        let anchor_shape = anchors.shape();
        let delta_shape = deltas.shape();

        if anchor_shape.len() != 2 || anchor_shape[1] != 4 {
            return Err(crate::error::BarracudaError::invalid_op(
                "BBoxTransform",
                format!("Anchors must be [N, 4], got {anchor_shape:?}"),
            ));
        }

        if delta_shape != anchor_shape {
            return Err(crate::error::BarracudaError::invalid_op(
                "BBoxTransform",
                format!("Deltas must match anchors shape: {delta_shape:?} vs {anchor_shape:?}"),
            ));
        }

        Ok(Self { anchors, deltas })
    }

    /// Execute the bbox transform operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.anchors.device();
        let shape = self.anchors.shape();

        let num_boxes = shape[0];
        let output_size = num_boxes * 4;

        let transformed_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BBoxTransform Output"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            num_boxes: u32,
        }

        let params = Params {
            num_boxes: num_boxes as u32,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("BBoxTransform Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "BBoxTransform")
            .shader(
                include_str!("../shaders/detection/bbox_transform_f64.wgsl"),
                "main",
            )
            .storage_read(0, self.anchors.buffer())
            .storage_read(1, self.deltas.buffer())
            .storage_rw(2, &transformed_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(num_boxes as u32)
            .submit()?;

        let output_data = crate::utils::read_buffer(device, &transformed_buffer, output_size)?;

        Ok(Tensor::new(output_data, vec![num_boxes, 4], device.clone()))
    }
}

impl Tensor {
    /// Transform bounding boxes with deltas
    ///
    /// # Arguments
    /// * `deltas` - Deltas tensor [N, 4] (dx, dy, dw, dh)
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn bbox_transform(self, deltas: Self) -> Result<Self> {
        BBoxTransform::new(self, deltas)?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // No longer needed - using Tensor method API

    #[tokio::test]
    async fn test_bbox_transform_basic() {
        let dev = crate::device::test_pool::get_test_device().await;
        let anchors_data = vec![0.0, 0.0, 10.0, 10.0];
        let deltas_data = vec![0.0, 0.0, 0.0, 0.0]; // Identity transform
        let anchors = Tensor::new(anchors_data, vec![1, 4], dev.clone());
        let deltas = Tensor::new(deltas_data, vec![1, 4], dev);
        let result = anchors.bbox_transform(deltas).unwrap();
        let output = result.to_vec().unwrap();
        assert_eq!(output.len(), 4);
        assert!(output.iter().all(|&x| x.is_finite()));
        // Zero deltas should preserve anchor center
        assert!((output[0] - 0.0).abs() < 0.01);
        assert!((output[2] - 10.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_bbox_transform_edge_cases() {
        let dev = crate::device::test_pool::get_test_device().await;
        // Test with single anchor at origin
        let anchors = Tensor::new(vec![0.0, 0.0, 1.0, 1.0], vec![1, 4], dev.clone());
        let deltas = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![1, 4], dev.clone());
        let result = anchors.bbox_transform(deltas).unwrap();
        let output = result.to_vec().unwrap();
        assert!(output.iter().all(|&x| x.is_finite()));

        // Test with translation only (no scaling)
        let anchors = Tensor::new(vec![10.0, 10.0, 20.0, 20.0], vec![1, 4], dev.clone());
        let deltas = Tensor::new(vec![0.5, 0.5, 0.0, 0.0], vec![1, 4], dev);
        let result = anchors.bbox_transform(deltas).unwrap();
        let output = result.to_vec().unwrap();
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_bbox_transform_boundary() {
        let dev = crate::device::test_pool::get_test_device().await;
        // Test with scaling (exponential deltas)
        let anchors = Tensor::new(vec![0.0, 0.0, 10.0, 10.0], vec![1, 4], dev.clone());
        let deltas = Tensor::new(vec![0.0, 0.0, 0.693, 0.693], vec![1, 4], dev.clone()); // exp(0.693) ≈ 2.0
        let result = anchors.bbox_transform(deltas).unwrap();
        let output = result.to_vec().unwrap();
        assert!(output.iter().all(|&x| x.is_finite()));

        // Width and height should approximately double
        let out_w = output[2] - output[0];
        let out_h = output[3] - output[1];
        assert!(out_w > 15.0); // Should be ~20
        assert!(out_h > 15.0);

        // Test with negative scaling
        let anchors = Tensor::new(vec![0.0, 0.0, 10.0, 10.0], vec![1, 4], dev.clone());
        let deltas = Tensor::new(vec![0.0, 0.0, -0.693, -0.693], vec![1, 4], dev); // exp(-0.693) ≈ 0.5
        let result = anchors.bbox_transform(deltas).unwrap();
        let output = result.to_vec().unwrap();
        let out_w = output[2] - output[0];
        assert!(out_w < 7.0); // Should be ~5
    }

    #[tokio::test]
    async fn test_bbox_transform_large_batch() {
        let dev = crate::device::test_pool::get_test_device().await;
        // Multiple anchors
        let num_boxes = 100;
        let mut anchors_data = Vec::new();
        let mut deltas_data = Vec::new();

        for i in 0..num_boxes {
            let base = (i * 10) as f32;
            anchors_data.extend_from_slice(&[base, base, base + 10.0, base + 10.0]);
            deltas_data.extend_from_slice(&[0.1, 0.1, 0.0, 0.0]);
        }

        let anchors = Tensor::new(anchors_data, vec![num_boxes, 4], dev.clone());
        let deltas = Tensor::new(deltas_data, vec![num_boxes, 4], dev);
        let result = anchors.bbox_transform(deltas).unwrap();
        let output = result.to_vec().unwrap();

        assert_eq!(output.len(), num_boxes * 4);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_bbox_transform_precision() {
        let dev = crate::device::test_pool::get_test_device().await;
        // Test with known values
        // Anchor: [0, 0, 10, 10] → center (5, 5), size (10, 10)
        // Deltas: [0.1, 0.2, 0, 0] → shift center by (1, 2)
        let anchors = Tensor::new(vec![0.0, 0.0, 10.0, 10.0], vec![1, 4], dev.clone());
        let deltas = Tensor::new(vec![0.1, 0.2, 0.0, 0.0], vec![1, 4], dev);
        let result = anchors.bbox_transform(deltas).unwrap();
        let output = result.to_vec().unwrap();

        // New center: (5 + 1, 5 + 2) = (6, 7)
        // New box: [1, 2, 11, 12]
        assert!((output[0] - 1.0).abs() < 0.01);
        assert!((output[1] - 2.0).abs() < 0.01);
        assert!((output[2] - 11.0).abs() < 0.01);
        assert!((output[3] - 12.0).abs() < 0.01);
    }
}
