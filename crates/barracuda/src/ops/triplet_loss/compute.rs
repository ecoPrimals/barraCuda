// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU compute operations for Triplet Loss
//!
//! This module contains the GPU execution for triplet loss computation
//! with support for L2 and cosine distance metrics.

use super::{DistanceMetric, TripletLoss, TripletParams};
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

impl TripletLoss {
    /// Execute Triplet loss (GPU distance computation)
    /// **Deep Debt**: Efficient single-pass distance computation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.anchors().device();

        let batch_size = self.anchors().shape()[0];
        let embedding_dim = self.anchors().shape()[1];

        // Create parameters
        let params = TripletParams {
            batch_size: batch_size as u32,
            embedding_dim: embedding_dim as u32,
            margin: self.margin(),
            distance_type: match self.distance_metric() {
                DistanceMetric::L2 => 0,
                DistanceMetric::Cosine => 1,
            },
        };

        let params_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Triplet Loss Params"),
            size: std::mem::size_of::<TripletParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Output buffer (one loss value per sample)
        let output_buffer = device.create_buffer_f32(batch_size)?;

        ComputeDispatch::new(device, "Triplet Loss")
            .shader(Self::shader(), "main")
            .storage_read(0, self.anchors().buffer())
            .storage_read(1, self.positives().buffer())
            .storage_read(2, self.negatives().buffer())
            .storage_rw(3, &output_buffer)
            .uniform(4, &params_buffer)
            .dispatch_1d(batch_size as u32)
            .submit()?;

        // Return output tensor [batch_size]
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size],
            device.clone(),
        ))
    }
}
