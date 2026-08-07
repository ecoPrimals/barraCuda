// SPDX-License-Identifier: AGPL-3.0-or-later
//! `CutMix` data augmentation operation
//!
//! `CutMix`: Cuts and pastes patches between training images
//! Improves model robustness and generalization

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};

const SHADER_F64: &str = include_str!("../shaders/augmentation/cutmix_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CutMixParams {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
    cut_x: u32,
    cut_y: u32,
    cut_w: u32,
    cut_h: u32,
    mix_idx: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
    _pad6: u32,
    _pad7: u32,
}

/// `CutMix` data augmentation operation
pub struct CutMix {
    input: Tensor,
    cut_x: u32,
    cut_y: u32,
    cut_w: u32,
    cut_h: u32,
    mix_idx: u32,
}

impl CutMix {
    /// Create `CutMix` operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(
        input: Tensor,
        cut_x: u32,
        cut_y: u32,
        cut_w: u32,
        cut_h: u32,
        mix_idx: u32,
    ) -> Result<Self> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(BarracudaError::invalid_op(
                "cutmix",
                format!("input must be 4D [B, C, H, W], got shape {shape:?}"),
            ));
        }

        let batch_size = shape[0];
        let _ = shape[1];
        let height = shape[2];
        let width = shape[3];

        if mix_idx >= batch_size as u32 {
            return Err(BarracudaError::invalid_op(
                "cutmix",
                format!("mix_idx {mix_idx} must be less than batch_size {batch_size}"),
            ));
        }

        if cut_x + cut_w > width as u32 || cut_y + cut_h > height as u32 {
            return Err(BarracudaError::invalid_op(
                "cutmix",
                format!(
                    "cut region [{cut_x}, {cut_y}] + [{cut_w}, {cut_h}] exceeds image size [{width}, {height}]"
                ),
            ));
        }

        Ok(Self {
            input,
            cut_x,
            cut_y,
            cut_w,
            cut_h,
            mix_idx,
        })
    }

    /// Execute `CutMix` on tensor
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
        let params = CutMixParams {
            batch_size: batch_size as u32,
            channels: channels as u32,
            height: height as u32,
            width: width as u32,
            cut_x: self.cut_x,
            cut_y: self.cut_y,
            cut_w: self.cut_w,
            cut_h: self.cut_h,
            mix_idx: self.mix_idx,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
            _pad4: 0,
            _pad5: 0,
            _pad6: 0,
            _pad7: 0,
        };

        let params_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CutMix Params"),
            size: std::mem::size_of::<CutMixParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        let workgroups_x = (width as u32).div_ceil(8);
        let workgroups_y = (height as u32).div_ceil(8);
        let workgroups_z = batch_size as u32;

        ComputeDispatch::new(device, "CutMix")
            .shader(SHADER_F64, "main")
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
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

    #[tokio::test]
    async fn test_cutmix_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        // Create 2x3x4x4 batch (batch=2, channels=3, height=4, width=4)
        let input = Tensor::from_vec_on(vec![1.0; 2 * 3 * 4 * 4], vec![2, 3, 4, 4], device)
            .await
            .unwrap();

        let output = CutMix::new(input, 1, 1, 2, 2, 1)
            .unwrap()
            .execute()
            .unwrap();
        let result = output.to_vec().unwrap();

        assert_eq!(result.len(), 2 * 3 * 4 * 4);
    }
}
