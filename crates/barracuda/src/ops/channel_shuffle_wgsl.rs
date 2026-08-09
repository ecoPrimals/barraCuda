// SPDX-License-Identifier: AGPL-3.0-or-later
//! Channel Shuffle - Rearrange CNN channels - Pure WGSL
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (GPU-optimized)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//!
//! ## Algorithm
//!
//! Rearranges channels for efficient group convolutions (ShuffleNet):
//! ```text
//! Input:  [B, C, H, W] with G groups
//! Output: [B, C, H, W] with shuffled channels
//!
//! Shuffle pattern: c = g*cpg + i → c' = i*G + g
//! where g = group, i = index in group, cpg = channels per group
//! ```

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Shuffle channels for group convolutions (`ShuffleNet`).
pub struct ChannelShuffle {
    input: Tensor,
    groups: usize,
}

impl ChannelShuffle {
    /// Create a channel shuffle. Channels must be divisible by groups.
    #[must_use]
    pub fn new(input: Tensor, groups: usize) -> Self {
        Self { input, groups }
    }

    /// Execute channel shuffle on GPU.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();

        // Expect 4D tensor [batch, channels, height, width]
        if shape.len() != 4 {
            return Err(crate::error::BarracudaError::invalid_input(format!(
                "ChannelShuffle requires 4D tensor [N, C, H, W], got {}D {:?}",
                shape.len(),
                shape
            )));
        }

        let batch_size = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];

        if !channels.is_multiple_of(self.groups) {
            return Err(crate::error::BarracudaError::InvalidShape {
                expected: vec![batch_size, channels, height, width],
                actual: shape.to_vec(),
            });
        }

        let channels_per_group = channels / self.groups;
        let total_size = self.input.len();

        let output_buffer = device.create_buffer_f32(total_size)?;

        // Create params buffer
        let params_data = [
            batch_size as u32,
            channels as u32,
            height as u32,
            width as u32,
            self.groups as u32,
            channels_per_group as u32,
        ];
        let params_buffer = device.create_uniform_buffer("Params", &params_data);
        let input_buffer = self.input.buffer();

        ComputeDispatch::new(device, "ChannelShuffle")
            .shader(
                include_str!("../shaders/tensor/channel_shuffle_f64.wgsl"),
                "main",
            )
            .storage_read(0, input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(total_size as u32)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            shape.to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Shuffle channels into groups. Expects [B, C, H, W], C divisible by groups.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn channel_shuffle_wgsl(self, groups: usize) -> Result<Self> {
        ChannelShuffle::new(self, groups).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_channel_shuffle_simple() {
        let device = crate::device::test_pool::get_test_device().await;
        // Shape: [1, 4, 2, 2] with 2 groups
        // Channels 0,1 in group 0, channels 2,3 in group 1
        let input_data = vec![
            // Channel 0
            1.0, 2.0, 3.0, 4.0, // Channel 1
            5.0, 6.0, 7.0, 8.0, // Channel 2
            9.0, 10.0, 11.0, 12.0, // Channel 3
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = Tensor::from_vec_on(input_data, vec![1, 4, 2, 2], device)
            .await
            .unwrap();

        let result = input.channel_shuffle_wgsl(2).unwrap();
        let output = result.to_vec().unwrap();

        // After shuffle with 2 groups (cpg=2):
        // c=0 (g=0,i=0) → c'=0*2+0=0
        // c=1 (g=0,i=1) → c'=1*2+0=2
        // c=2 (g=1,i=0) → c'=0*2+1=1
        // c=3 (g=1,i=1) → c'=1*2+1=3
        // New order: [0, 2, 1, 3] → [ch0, ch2, ch1, ch3]

        // Verify first spatial position (0,0) of each channel
        assert_eq!(output[0], 1.0); // ch0[0,0]
        assert_eq!(output[4], 9.0); // ch2[0,0]
        assert_eq!(output[8], 5.0); // ch1[0,0]
        assert_eq!(output[12], 13.0); // ch3[0,0]
    }
}
