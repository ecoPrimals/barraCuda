// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU compute operations for FHE Inverse Number Theoretic Transform
//!
//! This module contains the GPU execution logic for INTT transformation,
//! including bit-reversal, butterfly stages, and final scaling.

use super::FheIntt;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

const INTT_SHADER: &str = include_str!("../fhe_intt.wgsl");

impl FheIntt {
    /// Execute INTT transformation
    /// Returns a new tensor containing the coefficient-domain representation.
    /// ## Algorithm
    /// 1. Bit-reversal permutation
    /// 2. log₂(N) butterfly stages (using inverse twiddle factors)
    /// 3. Scale by N^(-1) mod q
    /// ## Complexity
    /// - Time: O(N log N)
    /// - Space: O(N) temporary buffers
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input().device();

        let buffer_size = self.degree() as u64 * 2 * std::mem::size_of::<u32>() as u64;

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("INTT Output Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let intermediate_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("INTT Intermediate Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let inv_twiddle_data: Vec<u32> = self
            .inv_twiddle_factors()
            .iter()
            .flat_map(|&factor| vec![(factor & 0xFFFF_FFFF) as u32, (factor >> 32) as u32])
            .collect();

        let inv_twiddle_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("INTT Inverse Twiddle Factors"),
                    contents: bytemuck::cast_slice(&inv_twiddle_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct InttParams {
            degree: u32,
            modulus_lo: u32,
            modulus_hi: u32,
            barrett_mu_lo: u32,
            barrett_mu_hi: u32,
            inv_root_lo: u32,
            inv_root_hi: u32,
            stage: u32,
        }

        let params = InttParams {
            degree: self.degree(),
            modulus_lo: (self.modulus() & 0xFFFF_FFFF) as u32,
            modulus_hi: (self.modulus() >> 32) as u32,
            barrett_mu_lo: (self.barrett_mu() & 0xFFFF_FFFF) as u32,
            barrett_mu_hi: (self.barrett_mu() >> 32) as u32,
            inv_root_lo: (self.inv_root_of_unity() & 0xFFFF_FFFF) as u32,
            inv_root_hi: (self.inv_root_of_unity() >> 32) as u32,
            stage: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("INTT Params (Bit Reverse)"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "INTT Bit Reverse")
            .shader(INTT_SHADER, "bit_reverse")
            .storage_read(0, self.input().buffer())
            .storage_rw(1, &intermediate_buffer)
            .storage_read(2, &inv_twiddle_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(self.degree())
            .submit()?;

        let num_stages = (self.degree() as f32).log2() as u32;
        let mut current_input = &intermediate_buffer;
        let mut current_output = &output_buffer;

        for stage in 0..num_stages {
            let stage_params = InttParams { stage, ..params };

            let stage_params_buffer =
                device
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("INTT Params (Stage {stage})")),
                        contents: bytemuck::bytes_of(&stage_params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            ComputeDispatch::new(device, "INTT Butterfly")
                .shader(INTT_SHADER, "main")
                .storage_read(0, current_input)
                .storage_rw(1, current_output)
                .storage_read(2, &inv_twiddle_buffer)
                .uniform(3, &stage_params_buffer)
                .dispatch_1d(self.degree() / 2)
                .submit()?;

            std::mem::swap(&mut current_input, &mut current_output);
        }

        let butterfly_result_buffer = current_input;

        let scaled_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("INTT Scaled Output"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let scale_params = InttParams {
            inv_root_lo: (self.inv_n() & 0xFFFF_FFFF) as u32,
            inv_root_hi: (self.inv_n() >> 32) as u32,
            ..params
        };

        let scale_params_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("INTT Scaling Params"),
                    contents: bytemuck::bytes_of(&scale_params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        ComputeDispatch::new(device, "INTT Scale")
            .shader(INTT_SHADER, "scale_by_n")
            .storage_read(0, butterfly_result_buffer)
            .storage_rw(1, &scaled_buffer)
            .storage_read(2, &inv_twiddle_buffer)
            .uniform(3, &scale_params_buffer)
            .dispatch_1d(self.degree())
            .submit()?;

        Ok(Tensor::from_buffer(
            scaled_buffer,
            vec![self.degree() as usize * 2],
            device.clone(),
        ))
    }
}
