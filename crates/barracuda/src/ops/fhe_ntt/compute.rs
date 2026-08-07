// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU compute operations for FHE Number Theoretic Transform
//!
//! This module contains the GPU execution logic for NTT transformation,
//! including bit-reversal and butterfly stages.

use super::FheNtt;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

const NTT_SHADER: &str = include_str!("../fhe_ntt.wgsl");

impl FheNtt {
    /// Execute NTT transformation
    /// Returns a new tensor containing the NTT-domain representation.
    /// The output can be used for fast polynomial multiplication.
    /// ## Algorithm
    /// 1. Bit-reversal permutation (preprocessing)
    /// 2. log₂(N) butterfly stages (Cooley-Tukey FFT)
    /// 3. Each stage processes N/2 butterflies in parallel
    /// ## Complexity
    /// - Time: O(N log N)
    /// - Space: O(N) temporary buffer
    /// - GPU parallelism: N/2 threads per stage
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input().device();

        let buffer_size = self.degree() as u64 * 2 * std::mem::size_of::<u32>() as u64;

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NTT Output Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let intermediate_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NTT Intermediate Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let twiddle_data: Vec<u32> = self
            .twiddle_factors()
            .iter()
            .flat_map(|&factor| vec![(factor & 0xFFFF_FFFF) as u32, (factor >> 32) as u32])
            .collect();

        let twiddle_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("NTT Twiddle Factors"),
                contents: bytemuck::cast_slice(&twiddle_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct NttParams {
            degree: u32,
            modulus_lo: u32,
            modulus_hi: u32,
            barrett_mu_lo: u32,
            barrett_mu_hi: u32,
            root_of_unity_lo: u32,
            root_of_unity_hi: u32,
            stage: u32,
        }

        let params = NttParams {
            degree: self.degree(),
            modulus_lo: (self.modulus() & 0xFFFF_FFFF) as u32,
            modulus_hi: (self.modulus() >> 32) as u32,
            barrett_mu_lo: (self.barrett_mu() & 0xFFFF_FFFF) as u32,
            barrett_mu_hi: (self.barrett_mu() >> 32) as u32,
            root_of_unity_lo: (self.root_of_unity() & 0xFFFF_FFFF) as u32,
            root_of_unity_hi: (self.root_of_unity() >> 32) as u32,
            stage: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("NTT Params (Bit Reverse)"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "NTT Bit Reverse")
            .shader(NTT_SHADER, "bit_reverse")
            .storage_read(0, self.input().buffer())
            .storage_rw(1, &intermediate_buffer)
            .storage_read(2, &twiddle_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(self.degree())
            .submit()?;

        let num_stages = (self.degree() as f32).log2() as u32;
        let mut current_input = &intermediate_buffer;
        let mut current_output = &output_buffer;

        for stage in 0..num_stages {
            let stage_params = NttParams { stage, ..params };

            let stage_params_buffer =
                device
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("NTT Params (Stage {stage})")),
                        contents: bytemuck::bytes_of(&stage_params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            ComputeDispatch::new(device, "NTT Butterfly")
                .shader(NTT_SHADER, "main")
                .storage_read(0, current_input)
                .storage_rw(1, current_output)
                .storage_read(2, &twiddle_buffer)
                .uniform(3, &stage_params_buffer)
                .dispatch_1d(self.degree() / 2)
                .submit()?;

            std::mem::swap(&mut current_input, &mut current_output);
        }

        let final_buffer = if std::ptr::eq(current_input, std::ptr::from_ref(&intermediate_buffer))
        {
            intermediate_buffer
        } else {
            output_buffer
        };

        Ok(Tensor::from_buffer(
            final_buffer,
            vec![self.degree() as usize * 2],
            device.clone(),
        ))
    }
}
