// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU compute operations for FHE Number Theoretic Transform
//!
//! This module contains the GPU execution logic for NTT transformation,
//! including bit-reversal and butterfly stages.

use super::FheNtt;
use crate::error::Result;
use crate::tensor::Tensor;

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

        // Buffer size: degree * 2 u32s (for u64 coefficients)
        let buffer_size = self.degree() as u64 * 2 * std::mem::size_of::<u32>() as u64;

        // Create output buffer
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NTT Output Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create intermediate buffer (for ping-pong between stages)
        let intermediate_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NTT Intermediate Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create twiddle factors buffer
        let twiddle_data: Vec<u32> = self
            .twiddle_factors()
            .iter()
            .flat_map(|&factor| {
                // Split u64 into two u32s (little-endian)
                vec![(factor & 0xFFFF_FFFF) as u32, (factor >> 32) as u32]
            })
            .collect();

        let twiddle_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("NTT Twiddle Factors"),
                contents: bytemuck::cast_slice(&twiddle_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Create params buffer (will be updated per stage)
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

        // Command encoder
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("NTT Command Encoder"),
        });

        // ============================================================
        // Pass 1: Bit-reversal permutation
        // ============================================================

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("NTT Params (Bit Reverse)"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("NTT Bit Reverse Bind Group"),
            layout: self.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input().buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: intermediate_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: twiddle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("NTT Bit Reverse Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(self.pipeline_bit_reverse());
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);

            // Dispatch: one thread per coefficient
            let workgroup_size = 256u32;
            let num_workgroups = self.degree().div_ceil(workgroup_size);
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        // Submit bit-reversal pass before butterfly stages
        device.submit_commands(std::iter::once(encoder.finish()));

        // ============================================================
        // Pass 2-N: Butterfly stages (log₂(N) stages)
        // ============================================================

        let num_stages = (self.degree() as f32).log2() as u32;
        let mut current_input = &intermediate_buffer;
        let mut current_output = &output_buffer;

        // Submit each stage separately to ensure sequential execution
        for stage in 0..num_stages {
            // Create new encoder for this stage (ensures sequential execution)
            let mut stage_encoder =
                device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("NTT Stage {stage} Encoder")),
                });

            // Update params for this stage
            let stage_params = NttParams { stage, ..params };

            let stage_params_buffer =
                device
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("NTT Params (Stage {stage})")),
                        contents: bytemuck::bytes_of(&stage_params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let stage_bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("NTT Butterfly Bind Group (Stage {stage})")),
                layout: self.bind_group_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: current_input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: current_output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: twiddle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: stage_params_buffer.as_entire_binding(),
                    },
                ],
            });

            {
                let mut compute_pass =
                    stage_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(&format!("NTT Butterfly Pass (Stage {stage})")),
                        timestamp_writes: None,
                    });

                compute_pass.set_pipeline(self.pipeline_butterfly());
                compute_pass.set_bind_group(0, Some(&stage_bind_group), &[]);

                // Dispatch: one thread per butterfly (N/2 butterflies per stage)
                let num_butterflies = self.degree() / 2;
                let workgroup_size = 256u32;
                let num_workgroups = num_butterflies.div_ceil(workgroup_size);
                compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
            }

            // Submit THIS stage before moving to next
            device.submit_commands(std::iter::once(stage_encoder.finish()));

            // Ping-pong buffers for next stage
            std::mem::swap(&mut current_input, &mut current_output);
        }

        // After each stage we swap, so current_input always references the last written buffer.
        let final_buffer = if std::ptr::eq(current_input, std::ptr::from_ref(&intermediate_buffer))
        {
            intermediate_buffer
        } else {
            output_buffer
        };

        // Create result tensor (data stays on GPU)
        Ok(Tensor::from_buffer(
            final_buffer,
            vec![self.degree() as usize * 2], // Shape: [degree * 2] (u32 pairs)
            device.clone(),
        ))
    }
}
