//! GPU compute operations for FHE Inverse Number Theoretic Transform
//!
//! This module contains the GPU execution logic for INTT transformation,
//! including bit-reversal, butterfly stages, and final scaling.

use super::FheIntt;
use crate::error::Result;
use crate::tensor::Tensor;
use wgpu::util::DeviceExt;

impl FheIntt {
    /// Execute INTT transformation
    ///
    /// Returns a new tensor containing the coefficient-domain representation.
    ///
    /// ## Algorithm
    ///
    /// 1. Bit-reversal permutation
    /// 2. log₂(N) butterfly stages (using inverse twiddle factors)
    /// 3. Scale by N^(-1) mod q
    ///
    /// ## Complexity
    ///
    /// - Time: O(N log N)
    /// - Space: O(N) temporary buffers
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input().device();

        // Buffer size
        let buffer_size = self.degree() as u64 * 2 * std::mem::size_of::<u32>() as u64;

        // Create buffers
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

        // Create inverse twiddle factors buffer
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

        // Create params
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

        // Command encoder
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("INTT Command Encoder"),
            });

        // Pass 1: Bit-reversal
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("INTT Params (Bit Reverse)"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("INTT Bit Reverse Bind Group"),
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
                    resource: inv_twiddle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("INTT Bit Reverse Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(self.pipeline_bit_reverse());
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_size = 256u32;
            let num_workgroups = self.degree().div_ceil(workgroup_size);
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        // Submit bit-reversal pass before butterfly stages
        device.submit_and_poll(std::iter::once(encoder.finish()));

        // Pass 2-N: Butterfly stages (submit each separately for sequential execution)
        let num_stages = (self.degree() as f32).log2() as u32;
        let mut current_input = &intermediate_buffer;
        let mut current_output = &output_buffer;

        for stage in 0..num_stages {
            // Create separate encoder for each stage
            let mut stage_encoder =
                device
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some(&format!("INTT Stage {stage} Encoder")),
                    });

            let stage_params = InttParams { stage, ..params };

            let stage_params_buffer =
                device
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("INTT Params (Stage {stage})")),
                        contents: bytemuck::bytes_of(&stage_params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let stage_bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("INTT Butterfly Bind Group (Stage {stage})")),
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
                        resource: inv_twiddle_buffer.as_entire_binding(),
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
                        label: Some(&format!("INTT Butterfly Pass (Stage {stage})")),
                        timestamp_writes: None,
                    });

                compute_pass.set_pipeline(self.pipeline_butterfly());
                compute_pass.set_bind_group(0, &stage_bind_group, &[]);

                let num_butterflies = self.degree() / 2;
                let workgroup_size = 256u32;
                let num_workgroups = num_butterflies.div_ceil(workgroup_size);
                compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
            }

            // Submit this stage before next
            device.submit_and_poll(std::iter::once(stage_encoder.finish()));

            std::mem::swap(&mut current_input, &mut current_output);
        }

        // After each stage we swap, so current_input always references the last written buffer.
        let butterfly_result_buffer = current_input;

        // ============================================================
        // Pass N+1: Scaling by N^(-1) mod q
        // ============================================================

        // Create scaled output buffer
        let scaled_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("INTT Scaled Output"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Update params with inv_n (reuse root_of_unity fields)
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

        let scale_bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("INTT Scaling Bind Group"),
            layout: self.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: butterfly_result_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scaled_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: inv_twiddle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scale_params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("INTT Scaling Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("INTT Scaling Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(self.pipeline_scale());
            compute_pass.set_bind_group(0, &scale_bind_group, &[]);

            let workgroup_size = 256u32;
            let num_workgroups = self.degree().div_ceil(workgroup_size);
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        // Submit scaling pass
        device.submit_and_poll(std::iter::once(encoder.finish()));

        // Create result tensor
        Ok(Tensor::from_buffer(
            scaled_buffer,
            vec![self.degree() as usize * 2],
            device.clone(),
        ))
    }
}
