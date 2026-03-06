// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sparse Matrix-Vector Product (CSR format) - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::DeviceCapabilities;
use crate::error::Result;
use crate::tensor::Tensor;

/// GPU shader for f64 sparse matrix-vector product (CSR format).
///
/// Includes `SpMV`, axpy, dot product, scale, copy, diagonal preconditioner,
/// linear combination, and full CG solver kernels — all in f64.
///
/// Entry points: `spmv_f64`, `axpy_f64`, `dot_f64`, `scale_f64`, `copy_f64`,
/// `precond_f64`, `linear_comb_f64`, `final_reduce_f64`, `cg_update_xr`,
/// `cg_update_p`, `compute_alpha`, `compute_beta`.
pub const WGSL_SPARSE_MATVEC_F64: &str = include_str!("../shaders/misc/sparse_matvec_f64.wgsl");

/// Sparse matrix-vector product in CSR (Compressed Sparse Row) format.
pub struct SparseMatVec {
    values: Tensor,
    col_indices: Vec<u32>,
    row_ptrs: Vec<u32>,
    vector: Tensor,
}

impl SparseMatVec {
    /// Create sparse matvec from CSR format (values, `col_indices`, `row_ptrs`) and dense vector.
    #[must_use]
    pub fn new(values: Tensor, col_indices: Vec<u32>, row_ptrs: Vec<u32>, vector: Tensor) -> Self {
        Self {
            values,
            col_indices,
            row_ptrs,
            vector,
        }
    }

    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/misc/sparse_matvec.wgsl")
    }

    /// Execute sparse matrix-vector product.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.values.device();
        let num_rows = self.row_ptrs.len() - 1;

        if num_rows == 0 {
            return Ok(Tensor::new(vec![], vec![0], device.clone()));
        }

        let output_buffer = device.create_buffer_f32(num_rows)?;

        // Create buffers for CSR data
        let col_indices_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("SparseMatVec col_indices"),
                    contents: bytemuck::cast_slice(&self.col_indices),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

        let row_ptrs_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SparseMatVec row_ptrs"),
                contents: bytemuck::cast_slice(&self.row_ptrs),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            num_rows: u32,
        }

        let params = Params {
            num_rows: num_rows as u32,
        };
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SparseMatVec Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("SparseMatVec Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SparseMatVec Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.values.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: col_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: row_ptrs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.vector.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let shader = device.compile_shader(Self::wgsl_shader(), Some("SparseMatVec"));

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("SparseMatVec Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("SparseMatVec Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("SparseMatVec Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SparseMatVec Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            let caps = DeviceCapabilities::from_device(device);
            let workgroups = caps.dispatch_1d(num_rows as u32);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![num_rows],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    async fn get_test_device() -> Option<Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_sparse_matvec() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // 2x3 matrix: [[1, 0, 2], [0, 3, 0]]
        // values: 1, 2, 3
        // col_indices: 0, 2, 1
        // row_ptrs: 0, 2, 3
        let values = vec![1.0f32, 2.0, 3.0];
        let col_indices = vec![0u32, 2, 1];
        let row_ptrs = vec![0u32, 2, 3];
        let vector = vec![1.0f32, 2.0, 3.0];

        let values_tensor = Tensor::new(values, vec![3], device.clone());
        let vector_tensor = Tensor::new(vector, vec![3], device.clone());

        let output = SparseMatVec::new(values_tensor, col_indices, row_ptrs, vector_tensor)
            .execute()
            .unwrap();

        let result = output.to_vec().unwrap();
        // Row 0: 1*1 + 2*3 = 7
        // Row 1: 3*2 = 6
        assert_eq!(result.len(), 2);
        assert!((result[0] - 7.0).abs() < 1e-5);
        assert!((result[1] - 6.0).abs() < 1e-5);
    }
}
