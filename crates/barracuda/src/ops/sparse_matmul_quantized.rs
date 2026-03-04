// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sparse matrix multiplication with quantization
//!
//! Performs matrix multiplication with sparse matrices and quantized values.
//! Critical for neuromorphic and edge computing where memory/power are constrained.
//!
//! # Neuromorphic Computing
//!
//! Sparse quantized operations are essential for NPU efficiency:
//! - Sparse matrices: Only store non-zero values
//! - Quantization: Use int8 instead of fp32 (4x memory savings)
//! - Combined: Massive efficiency gains for SNNs
//!
//! # Storage Format
//!
//! **COO (Coordinate) Format**:
//! - Values: Non-zero elements
//! - Rows: Row indices
//! - Cols: Column indices
//!
//! # Example
//!
//! ```no_run
//! use barracuda::sparse_matmul_quantized;
//! use barracuda::prelude::WgpuDevice;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let device = WgpuDevice::new().await?;
//!
//! // Sparse matrix: only non-zero values
//! let values = vec![127, -128, 64];  // int8 values
//! let rows = vec![0, 1, 2];
//! let cols = vec![0, 1, 0];
//! let dense_vec = vec![100, -50];     // int8 input
//!
//! let result = sparse_matmul_quantized(
//!     &device,
//!     &values,
//!     &rows,
//!     &cols,
//!     &dense_vec,
//!     3, // output size
//!     127.0, // scale factor
//! )?;
//! # Ok(())
//! # }
//! ```

use std::borrow::Cow;
use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result as BarracudaResult};

/// Sparse matrix multiply with quantized int8 values
///
/// # Arguments
///
/// * `device` - The `WgpuDevice` (provides device, queue, and readback)
/// * `sparse_values` - Non-zero values (int8)
/// * `sparse_rows` - Row indices
/// * `sparse_cols` - Column indices
/// * `dense_vector` - Dense input vector (int8)
/// * `output_size` - Size of output vector
/// * `scale` - Quantization scale factor
///
/// # Returns
///
/// Dense output vector (fp32, dequantized)
pub fn sparse_matmul_quantized(
    device: &WgpuDevice,
    sparse_values: &[i8],
    sparse_rows: &[u32],
    sparse_cols: &[u32],
    dense_vector: &[i8],
    output_size: u32,
    scale: f32,
) -> BarracudaResult<Vec<f32>> {
    if sparse_values.is_empty() {
        return Err(BarracudaError::InvalidInput {
            message: "Sparse values cannot be empty".to_string(),
        });
    }

    if sparse_values.len() != sparse_rows.len() || sparse_values.len() != sparse_cols.len() {
        return Err(BarracudaError::InvalidInput {
            message: "Sparse arrays must have same length".to_string(),
        });
    }

    let nnz = sparse_values.len() as u32;
    let d = device.device();
    let q = device.queue();

    let shader = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Sparse MatMul Quantized Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
            "sparse_matmul_quantized.wgsl"
        ))),
    });

    // Convert i8 to i32 for GPU (WGSL doesn't have i8)
    let values_i32: Vec<i32> = sparse_values.iter().map(|&x| x as i32).collect();
    let dense_i32: Vec<i32> = dense_vector.iter().map(|&x| x as i32).collect();

    let values_buffer = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Sparse Values"),
        contents: bytemuck::cast_slice(&values_i32),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let rows_buffer = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Sparse Rows"),
        contents: bytemuck::cast_slice(sparse_rows),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let cols_buffer = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Sparse Cols"),
        contents: bytemuck::cast_slice(sparse_cols),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let dense_buffer = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Dense Vector"),
        contents: bytemuck::cast_slice(&dense_i32),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_buffer = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output"),
        size: (output_size * std::mem::size_of::<f32>() as u32) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        nnz: u32,
        output_size: u32,
        scale: f32,
        _padding: u32,
    }

    let params = Params {
        nnz,
        output_size,
        scale,
        _padding: 0,
    };

    let params_buffer = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group_layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Sparse MatMul Layout"),
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

    let bind_group = d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Sparse MatMul Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: values_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rows_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cols_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: dense_buffer.as_entire_binding(),
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

    let pipeline_layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Sparse MatMul Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Sparse MatMul Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "sparse_matmul_quantized",
        cache: None,
        compilation_options: Default::default(),
    });

    let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
        label: Some("Sparse MatMul Encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Sparse MatMul Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        // Deep Debt Evolution: Capability-based dispatch
        // Note: This function uses raw wgpu::Device, so we use device limits for capability awareness
        let limits = d.limits();
        let max_invocations = limits.max_compute_invocations_per_workgroup;
        // Use capability-aware workgroup size, respecting device limits
        // Choose optimal size based on device capabilities: prefer 256 for discrete GPUs,
        // 128 for integrated GPUs, but always respect device limits
        let optimal_wg_size = if max_invocations >= 256 {
            256 // Optimal for discrete GPUs
        } else if max_invocations >= 128 {
            128 // Good for integrated GPUs
        } else {
            max_invocations.max(64) // Fallback: use device max or minimum viable size
        };
        let workgroups = output_size.div_ceil(optimal_wg_size);
        cpass.dispatch_workgroups(workgroups.max(1), 1, 1);
    }

    let staging_buffer = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging"),
        size: (output_size * std::mem::size_of::<f32>() as u32) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(
        &output_buffer,
        0,
        &staging_buffer,
        0,
        (output_size * std::mem::size_of::<f32>() as u32) as u64,
    );
    q.submit(Some(encoder.finish()));

    device.map_staging_buffer::<f32>(&staging_buffer, output_size as usize)
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sparse_matmul_quantized_basic() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let values = vec![127, -64, 32];
        let rows = vec![0, 1, 2];
        let cols = vec![0, 0, 1];
        let dense = vec![10, 20];
        let result =
            sparse_matmul_quantized(device.as_ref(), &values, &rows, &cols, &dense, 3, 1.0)
                .unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1270.0).abs() < 1.0);
        assert!((result[1] - -640.0).abs() < 1.0);
        assert!((result[2] - 640.0).abs() < 1.0);
    }

    #[tokio::test]
    async fn test_sparse_matmul_quantized_edge_cases() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let values = vec![0];
        let rows = vec![0];
        let cols = vec![0];
        let dense = vec![100];
        let result =
            sparse_matmul_quantized(device.as_ref(), &values, &rows, &cols, &dense, 1, 1.0)
                .unwrap();
        assert!(result[0].abs() < 0.1);
    }

    #[tokio::test]
    async fn test_sparse_matmul_quantized_boundary() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let empty: Vec<i8> = vec![];
        assert!(sparse_matmul_quantized(device.as_ref(), &empty, &[], &[], &[1], 1, 1.0).is_err());
    }

    #[tokio::test]
    async fn test_sparse_matmul_quantized_large_tensor() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let values: Vec<i8> = (0..1000).map(|i| (i % 128) as i8).collect();
        let rows: Vec<u32> = (0..1000).map(|i| i % 100).collect();
        let cols: Vec<u32> = (0..1000).map(|i| i % 50).collect();
        let dense: Vec<i8> = (0..50).map(|i| (i % 10) as i8).collect();
        let result =
            sparse_matmul_quantized(device.as_ref(), &values, &rows, &cols, &dense, 100, 1.0)
                .unwrap();
        assert_eq!(result.len(), 100);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_sparse_matmul_quantized_precision() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let values = vec![127, 127];
        let rows = vec![0, 0];
        let cols = vec![0, 1];
        let dense = vec![1, 1];
        let result =
            sparse_matmul_quantized(device.as_ref(), &values, &rows, &cols, &dense, 1, 1.0)
                .unwrap();
        assert!((result[0] - 254.0).abs() < 1.0);
    }
}
