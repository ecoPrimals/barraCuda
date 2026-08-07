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

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
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
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
/// readback fails (e.g. device lost or out of memory).
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
        return Err(BarracudaError::invalid_input(
            "Sparse values cannot be empty",
        ));
    }

    if sparse_values.len() != sparse_rows.len() || sparse_values.len() != sparse_cols.len() {
        return Err(BarracudaError::invalid_input(
            "Sparse arrays must have same length",
        ));
    }

    let nnz = crate::utils::checked_u32(sparse_values.len(), "sparse_matmul nnz")?;
    let d = device.device();
    let q = device.queue();

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

    ComputeDispatch::new(device, "Sparse MatMul Quantized")
        .shader(
            include_str!("sparse_matmul_quantized.wgsl"),
            "sparse_matmul_quantized",
        )
        .storage_read(0, &values_buffer)
        .storage_read(1, &rows_buffer)
        .storage_read(2, &cols_buffer)
        .storage_read(3, &dense_buffer)
        .storage_rw(4, &output_buffer)
        .uniform(5, &params_buffer)
        .dispatch_1d(output_size)
        .submit()?;

    let staging_buffer = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging"),
        size: (output_size * std::mem::size_of::<f32>() as u32) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
        label: Some("Sparse MatMul Readback Encoder"),
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
