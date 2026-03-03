// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU compute operations for Transpose
//!
//! This module contains GPU pipeline setup, buffer creation, and execution
//! logic for both 2D and N-D transpose operations.

use super::TransposeParams2D;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;
use wgpu::util::DeviceExt;

/// Execute transpose operation
pub fn execute_transpose(input: Tensor, permutation: Option<Vec<usize>>) -> Result<Tensor> {
    let device = input.device().clone();
    let shape = input.shape().to_vec();
    let num_dims = shape.len();
    let size = input.len();

    // Determine if 2D or N-D
    let is_2d = num_dims == 2 && permutation.is_none();

    if is_2d {
        // Optimized 2D transpose
        execute_2d(input, &device, &shape, size)
    } else {
        // N-D transpose with permutation
        let perm = permutation.unwrap_or_else(|| {
            // Default: swap last two dimensions
            let mut p: Vec<usize> = (0..num_dims).collect();
            if num_dims >= 2 {
                p.swap(num_dims - 2, num_dims - 1);
            }
            p
        });
        execute_nd(input, &device, &shape, size, perm)
    }
}

fn execute_2d(
    input: Tensor,
    device: &std::sync::Arc<crate::device::WgpuDevice>,
    shape: &[usize],
    size: usize,
) -> Result<Tensor> {
    let rows = shape[0] as u32;
    let cols = shape[1] as u32;

    // Create output buffer
    let output_buffer = device.create_buffer_f32(size)?;

    // Create params buffer
    let params_2d = TransposeParams2D {
        rows,
        cols,
        _padding: [0, 0],
    };
    let params_buffer = device.create_uniform_buffer("Transpose Params 2D", &params_2d);

    // S-16: 2D tiled transpose uses @workgroup_size(16, 16) in the shader.
    const TILE: u32 = 16;
    let workgroups_x = cols.div_ceil(TILE).max(1);
    let workgroups_y = rows.div_ceil(TILE).max(1);

    ComputeDispatch::new(device, "transpose_2d")
        .shader(super::Transpose::wgsl_shader(), "main_2d")
        .storage_read(0, input.buffer())
        .storage_rw(1, &output_buffer)
        .uniform(3, &params_buffer)
        .dispatch(workgroups_x, workgroups_y, 1)
        .submit();

    // Create output tensor with transposed shape
    let new_shape = vec![shape[1], shape[0]];
    Ok(Tensor::from_buffer(
        output_buffer,
        new_shape,
        device.clone(),
    ))
}

fn execute_nd(
    input: Tensor,
    device: &std::sync::Arc<crate::device::WgpuDevice>,
    shape: &[usize],
    size: usize,
    permutation: Vec<usize>,
) -> Result<Tensor> {
    let num_dims = shape.len();

    // Compute output shape
    let output_shape: Vec<usize> = permutation.iter().map(|&idx| shape[idx]).collect();

    // Compute input strides
    let mut input_strides = vec![1; num_dims];
    for i in (0..num_dims - 1).rev() {
        input_strides[i] = input_strides[i + 1] * shape[i + 1];
    }

    // Compute output strides
    let mut output_strides = vec![1; num_dims];
    for i in (0..num_dims - 1).rev() {
        output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
    }

    // Create output buffer
    let output_buffer = device.create_buffer_f32(size)?;

    // Create buffers for shape and stride data
    let input_shape_buffer = device
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Transpose Input Shape"),
            contents: bytemuck::cast_slice(&shape.iter().map(|&x| x as u32).collect::<Vec<_>>()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

    let output_shape_buffer = device
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Transpose Output Shape"),
            contents: bytemuck::cast_slice(
                &output_shape.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            ),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

    let permutation_buffer = device
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Transpose Permutation"),
            contents: bytemuck::cast_slice(
                &permutation.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            ),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

    let input_strides_buffer =
        device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Transpose Input Strides"),
                contents: bytemuck::cast_slice(
                    &input_strides.iter().map(|&x| x as u32).collect::<Vec<_>>(),
                ),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

    let output_strides_buffer =
        device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Transpose Output Strides"),
                contents: bytemuck::cast_slice(
                    &output_strides.iter().map(|&x| x as u32).collect::<Vec<_>>(),
                ),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

    // Create params
    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        total_size: u32,
        num_dims: u32,
        is_2d: u32,
        _padding: u32,
    }

    let params = Params {
        total_size: size as u32,
        num_dims: num_dims as u32,
        is_2d: 0,
        _padding: 0,
    };

    let params_buffer = device.create_uniform_buffer("Transpose Params", &params);

    let caps = DeviceCapabilities::from_device(device);
    let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
    let workgroups = (size as u32).div_ceil(optimal_wg_size);

    ComputeDispatch::new(device, "transpose_nd")
        .shader(super::Transpose::wgsl_shader(), "main_nd")
        .storage_read(0, input.buffer())
        .storage_rw(1, &output_buffer)
        .uniform(2, &params_buffer)
        .storage_read(4, &input_shape_buffer)
        .storage_read(5, &output_shape_buffer)
        .storage_read(6, &permutation_buffer)
        .storage_read(7, &input_strides_buffer)
        .storage_read(8, &output_strides_buffer)
        .dispatch(workgroups.max(1), 1, 1)
        .submit();

    // Create output tensor with transposed shape
    Ok(Tensor::from_buffer(
        output_buffer,
        output_shape,
        std::sync::Arc::clone(device),
    ))
}
