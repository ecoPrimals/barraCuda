// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bind group layout builders for sparse GPU operations.
//!
//! Single responsibility: BGL creation for SpMV, dot, reduce, CG update steps,
//! and preconditioner. Shared by CG and BiCGSTAB solvers.

use crate::device::WgpuDevice;
use std::sync::Arc;

/// Bind group layout builders for common sparse operations
pub struct SparseBindGroupLayouts;

impl SparseBindGroupLayouts {
    /// Helper to create a storage buffer entry (read-only)
    fn storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    /// Helper to create a storage buffer entry (read-write)
    fn storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    /// Helper to create a uniform buffer entry
    fn uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    /// SpMV: values, col_idx, row_ptr, x, y, params
    pub fn spmv(device: &Arc<WgpuDevice>) -> wgpu::BindGroupLayout {
        device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SpMV BGL"),
                entries: &[
                    Self::storage_ro(0), // values
                    Self::storage_ro(1), // col_indices
                    Self::storage_ro(2), // row_ptr
                    Self::storage_ro(3), // x
                    Self::storage_rw(4), // y
                    Self::uniform(5),    // params
                ],
            })
    }

    /// Dot product: a, b, partial_sums, params
    pub fn dot(device: &Arc<WgpuDevice>) -> wgpu::BindGroupLayout {
        device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Dot BGL"),
                entries: &[
                    Self::storage_ro(0), // a
                    Self::storage_ro(1), // b
                    Self::storage_rw(2), // partial_sums
                    Self::uniform(3),    // params
                ],
            })
    }

    /// Final reduction: partial_sums, result, params
    pub fn reduce(device: &Arc<WgpuDevice>) -> wgpu::BindGroupLayout {
        device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Reduce BGL"),
                entries: &[
                    Self::storage_ro(0), // partial_sums
                    Self::storage_rw(1), // result
                    Self::uniform(2),    // params
                ],
            })
    }

    /// AXPY: x, y, params (alpha is in params struct)
    pub fn axpy(device: &Arc<WgpuDevice>) -> wgpu::BindGroupLayout {
        device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("AXPY BGL"),
                entries: &[
                    Self::storage_ro(0), // x
                    Self::storage_rw(1), // y (y = y + alpha*x)
                    Self::uniform(2),    // params (includes alpha)
                ],
            })
    }

    /// CG update xr: x, r, p, Ap, alpha, params (all read_write for consistency)
    pub fn cg_update_xr(device: &Arc<WgpuDevice>) -> wgpu::BindGroupLayout {
        device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("CG update_xr BGL"),
                entries: &[
                    Self::storage_rw(0), // x
                    Self::storage_rw(1), // r
                    Self::storage_rw(2), // p
                    Self::storage_rw(3), // Ap
                    Self::storage_rw(4), // alpha
                    Self::uniform(5),    // params
                ],
            })
    }

    /// CG update p: r, p, beta, params (all read_write for consistency)
    pub fn cg_update_p(device: &Arc<WgpuDevice>) -> wgpu::BindGroupLayout {
        device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("CG update_p BGL"),
                entries: &[
                    Self::storage_rw(0), // r (or z for preconditioned)
                    Self::storage_rw(1), // p
                    Self::storage_rw(2), // beta
                    Self::uniform(3),    // params
                ],
            })
    }

    /// Compute alpha: rz, pAp, alpha (all read_write for consistency)
    pub fn compute_alpha(device: &Arc<WgpuDevice>) -> wgpu::BindGroupLayout {
        device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute alpha BGL"),
                entries: &[
                    Self::storage_rw(0), // rz
                    Self::storage_rw(1), // pAp
                    Self::storage_rw(2), // alpha
                ],
            })
    }

    /// Compute beta: rz_new, rz, beta (all read_write for consistency)
    pub fn compute_beta(device: &Arc<WgpuDevice>) -> wgpu::BindGroupLayout {
        device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute beta BGL"),
                entries: &[
                    Self::storage_rw(0), // rz_new
                    Self::storage_rw(1), // rz
                    Self::storage_rw(2), // beta
                ],
            })
    }

    /// Preconditioner: r, diag, z, params
    pub fn precond(device: &Arc<WgpuDevice>) -> wgpu::BindGroupLayout {
        device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Precond BGL"),
                entries: &[
                    Self::storage_ro(0), // r
                    Self::storage_ro(1), // diag (M⁻¹)
                    Self::storage_rw(2), // z
                    Self::uniform(3),    // params
                ],
            })
    }
}
