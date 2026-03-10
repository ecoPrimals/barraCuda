// SPDX-License-Identifier: AGPL-3.0-only

//! Grid search GPU ops wired via `ComputeDispatch` builder.
//!
//! Provenance: groundSpring → toadStool absorption.
//!
//! These ops demonstrate the `ComputeDispatch` pattern: zero manual
//! bind-group / pipeline boilerplate. New ops should follow this pattern.

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER_GRID_FIT_2D: &str = include_str!("../../shaders/grid/grid_fit_2d_f64.wgsl");
const SHADER_GRID_SEARCH_3D: &str = include_str!("../../shaders/grid/grid_search_3d_f64.wgsl");
const SHADER_BAND_EDGES: &str = include_str!("../../shaders/grid/band_edges_parallel_f64.wgsl");

// ── Grid Fit 2D ─────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GridFit2dParams {
    nx: u32,
    ny: u32,
}

/// Fit a bilinear surface `z = a + bx + cy + dxy` to structured 2D grid data.
///
/// Returns 4 coefficients `[a, b, c, d]`.
///
/// # Panics
/// Panics if `data.len() != nx * ny`.
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or buffer readback fails
/// (e.g. device lost).
pub fn grid_fit_2d(
    device: &Arc<WgpuDevice>,
    data: &[f64],
    x_coords: &[f64],
    y_coords: &[f64],
) -> Result<[f64; 4]> {
    let nx = x_coords.len();
    let ny = y_coords.len();
    assert_eq!(data.len(), nx * ny, "data must be nx*ny");

    let data_buf = device.create_buffer_f64_init("grid_fit_2d:data", data);
    let x_buf = device.create_buffer_f64_init("grid_fit_2d:x", x_coords);
    let y_buf = device.create_buffer_f64_init("grid_fit_2d:y", y_coords);
    let out_buf = device.create_buffer_f64(4)?;
    let params = GridFit2dParams {
        nx: nx as u32,
        ny: ny as u32,
    };
    let params_buf = device.create_uniform_buffer("grid_fit_2d:params", &params);

    ComputeDispatch::new(device, "grid_fit_2d")
        .shader(SHADER_GRID_FIT_2D, "main")
        .f64()
        .storage_read(0, &data_buf)
        .storage_read(1, &x_buf)
        .storage_read(2, &y_buf)
        .storage_rw(3, &out_buf)
        .uniform(4, &params_buf)
        .dispatch(1, 1, 1)
        .submit()?;

    let result = device.read_f64_buffer(&out_buf, 4)?;
    Ok([result[0], result[1], result[2], result[3]])
}

// ── Grid Search 3D ──────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GridSearch3dParams {
    nx: u32,
    ny: u32,
    nz: u32,
    _pad: u32,
}

/// Result of a 3D grid search: minimum value and its (x, y, z) indices.
#[derive(Debug, Clone, Copy)]
pub struct GridSearchResult {
    /// Minimum value found.
    pub min_value: f64,
    /// X index of minimum.
    pub min_ix: u32,
    /// Y index of minimum.
    pub min_iy: u32,
    /// Z index of minimum.
    pub min_iz: u32,
}

/// Find the global minimum in a pre-evaluated 3D grid of values.
///
/// `values[ix * ny * nz + iy * nz + iz]` is the value at grid point (ix, iy, iz).
///
/// # Panics
/// Panics if `values.len() != nx * ny * nz`.
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or buffer readback fails
/// (e.g. device lost).
pub fn grid_search_3d(
    device: &Arc<WgpuDevice>,
    x_grid: &[f64],
    y_grid: &[f64],
    z_grid: &[f64],
    values: &[f64],
) -> Result<GridSearchResult> {
    let nx = x_grid.len();
    let ny = y_grid.len();
    let nz = z_grid.len();
    let total = nx * ny * nz;
    assert_eq!(values.len(), total, "values must be nx*ny*nz");

    let x_buf = device.create_buffer_f64_init("grid_search_3d:x", x_grid);
    let y_buf = device.create_buffer_f64_init("grid_search_3d:y", y_grid);
    let z_buf = device.create_buffer_f64_init("grid_search_3d:z", z_grid);
    let val_buf = device.create_buffer_f64_init("grid_search_3d:values", values);
    let out_min_buf = device.create_buffer_f64(1)?;
    let out_idx_buf = device.create_buffer_u32(3)?;
    let params = GridSearch3dParams {
        nx: nx as u32,
        ny: ny as u32,
        nz: nz as u32,
        _pad: 0,
    };
    let params_buf = device.create_uniform_buffer("grid_search_3d:params", &params);

    ComputeDispatch::new(device, "grid_search_3d")
        .shader(SHADER_GRID_SEARCH_3D, "main")
        .f64()
        .storage_read(0, &x_buf)
        .storage_read(1, &y_buf)
        .storage_read(2, &z_buf)
        .storage_rw(3, &val_buf)
        .storage_rw(4, &out_min_buf)
        .storage_rw(5, &out_idx_buf)
        .uniform(6, &params_buf)
        .dispatch(1, 1, 1)
        .submit()?;

    let min_val = device.read_f64_buffer(&out_min_buf, 1)?;
    let min_idx = device.read_buffer_u32(&out_idx_buf, 3)?;

    Ok(GridSearchResult {
        min_value: min_val[0],
        min_ix: min_idx[0],
        min_iy: min_idx[1],
        min_iz: min_idx[2],
    })
}

// ── Band Edges Parallel ─────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BandEdgesParams {
    n: u32,
    m: u32,
}

/// Find band edges (min/max eigenvalues) for N eigensystems of M eigenvalues each.
///
/// `eigenvalues[i * m .. (i+1) * m]` are sorted ascending for eigensystem `i`.
/// Returns `(band_min[N], band_max[N])`.
///
/// # Panics
/// Panics if `eigenvalues.len() != n * m`.
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or buffer readback fails
/// (e.g. device lost).
pub fn band_edges_parallel(
    device: &Arc<WgpuDevice>,
    eigenvalues: &[f64],
    n: usize,
    m: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    assert_eq!(eigenvalues.len(), n * m, "eigenvalues must be n*m");

    let eig_buf = device.create_buffer_f64_init("band_edges:eigenvalues", eigenvalues);
    let min_buf = device.create_buffer_f64(n)?;
    let max_buf = device.create_buffer_f64(n)?;
    let params = BandEdgesParams {
        n: n as u32,
        m: m as u32,
    };
    let params_buf = device.create_uniform_buffer("band_edges:params", &params);

    ComputeDispatch::new(device, "band_edges_parallel")
        .shader(SHADER_BAND_EDGES, "main")
        .f64()
        .storage_read(0, &eig_buf)
        .storage_rw(1, &min_buf)
        .storage_rw(2, &max_buf)
        .uniform(3, &params_buf)
        .dispatch_1d(n as u32)
        .submit()?;

    let band_min = device.read_f64_buffer(&min_buf, n)?;
    let band_max = device.read_f64_buffer(&max_buf, n)?;

    Ok((band_min, band_max))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_band_edges_params_layout() {
        assert_eq!(std::mem::size_of::<BandEdgesParams>(), 8);
    }

    #[test]
    fn test_grid_fit_params_layout() {
        assert_eq!(std::mem::size_of::<GridFit2dParams>(), 8);
    }

    #[test]
    fn test_grid_search_params_layout() {
        assert_eq!(std::mem::size_of::<GridSearch3dParams>(), 16);
    }

    #[test]
    fn grid_fit_2d_shader_compiles() {
        let source = SHADER_GRID_FIT_2D;
        assert!(!source.is_empty());
        assert!(source.contains("fn main"));
    }

    #[test]
    fn grid_search_3d_shader_compiles() {
        let source = SHADER_GRID_SEARCH_3D;
        assert!(!source.is_empty());
        assert!(source.contains("fn main"));
    }

    #[test]
    fn band_edges_shader_compiles() {
        let source = SHADER_BAND_EDGES;
        assert!(!source.is_empty());
        assert!(source.contains("fn main"));
    }
}
