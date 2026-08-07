// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-Resident Cell-List Construction
//!
//! Three-pass GPU pipeline that builds a spatially sorted particle index table
//! without any CPU readback.  Eliminates the 240 KB readback + 240 KB re-upload
//! every 20 steps that the CPU [`CellList`] requires at N=10,000.
//!
//! # hotSpring Feedback (Feb 19 2026)
//!
//! The CPU cell-list bottleneck:
//!
//! 1. Read all N positions (N × 24 bytes)
//! 2. CPU sorts particles into cells
//! 3. Re-upload sorted positions + cell metadata (N × 24 + Nc × 8 bytes)
//!
//! GPU-resident alternative (this module):
//!
//! | Pass | Shader | Description |
//! |------|--------|-------------|
//! | 1 | `atomic_cell_bin.wgsl` | One thread/particle → atomicAdd cell count |
//! | 2 | `prefix_sum.wgsl` | Parallel exclusive scan → `cell_start` offsets |
//! | 3 | `cell_list_scatter.wgsl` | Each particle scatters its index |
//!
//! All three passes fit in one `queue.submit()`.  The resulting buffers
//! (`cell_start`, `sorted_indices`) remain GPU-resident and can be bound
//! directly by the force kernel.
//!
//! # Output Buffers
//!
//! After [`CellListGpu::build`] completes:
//!
//! - [`CellListGpu::sorted_indices`] — `[N] u32` particle indices sorted by cell
//! - [`CellListGpu::cell_start`] — `[Nc] u32` exclusive prefix sum of cell counts
//! - [`CellListGpu::cell_count`] — `[Nc] u32` number of particles per cell
//!
//! Force kernels iterate:
//! ```wgsl
//! for nc in neighbour_cells(cell_of_i) {
//!     for slot in cell_start[nc] .. cell_start[nc] + cell_count[nc] {
//!         let j = sorted_indices[slot];
//!         // compute force between i and j
//!     }
//! }
//! ```

use crate::device::WgpuDevice;
use crate::device::capabilities::{WORKGROUP_SIZE_1D, WORKGROUP_SIZE_COMPACT};
use crate::device::compute_pipeline::{BatchedComputeDispatch, ComputeDispatch};
use crate::error::Result;
use std::sync::Arc;

/// Atomic cell binning pass (pass 1).
pub const WGSL_ATOMIC_CELL_BIN: &str = include_str!("../../../shaders/misc/atomic_cell_bin.wgsl");

/// Cell list scatter pass (pass 3).
pub const WGSL_CELL_LIST_SCATTER: &str =
    include_str!("../../../shaders/misc/cell_list_scatter.wgsl");

struct GpuBuffers {
    // Input (caller-owned; we borrow them via bind groups)
    // Output
    cell_ids: wgpu::Buffer,       // [N] u32 — particle → cell assignment
    cell_counts: wgpu::Buffer,    // [Nc] u32 — atom: count per cell (pass 1 output)
    cell_start: wgpu::Buffer,     // [Nc] u32 — exclusive prefix sum (pass 2 output)
    write_cursors: wgpu::Buffer,  // [Nc] u32 — per-cell write cursor (pass 3 scratch)
    sorted_indices: wgpu::Buffer, // [N] u32  — sorted particle indices (pass 3 output)
    // Prefix-sum intermediate
    scan_partial: wgpu::Buffer, // [ceil(Nc/256)] u32 — partial scan results
    // Params
    bin_params: wgpu::Buffer,     // uniform for pass 1
    scan_params: wgpu::Buffer,    // uniform for pass 2 (Nc)
    scatter_params: wgpu::Buffer, // uniform for pass 3
}

/// GPU-resident cell-list builder.
///
/// Holds all GPU buffers and compiled pipelines.  Call [`build`] each time
/// the particle positions change and the neighbour list needs rebuilding
/// (typically every 20 MD steps).
pub struct CellListGpu {
    device: Arc<WgpuDevice>,
    n: u32,  // particle count
    nc: u32, // total cell count (mx × my × mz)
    mx: u32,
    my: u32,
    mz: u32,
    bufs: GpuBuffers,
}

impl CellListGpu {
    const BIN_SHADER: &'static str = WGSL_ATOMIC_CELL_BIN;
    const SCAN_SHADER: &'static str = include_str!("../../../shaders/misc/prefix_sum.wgsl");
    const SCATTER_SHADER: &'static str = WGSL_CELL_LIST_SCATTER;

    /// Create a GPU cell-list builder.
    ///
    /// # Arguments
    ///
    /// * `device` — barracuda device wrapper
    /// * `n` — number of particles (fixed for lifetime of this builder)
    /// * `box_l` — simulation box side length `[Lx, Ly, Lz]` in Å
    /// * `cutoff` — force cutoff radius; cell side = cutoff
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(device: Arc<WgpuDevice>, n: usize, box_l: [f64; 3], cutoff: f64) -> Result<Self> {
        let n_u32 = n as u32;
        let mx = ((box_l[0] / cutoff).floor() as u32).max(1);
        let my = ((box_l[1] / cutoff).floor() as u32).max(1);
        let mz = ((box_l[2] / cutoff).floor() as u32).max(1);
        let nc = mx * my * mz;
        let cell_size = (box_l[0] / mx as f64) as f32;

        // ── Buffers ──────────────────────────────────────────────────────────
        let cell_ids = buf(&device, n_u32 as u64 * 4, "cell_ids", false);
        let cell_counts = buf(&device, nc as u64 * 4, "cell_counts", false);
        let cell_start = buf(&device, nc as u64 * 4, "cell_start", false);
        let write_cursors = buf(&device, nc as u64 * 4, "write_cursors", false);
        let sorted_indices = buf(&device, n_u32 as u64 * 4, "sorted_indices", false);
        let scan_partial = buf(
            &device,
            (nc.div_ceil(WORKGROUP_SIZE_1D)) as u64 * 4,
            "scan_partial",
            false,
        );

        // Pass 1 params
        let bin_params_data = [
            n_u32,
            mx,
            my,
            mz,
            (box_l[0] as f32).to_bits(),
            (box_l[1] as f32).to_bits(),
            (box_l[2] as f32).to_bits(),
            cell_size.to_bits(),
        ];
        let bin_params = uniform_buf(&device, u32_bytes(&bin_params_data), "bin_params");

        // Pass 2 params: n = nc, n_groups = ceil(nc / WORKGROUP_SIZE_1D) (matches ScanConfig in WGSL)
        let n_groups = nc.div_ceil(WORKGROUP_SIZE_1D);
        let scan_params_data = [nc, n_groups, 0u32, 0u32];
        let scan_params = uniform_buf(&device, u32_bytes(&scan_params_data), "scan_params");

        // Pass 3 params
        let scatter_params_data = [n_u32, nc, 0u32, 0u32];
        let scatter_params =
            uniform_buf(&device, u32_bytes(&scatter_params_data), "scatter_params");

        let bufs = GpuBuffers {
            cell_ids,
            cell_counts,
            cell_start,
            write_cursors,
            sorted_indices,
            scan_partial,
            bin_params,
            scan_params,
            scatter_params,
        };

        Ok(Self {
            device,
            n: n_u32,
            nc,
            mx,
            my,
            mz,
            bufs,
        })
    }

    /// Rebuild the cell list from current GPU-resident particle positions.
    ///
    /// `positions_buf` must be a STORAGE buffer holding `[N × 3]` f64 values
    /// (interleaved x, y, z).  It is never read back to CPU.
    ///
    /// After this returns, [`sorted_indices`] and [`cell_start`] are ready
    /// for the force kernel's bind group.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn build(&self, positions_buf: &wgpu::Buffer) -> Result<()> {
        // ── Zero cell_counts and write_cursors ───────────────────────────────
        let zeros: Vec<u8> = vec![0u8; self.nc as usize * 4];
        self.device
            .queue
            .write_buffer(&self.bufs.cell_counts, 0, &zeros);
        self.device
            .queue
            .write_buffer(&self.bufs.write_cursors, 0, &zeros);

        let mut batch = BatchedComputeDispatch::new(&self.device);

        batch.push(
            ComputeDispatch::new(&self.device, "cell_bin")
                .shader(Self::BIN_SHADER, "atomic_cell_bin")
                .uniform(0, &self.bufs.bin_params)
                .storage_read(1, positions_buf)
                .storage_rw(2, &self.bufs.cell_counts)
                .storage_rw(3, &self.bufs.cell_ids)
                .dispatch(self.n.div_ceil(WORKGROUP_SIZE_COMPACT), 1, 1),
        )?;

        batch.push(
            ComputeDispatch::new(&self.device, "scan_local")
                .shader(Self::SCAN_SHADER, "local_scan")
                .uniform(0, &self.bufs.scan_params)
                .storage_read(1, &self.bufs.cell_counts)
                .storage_rw(2, &self.bufs.cell_start)
                .storage_rw(3, &self.bufs.scan_partial)
                .dispatch(self.nc.div_ceil(WORKGROUP_SIZE_1D), 1, 1),
        )?;

        batch.push(
            ComputeDispatch::new(&self.device, "scan_add_offsets")
                .shader(Self::SCAN_SHADER, "add_wg_offsets")
                .uniform(0, &self.bufs.scan_params)
                .storage_read(1, &self.bufs.cell_counts)
                .storage_rw(2, &self.bufs.cell_start)
                .storage_rw(3, &self.bufs.scan_partial)
                .dispatch(1, 1, 1),
        )?;

        batch.push(
            ComputeDispatch::new(&self.device, "scatter")
                .shader(Self::SCATTER_SHADER, "cell_list_scatter")
                .uniform(0, &self.bufs.scatter_params)
                .storage_read(1, &self.bufs.cell_ids)
                .storage_read(2, &self.bufs.cell_start)
                .storage_rw(3, &self.bufs.write_cursors)
                .storage_rw(4, &self.bufs.sorted_indices)
                .dispatch(self.n.div_ceil(WORKGROUP_SIZE_COMPACT), 1, 1),
        )?;

        batch.submit()?;
        Ok(())
    }

    /// GPU buffer: `[N] u32` particle indices sorted by cell.
    #[must_use]
    pub fn sorted_indices(&self) -> &wgpu::Buffer {
        &self.bufs.sorted_indices
    }
    /// GPU buffer: `[Nc] u32` exclusive prefix sum (cell start offsets).
    #[must_use]
    pub fn cell_start(&self) -> &wgpu::Buffer {
        &self.bufs.cell_start
    }
    /// GPU buffer: `[Nc] u32` particle count per cell.
    #[must_use]
    pub fn cell_count(&self) -> &wgpu::Buffer {
        &self.bufs.cell_counts
    }
    /// Total number of cells.
    #[must_use]
    pub fn n_cells(&self) -> u32 {
        self.nc
    }
    /// Cell grid dimensions.
    #[must_use]
    pub fn grid(&self) -> (u32, u32, u32) {
        (self.mx, self.my, self.mz)
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn buf(device: &WgpuDevice, size: u64, label: &str, read_back: bool) -> wgpu::Buffer {
    let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    if read_back {
        usage |= wgpu::BufferUsages::COPY_SRC;
    }
    device.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn uniform_buf(device: &WgpuDevice, data: &[u8], label: &str) -> wgpu::Buffer {
    let buf = device.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: data.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    device.queue.write_buffer(&buf, 0, data);
    buf
}

fn u32_bytes(data: &[u32]) -> &[u8] {
    bytemuck::cast_slice(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_grid_calculation() {
        // Verify cell count formula used in new()
        let box_l = [10.0f64, 10.0, 10.0];
        let cutoff = 2.5f64;
        let mx = ((box_l[0] / cutoff).floor() as u32).max(1);
        let my = ((box_l[1] / cutoff).floor() as u32).max(1);
        let mz = ((box_l[2] / cutoff).floor() as u32).max(1);
        assert_eq!((mx, my, mz), (4, 4, 4));
        assert_eq!(mx * my * mz, 64);
    }

    #[test]
    fn test_cell_grid_small_box() {
        // Box smaller than cutoff → 1 cell per dimension
        let box_l = [2.0f64, 2.0, 2.0];
        let cutoff = 2.5f64;
        let mx = ((box_l[0] / cutoff).floor() as u32).max(1);
        assert_eq!(mx, 1);
    }

    #[test]
    fn test_workgroup_sizes() {
        assert_eq!(64u32.div_ceil(WORKGROUP_SIZE_COMPACT), 1);
        assert_eq!(65u32.div_ceil(WORKGROUP_SIZE_COMPACT), 2);
        assert_eq!(10_000u32.div_ceil(WORKGROUP_SIZE_COMPACT), 157);
    }

    #[test]
    fn test_u32_bytes_roundtrip() {
        let data = [42u32, 0, 1, 99];
        let bytes = u32_bytes(&data);
        assert_eq!(bytes.len(), 16);
        let back: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(back, data);
    }
}
