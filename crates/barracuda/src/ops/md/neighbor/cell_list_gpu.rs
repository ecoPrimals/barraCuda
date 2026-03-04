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
//! | 2 | `prefix_sum.wgsl` | Parallel exclusive scan → cell_start offsets |
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
use crate::error::Result;
use std::sync::Arc;

/// Atomic cell binning pass (pass 1).
pub const WGSL_ATOMIC_CELL_BIN: &str = include_str!("../../../shaders/misc/atomic_cell_bin.wgsl");

/// Cell list scatter pass (pass 3).
pub const WGSL_CELL_LIST_SCATTER: &str =
    include_str!("../../../shaders/misc/cell_list_scatter.wgsl");

const BIN_WG: u32 = 64;
const SCAN_WG: u32 = 256; // must match prefix_sum.wgsl workgroup size

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
    // Compiled pipelines
    bin_pl: wgpu::ComputePipeline,
    scan_pass_a_pl: wgpu::ComputePipeline, // local_scan (per-workgroup Blelloch)
    scan_pass_b_pl: wgpu::ComputePipeline, // add_wg_offsets (propagate totals)
    scatter_pl: wgpu::ComputePipeline,
    // Bind group layouts (for rebuild)
    bin_bgl: wgpu::BindGroupLayout,
    scatter_bgl: wgpu::BindGroupLayout,
    scan_bgl: wgpu::BindGroupLayout,
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
    pub fn new(device: Arc<WgpuDevice>, n: usize, box_l: [f64; 3], cutoff: f64) -> Result<Self> {
        let n_u32 = n as u32;
        let mx = ((box_l[0] / cutoff).floor() as u32).max(1);
        let my = ((box_l[1] / cutoff).floor() as u32).max(1);
        let mz = ((box_l[2] / cutoff).floor() as u32).max(1);
        let nc = mx * my * mz;
        let cell_size = (box_l[0] / mx as f64) as f32;

        // ── Shaders ──────────────────────────────────────────────────────────
        let bin_mod = compile(&device, Self::BIN_SHADER, "atomic_cell_bin");
        let scan_mod = compile(&device, Self::SCAN_SHADER, "prefix_scan");
        let scatter_mod = compile(&device, Self::SCATTER_SHADER, "cell_list_scatter");

        // ── Bind group layouts ───────────────────────────────────────────────
        let bin_bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("CellListGpu:bin_bgl"),
                entries: &[
                    uniform_bgl(0),
                    storage_bgl(1, true),
                    storage_bgl(2, false),
                    storage_bgl(3, false),
                ],
            });
        // prefix_sum.wgsl layout (must match exactly):
        //   binding 0 — uniform  ScanConfig  { n, n_groups, _pad, _pad }
        //   binding 1 — storage  flags_in    (read-only input)
        //   binding 2 — storage  scan_out    (read-write, exclusive scan result)
        //   binding 3 — storage  wg_sums     (read-write, per-workgroup totals)
        let scan_bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("CellListGpu:scan_bgl"),
                entries: &[
                    uniform_bgl(0),
                    storage_bgl(1, true),
                    storage_bgl(2, false),
                    storage_bgl(3, false),
                ],
            });
        let scatter_bgl =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("CellListGpu:scatter_bgl"),
                    entries: &[
                        uniform_bgl(0),
                        storage_bgl(1, true),
                        storage_bgl(2, true),
                        storage_bgl(3, false),
                        storage_bgl(4, false),
                    ],
                });

        // ── Pipelines ────────────────────────────────────────────────────────
        let bin_pl = make_pipeline(&device, &bin_mod, &bin_bgl, "atomic_cell_bin", "bin_pl");
        let scan_pass_a_pl = make_pipeline(
            &device,
            &scan_mod,
            &scan_bgl,
            "local_scan",
            "scan_pass_a_pl",
        );
        let scan_pass_b_pl = make_pipeline(
            &device,
            &scan_mod,
            &scan_bgl,
            "add_wg_offsets",
            "scan_pass_b_pl",
        );
        let scatter_pl = make_pipeline(
            &device,
            &scatter_mod,
            &scatter_bgl,
            "cell_list_scatter",
            "scatter_pl",
        );

        // ── Buffers ──────────────────────────────────────────────────────────
        let cell_ids = buf(&device, n_u32 as u64 * 4, "cell_ids", false);
        let cell_counts = buf(&device, nc as u64 * 4, "cell_counts", false);
        let cell_start = buf(&device, nc as u64 * 4, "cell_start", false);
        let write_cursors = buf(&device, nc as u64 * 4, "write_cursors", false);
        let sorted_indices = buf(&device, n_u32 as u64 * 4, "sorted_indices", false);
        let scan_partial = buf(
            &device,
            (nc.div_ceil(SCAN_WG)) as u64 * 4,
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
        let bin_params = uniform_buf(&device, &u32_bytes(&bin_params_data), "bin_params");

        // Pass 2 params: n = nc, n_groups = ceil(nc / SCAN_WG) (matches ScanConfig in WGSL)
        let n_groups = nc.div_ceil(SCAN_WG);
        let scan_params_data = [nc, n_groups, 0u32, 0u32];
        let scan_params = uniform_buf(&device, &u32_bytes(&scan_params_data), "scan_params");

        // Pass 3 params
        let scatter_params_data = [n_u32, nc, 0u32, 0u32];
        let scatter_params =
            uniform_buf(&device, &u32_bytes(&scatter_params_data), "scatter_params");

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
            bin_pl,
            scan_pass_a_pl,
            scan_pass_b_pl,
            scatter_pl,
            bin_bgl,
            scan_bgl,
            scatter_bgl,
        })
    }

    /// Rebuild the cell list from current GPU-resident particle positions.
    ///
    /// `positions_buf` must be a STORAGE buffer holding `[N × 3]` f64 values
    /// (interleaved x, y, z).  It is never read back to CPU.
    ///
    /// After this returns, [`sorted_indices`] and [`cell_start`] are ready
    /// for the force kernel's bind group.
    pub fn build(&self, positions_buf: &wgpu::Buffer) -> Result<()> {
        let dev = &self.device.device;

        // ── Zero cell_counts and write_cursors ───────────────────────────────
        let zeros: Vec<u8> = vec![0u8; self.nc as usize * 4];
        self.device
            .queue
            .write_buffer(&self.bufs.cell_counts, 0, &zeros);
        self.device
            .queue
            .write_buffer(&self.bufs.write_cursors, 0, &zeros);

        // ── Pass 1: atomic bin ───────────────────────────────────────────────
        let bg_bin = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CellListGpu:bg_bin"),
            layout: &self.bin_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.bufs.bin_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: positions_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.bufs.cell_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.bufs.cell_ids.as_entire_binding(),
                },
            ],
        });

        // ── Pass 2 bind groups: cell_counts → cell_start (Blelloch prefix sum) ─
        //
        // Pass A (local_scan): each workgroup scans its 256-element segment.
        //   flags_in  = cell_counts  (source counts, read-only)
        //   scan_out  = cell_start   (per-element exclusive scan, written)
        //   wg_sums   = scan_partial (per-workgroup totals, written)
        //
        // Pass B (add_wg_offsets): one workgroup propagates totals globally.
        //   flags_in  = cell_counts  (read-only; not accessed in pass B but
        //                             required by the shared BGL declaration)
        //   scan_out  = cell_start   (gets global offsets added back)
        //   wg_sums   = scan_partial (source of per-group offsets)
        let bg_scan1 = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CellListGpu:bg_scan_pass_a"),
            layout: &self.scan_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.bufs.scan_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.bufs.cell_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.bufs.cell_start.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.bufs.scan_partial.as_entire_binding(),
                },
            ],
        });
        let bg_scan2 = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CellListGpu:bg_scan_pass_b"),
            layout: &self.scan_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.bufs.scan_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.bufs.cell_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.bufs.cell_start.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.bufs.scan_partial.as_entire_binding(),
                },
            ],
        });

        // ── Pass 3: scatter ──────────────────────────────────────────────────
        let bg_scatter = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CellListGpu:bg_scatter"),
            layout: &self.scatter_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.bufs.scatter_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.bufs.cell_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.bufs.cell_start.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.bufs.write_cursors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.bufs.sorted_indices.as_entire_binding(),
                },
            ],
        });

        // ── Single submit ────────────────────────────────────────────────────
        let mut enc = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("CellListGpu:build"),
            });

        dispatch_pass(
            &mut enc,
            &self.bin_pl,
            &bg_bin,
            "cell_bin",
            self.n.div_ceil(BIN_WG),
            1,
            1,
        );
        dispatch_pass(
            &mut enc,
            &self.scan_pass_a_pl,
            &bg_scan1,
            "scan_local",
            self.nc.div_ceil(SCAN_WG),
            1,
            1,
        );
        dispatch_pass(
            &mut enc,
            &self.scan_pass_b_pl,
            &bg_scan2,
            "scan_add_offsets",
            1,
            1,
            1,
        );
        dispatch_pass(
            &mut enc,
            &self.scatter_pl,
            &bg_scatter,
            "scatter",
            self.n.div_ceil(BIN_WG),
            1,
            1,
        );

        self.device.submit_and_poll(Some(enc.finish()));
        Ok(())
    }

    /// GPU buffer: `[N] u32` particle indices sorted by cell.
    pub fn sorted_indices(&self) -> &wgpu::Buffer {
        &self.bufs.sorted_indices
    }
    /// GPU buffer: `[Nc] u32` exclusive prefix sum (cell start offsets).
    pub fn cell_start(&self) -> &wgpu::Buffer {
        &self.bufs.cell_start
    }
    /// GPU buffer: `[Nc] u32` particle count per cell.
    pub fn cell_count(&self) -> &wgpu::Buffer {
        &self.bufs.cell_counts
    }
    /// Total number of cells.
    pub fn n_cells(&self) -> u32 {
        self.nc
    }
    /// Cell grid dimensions.
    pub fn grid(&self) -> (u32, u32, u32) {
        (self.mx, self.my, self.mz)
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn compile(device: &WgpuDevice, src: &str, label: &str) -> wgpu::ShaderModule {
    device
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        })
}

fn make_pipeline(
    device: &WgpuDevice,
    module: &wgpu::ShaderModule,
    bgl: &wgpu::BindGroupLayout,
    entry: &str,
    label: &str,
) -> wgpu::ComputePipeline {
    let layout = device
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[bgl],
            immediate_size: 0,
        });
    device
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
}

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

fn u32_bytes(data: &[u32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn storage_bgl(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_bgl(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

fn dispatch_pass(
    enc: &mut wgpu::CommandEncoder,
    pl: &wgpu::ComputePipeline,
    bg: &wgpu::BindGroup,
    name: &str,
    x: u32,
    y: u32,
    z: u32,
) {
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(name),
        timestamp_writes: None,
    });
    pass.set_pipeline(pl);
    pass.set_bind_group(0, Some(bg), &[]);
    pass.dispatch_workgroups(x, y, z);
}

#[expect(clippy::unwrap_used, reason = "tests")]
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
        // dispatch math: ceil(N / 64)
        assert_eq!(64u32.div_ceil(BIN_WG), 1);
        assert_eq!(65u32.div_ceil(BIN_WG), 2);
        assert_eq!(10_000u32.div_ceil(BIN_WG), 157);
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
