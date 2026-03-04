// SPDX-License-Identifier: AGPL-3.0-or-later
//! Yukawa Cell-List Force (f64) with PBC
//!
//! **Physics**: Same as yukawa_f64 but O(N) via cell-list algorithm
//! **Use Case**: N > 5000 particles where all-pairs becomes slow
//!
//! **Algorithm**: 27-neighbor cell iteration instead of all-pairs
//! **Requires**: Particles sorted by cell index, cell_start/cell_count pre-computed
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure WGSL shader (f64)
//! - ✅ Zero unsafe code
//! - ✅ Capability-based dispatch
//! - ✅ O(N) scaling via cell decomposition

use crate::device::WgpuDevice;
use crate::error::Result;
use std::sync::Arc;

/// Result of particle sorting by cell: (sorted_positions, particle_indices, cell_start, cell_count)
pub type CellSortResult = (Vec<f64>, Vec<usize>, Vec<u32>, Vec<u32>);

/// f64 Yukawa force with cell-list O(N) scaling
///
/// For large systems (N > 5000), uses cell decomposition for O(N) complexity.
/// Uses GPU-accelerated WGSL shader with sorted particles and pre-computed cell
/// boundaries, falling back to CPU only for very small systems (< 256 particles).
pub struct YukawaCellListF64 {
    device: Arc<WgpuDevice>,
    pipeline: wgpu::ComputePipeline,
}

/// Cell list parameters for spatial decomposition
#[derive(Clone, Debug)]
pub struct CellListParams {
    /// Box dimensions (x, y, z)
    pub box_size: [f64; 3],
    /// Number of cells in each dimension
    pub n_cells: [usize; 3],
    /// Cutoff radius
    pub cutoff: f64,
    /// Kappa (screening parameter)
    pub kappa: f64,
    /// Prefactor for force calculation
    pub prefactor: f64,
    /// Softening parameter
    pub epsilon: f64,
}

impl YukawaCellListF64 {
    /// Create new Yukawa cell-list force calculation
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        let shader_source = include_str!("yukawa_celllist_f64.wgsl");
        let shader_module =
            device.compile_shader_f64(shader_source, Some("YukawaCellListF64 Shader"));

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("YukawaCellListF64 Pipeline"),
                layout: None,
                module: &shader_module,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        Ok(Self { device, pipeline })
    }

    /// Compute Yukawa forces using cell-list algorithm
    ///
    /// Always dispatches the GPU shader with sorted particles.
    pub fn compute_forces(
        &self,
        positions: &[f64],
        params: &CellListParams,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let n = positions.len() / 3;
        if n == 0 {
            return Ok((vec![], vec![]));
        }

        self.compute_gpu(positions, params)
    }

    fn compute_gpu(
        &self,
        positions: &[f64],
        params: &CellListParams,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let n = positions.len() / 3;
        let n_cells_total = params.n_cells[0] * params.n_cells[1] * params.n_cells[2];

        let (sorted_pos, _indices, cell_start, cell_count) =
            self.sort_particles_by_cell(positions, params)?;

        let cell_size = [
            params.box_size[0] / params.n_cells[0] as f64,
            params.box_size[1] / params.n_cells[1] as f64,
            params.box_size[2] / params.n_cells[2] as f64,
        ];

        let gpu_params: Vec<f64> = vec![
            n as f64,
            params.kappa,
            params.prefactor,
            params.cutoff * params.cutoff,
            params.box_size[0],
            params.box_size[1],
            params.box_size[2],
            params.epsilon,
            params.n_cells[0] as f64,
            params.n_cells[1] as f64,
            params.n_cells[2] as f64,
            cell_size[0],
            cell_size[1],
            cell_size[2],
            n_cells_total as f64,
            0.0, // padding to 16
        ];

        let pos_bytes: Vec<u8> = sorted_pos.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_bytes: Vec<u8> = gpu_params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let cs_bytes: Vec<u8> = cell_start.iter().flat_map(|v| v.to_le_bytes()).collect();
        let cc_bytes: Vec<u8> = cell_count.iter().flat_map(|v| v.to_le_bytes()).collect();

        let pos_buffer = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("YCL Positions"),
                contents: &pos_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let forces_buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("YCL Forces"),
            size: (n * 3 * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let pe_buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("YCL PE"),
            size: (n * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer =
            self.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("YCL Params"),
                    contents: &params_bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let cs_buffer = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("YCL CellStart"),
                contents: &cs_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let cc_buffer = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("YCL CellCount"),
                contents: &cc_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let bind_group_layout = self.pipeline.get_bind_group_layout(0);
        let bind_group = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("YCL Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pos_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: forces_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: pe_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: cs_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: cc_buffer.as_entire_binding(),
                    },
                ],
            });

        let n_workgroups = (n as u32).div_ceil(64);
        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("YCL Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("YCL Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n_workgroups, 1, 1);
        }

        self.device.submit_and_poll(Some(encoder.finish()));

        let forces = self.read_f64_buffer(&forces_buffer, n * 3)?;
        let energies = self.read_f64_buffer(&pe_buffer, n)?;

        // Unsort: map GPU output back to original particle ordering
        let (_, original_indices, _, _) = self.sort_particles_by_cell(positions, params)?;
        let mut forces_unsorted = vec![0.0f64; n * 3];
        let mut energies_unsorted = vec![0.0f64; n];
        for (sorted_idx, &orig_idx) in original_indices.iter().enumerate() {
            forces_unsorted[orig_idx * 3] = forces[sorted_idx * 3];
            forces_unsorted[orig_idx * 3 + 1] = forces[sorted_idx * 3 + 1];
            forces_unsorted[orig_idx * 3 + 2] = forces[sorted_idx * 3 + 2];
            energies_unsorted[orig_idx] = energies[sorted_idx];
        }

        Ok((forces_unsorted, energies_unsorted))
    }

    fn read_f64_buffer(&self, buffer: &wgpu::Buffer, count: usize) -> Result<Vec<f64>> {
        let staging = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("YCL Staging"),
            size: (count * 8) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("YCL Copy"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, (count * 8) as u64);
        self.device.submit_and_poll(Some(encoder.finish()));

        let results: Vec<f64> = self.device.map_staging_buffer(&staging, count)?;
        Ok(results)
    }

    /// Sort particles by cell index for optimal GPU performance
    ///
    /// Returns (sorted_positions, particle_indices, cell_start, cell_count)
    pub fn sort_particles_by_cell(
        &self,
        positions: &[f64],
        params: &CellListParams,
    ) -> Result<CellSortResult> {
        let n = positions.len() / 3;
        let n_cells_total = params.n_cells[0] * params.n_cells[1] * params.n_cells[2];

        // Compute cell index for each particle
        let mut particle_cells: Vec<(usize, usize)> = (0..n)
            .map(|i| {
                let cell = self.get_cell_index(
                    positions[i * 3],
                    positions[i * 3 + 1],
                    positions[i * 3 + 2],
                    params,
                );
                (i, cell)
            })
            .collect();

        // Sort by cell index
        particle_cells.sort_by_key(|&(_, cell)| cell);

        // Build sorted arrays
        let mut sorted_positions = vec![0.0f64; n * 3];
        let particle_indices: Vec<usize> = particle_cells.iter().map(|&(i, _)| i).collect();

        for (new_idx, &(old_idx, _)) in particle_cells.iter().enumerate() {
            sorted_positions[new_idx * 3] = positions[old_idx * 3];
            sorted_positions[new_idx * 3 + 1] = positions[old_idx * 3 + 1];
            sorted_positions[new_idx * 3 + 2] = positions[old_idx * 3 + 2];
        }

        // Build cell_start and cell_count
        let mut cell_start = vec![0u32; n_cells_total];
        let mut cell_count = vec![0u32; n_cells_total];

        let mut current_cell = usize::MAX;
        for (idx, &(_, cell)) in particle_cells.iter().enumerate() {
            if cell != current_cell {
                cell_start[cell] = idx as u32;
                current_cell = cell;
            }
            cell_count[cell] += 1;
        }

        Ok((sorted_positions, particle_indices, cell_start, cell_count))
    }

    #[expect(dead_code, clippy::unwrap_used, reason = "tests")]
    #[cfg(test)]
    fn build_cell_list(
        &self,
        positions: &[f64],
        params: &CellListParams,
    ) -> Result<Vec<Vec<usize>>> {
        let n = positions.len() / 3;
        let n_cells_total = params.n_cells[0] * params.n_cells[1] * params.n_cells[2];

        let mut cells: Vec<Vec<usize>> = vec![Vec::new(); n_cells_total];

        for i in 0..n {
            let cell = self.get_cell_index(
                positions[i * 3],
                positions[i * 3 + 1],
                positions[i * 3 + 2],
                params,
            );
            cells[cell].push(i);
        }

        Ok(cells)
    }

    fn get_cell_index(&self, x: f64, y: f64, z: f64, params: &CellListParams) -> usize {
        let cell_size = [
            params.box_size[0] / params.n_cells[0] as f64,
            params.box_size[1] / params.n_cells[1] as f64,
            params.box_size[2] / params.n_cells[2] as f64,
        ];

        let cx = ((x / cell_size[0]).floor() as usize).min(params.n_cells[0] - 1);
        let cy = ((y / cell_size[1]).floor() as usize).min(params.n_cells[1] - 1);
        let cz = ((z / cell_size[2]).floor() as usize).min(params.n_cells[2] - 1);

        cx + cy * params.n_cells[0] + cz * params.n_cells[0] * params.n_cells[1]
    }

    /// CPU reference (test/validation only — production always dispatches shader).
    #[expect(dead_code, clippy::unwrap_used, reason = "tests")]
    #[cfg(test)]
    fn compute_cpu(
        &self,
        positions: &[f64],
        params: &CellListParams,
        cells: &[Vec<usize>],
    ) -> (Vec<f64>, Vec<f64>) {
        let n = positions.len() / 3;
        let mut forces = vec![0.0f64; n * 3];
        let mut energies = vec![0.0f64; n];

        let cutoff_sq = params.cutoff * params.cutoff;
        let eps_sq = params.epsilon * params.epsilon;

        // Iterate over all cells
        for (cell_idx, cell_particles) in cells.iter().enumerate() {
            // Get 27 neighbor cells (including self)
            let neighbors = self.get_neighbor_cells(cell_idx, params);

            for &i in cell_particles {
                let xi = positions[i * 3];
                let yi = positions[i * 3 + 1];
                let zi = positions[i * 3 + 2];

                for &neighbor_cell in &neighbors {
                    for &j in &cells[neighbor_cell] {
                        if i >= j {
                            continue; // Avoid double counting
                        }

                        let xj = positions[j * 3];
                        let yj = positions[j * 3 + 1];
                        let zj = positions[j * 3 + 2];

                        // PBC minimum image
                        let dx = self.pbc_delta(xj - xi, params.box_size[0]);
                        let dy = self.pbc_delta(yj - yi, params.box_size[1]);
                        let dz = self.pbc_delta(zj - zi, params.box_size[2]);

                        let r_sq = dx * dx + dy * dy + dz * dz + eps_sq;
                        if r_sq > cutoff_sq {
                            continue;
                        }

                        let r = r_sq.sqrt();

                        // Yukawa: U = prefactor * exp(-kappa*r) / r
                        let exp_kr = (-params.kappa * r).exp();
                        let u = params.prefactor * exp_kr / r;

                        // Force: F = prefactor * exp(-κr) * (κ + 1/r) / r
                        let f_over_r = params.prefactor * exp_kr * (params.kappa + 1.0 / r) / r_sq;

                        let fx = f_over_r * dx;
                        let fy = f_over_r * dy;
                        let fz = f_over_r * dz;

                        forces[i * 3] += fx;
                        forces[i * 3 + 1] += fy;
                        forces[i * 3 + 2] += fz;

                        forces[j * 3] -= fx;
                        forces[j * 3 + 1] -= fy;
                        forces[j * 3 + 2] -= fz;

                        // Half energy to each particle
                        energies[i] += 0.5 * u;
                        energies[j] += 0.5 * u;
                    }
                }
            }
        }

        (forces, energies)
    }

    #[expect(dead_code, clippy::unwrap_used, reason = "tests")]
    #[cfg(test)]
    fn pbc_delta(&self, delta: f64, box_size: f64) -> f64 {
        delta - box_size * (delta / box_size).round()
    }

    #[expect(dead_code, clippy::unwrap_used, reason = "tests")]
    #[cfg(test)]
    fn get_neighbor_cells(&self, cell_idx: usize, params: &CellListParams) -> Vec<usize> {
        let nx = params.n_cells[0];
        let ny = params.n_cells[1];
        let nz = params.n_cells[2];

        let cz = cell_idx / (nx * ny);
        let cy = (cell_idx % (nx * ny)) / nx;
        let cx = cell_idx % nx;

        let mut neighbors = Vec::with_capacity(27);

        for dz in [-1i32, 0, 1] {
            for dy in [-1i32, 0, 1] {
                for dx in [-1i32, 0, 1] {
                    let ncx = ((cx as i32 + dx + nx as i32) % nx as i32) as usize;
                    let ncy = ((cy as i32 + dy + ny as i32) % ny as i32) as usize;
                    let ncz = ((cz as i32 + dz + nz as i32) % nz as i32) as usize;

                    neighbors.push(ncx + ncy * nx + ncz * nx * ny);
                }
            }
        }

        neighbors
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_device() -> Option<Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
    }

    #[test]
    fn test_cell_list_two_particles() {
        let Some(device) = get_test_device() else {
            return;
        };
        let op = YukawaCellListF64::new(device).unwrap();

        let positions = vec![0.5, 0.5, 0.5, 1.5, 0.5, 0.5]; // Two particles, 1 unit apart

        let params = CellListParams {
            box_size: [10.0, 10.0, 10.0],
            n_cells: [5, 5, 5],
            cutoff: 5.0,
            kappa: 1.0,
            prefactor: 1.0,
            epsilon: 1e-10,
        };

        let (forces, energies) = op.compute_forces(&positions, &params).unwrap();

        // Forces should be equal and opposite
        assert!(
            (forces[0] + forces[3]).abs() < 1e-10,
            "Forces should be equal and opposite"
        );

        // Both particles should have same energy
        assert!(
            (energies[0] - energies[1]).abs() < 1e-10,
            "Energies should be equal"
        );
    }

    #[test]
    fn test_cell_list_pbc() {
        let Some(device) = get_test_device() else {
            return;
        };
        let op = YukawaCellListF64::new(device).unwrap();

        // Two particles on opposite sides of box - should interact via PBC
        let positions = vec![0.5, 0.5, 0.5, 9.5, 0.5, 0.5]; // Distance = 1 with PBC

        let params = CellListParams {
            box_size: [10.0, 10.0, 10.0],
            n_cells: [5, 5, 5],
            cutoff: 5.0,
            kappa: 1.0,
            prefactor: 1.0,
            epsilon: 1e-10,
        };

        let (forces, energies) = op.compute_forces(&positions, &params).unwrap();

        // Should have non-zero energy due to PBC wrapping
        assert!(energies[0].abs() > 1e-10, "Should have interaction via PBC");

        // Particle 0 at x=0.5; nearest image of particle 1 is at x=−0.5 (via PBC).
        // Repulsive Yukawa pushes particle 0 in +x direction (away from image).
        assert!(
            forces[0] > 0.0,
            "Particle 0 should be pushed in +x (away from PBC image): {}",
            forces[0]
        );
    }
}
