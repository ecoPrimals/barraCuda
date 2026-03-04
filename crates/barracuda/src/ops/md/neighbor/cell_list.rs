// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cell List for O(N) Neighbor Search
//!
//! Spatial hashing algorithm that reduces force computation from O(N²) to O(N)
//! by only considering particles in neighboring cells.

/// Cell list for O(N) neighbor search
///
/// Divides the simulation box into cubic cells. For short-range forces with
/// cutoff rc, only particles in the same cell or 26 neighboring cells can
/// interact. This reduces neighbor search from O(N²) to O(27·N_per_cell·N).
///
/// # Example
///
/// ```ignore
/// let cell_list = CellList::new(rc, box_side);
/// cell_list.rebuild(&positions, n);
///
/// // Sort positions for coalesced GPU access
/// let sorted_positions = cell_list.sort_array(&positions, 3);
///
/// // Upload to GPU: sorted_positions, cell_start, cell_count
/// // GPU kernel iterates only over 27 neighboring cells
/// ```
#[derive(Clone, Debug)]
pub struct CellList {
    /// Number of cells per dimension [nx, ny, nz]
    pub n_cells: [usize; 3],

    /// Cell side length [cx, cy, cz]
    pub cell_size: [f64; 3],

    /// Total number of cells (nx * ny * nz)
    pub n_cells_total: usize,

    /// First particle index for each cell (length = n_cells_total)
    pub cell_start: Vec<u32>,

    /// Number of particles in each cell (length = n_cells_total)
    pub cell_count: Vec<u32>,

    /// Particle indices sorted by cell (length = n_particles)
    pub sorted_indices: Vec<usize>,

    /// Cutoff distance
    rc: f64,

    /// Box dimensions
    box_dims: [f64; 3],
}

impl CellList {
    /// Create a new cell list for given cutoff and box size
    ///
    /// # Arguments
    /// * `rc` - Cutoff distance (cell size will be at least this)
    /// * `box_side` - Box side length (assumes cubic box)
    pub fn new(rc: f64, box_side: f64) -> Self {
        Self::new_with_dims(rc, [box_side; 3])
    }

    /// Create a new cell list for non-cubic boxes
    ///
    /// # Arguments
    /// * `rc` - Cutoff distance (cell size will be at least this)
    /// * `box_dims` - Box dimensions [Lx, Ly, Lz]
    pub fn new_with_dims(rc: f64, box_dims: [f64; 3]) -> Self {
        // Calculate number of cells per dimension
        // Each cell must be at least rc to ensure neighbors are within cutoff
        let n_cells = [
            (box_dims[0] / rc).floor() as usize,
            (box_dims[1] / rc).floor() as usize,
            (box_dims[2] / rc).floor() as usize,
        ];

        // Minimum 3 cells per dimension for PBC neighbor iteration
        let n_cells = [n_cells[0].max(3), n_cells[1].max(3), n_cells[2].max(3)];

        let cell_size = [
            box_dims[0] / n_cells[0] as f64,
            box_dims[1] / n_cells[1] as f64,
            box_dims[2] / n_cells[2] as f64,
        ];

        let n_cells_total = n_cells[0] * n_cells[1] * n_cells[2];

        CellList {
            n_cells,
            cell_size,
            n_cells_total,
            cell_start: vec![0; n_cells_total],
            cell_count: vec![0; n_cells_total],
            sorted_indices: Vec::new(),
            rc,
            box_dims,
        }
    }

    /// Rebuild cell list from current positions
    ///
    /// This should be called:
    /// - Every step for guaranteed correctness (rebuild_interval=1)
    /// - Or with Verlet skin radius for better performance (advanced)
    ///
    /// # Arguments
    /// * `positions` - Flat array of positions [x0,y0,z0, x1,y1,z1, ...]
    /// * `n` - Number of particles
    pub fn rebuild(&mut self, positions: &[f64], n: usize) {
        // Assign each particle to a cell
        let mut cell_ids = Vec::with_capacity(n);

        for i in 0..n {
            let cell_id = self.position_to_cell_id(
                positions[i * 3],
                positions[i * 3 + 1],
                positions[i * 3 + 2],
            );
            cell_ids.push(cell_id);
        }

        // Sort particle indices by cell ID for coalesced GPU access
        self.sorted_indices = (0..n).collect();
        self.sorted_indices.sort_by_key(|&i| cell_ids[i]);

        // Reset and rebuild cell_count
        self.cell_count.fill(0);
        for &idx in &self.sorted_indices {
            self.cell_count[cell_ids[idx]] += 1;
        }

        // Compute cell_start (prefix sum)
        let mut offset = 0u32;
        for c in 0..self.n_cells_total {
            self.cell_start[c] = offset;
            offset += self.cell_count[c];
        }
    }

    /// Map 3D position to cell ID
    fn position_to_cell_id(&self, x: f64, y: f64, z: f64) -> usize {
        let cx = ((x / self.cell_size[0]) as usize).min(self.n_cells[0] - 1);
        let cy = ((y / self.cell_size[1]) as usize).min(self.n_cells[1] - 1);
        let cz = ((z / self.cell_size[2]) as usize).min(self.n_cells[2] - 1);

        cx + cy * self.n_cells[0] + cz * self.n_cells[0] * self.n_cells[1]
    }

    /// Sort a particle array according to cell ordering
    ///
    /// After sorting, particles in the same cell are contiguous in memory,
    /// improving GPU cache locality.
    ///
    /// # Arguments
    /// * `data` - Flat array of particle data
    /// * `stride` - Elements per particle (3 for positions/velocities/forces)
    pub fn sort_array(&self, data: &[f64], stride: usize) -> Vec<f64> {
        let mut sorted = vec![0.0f64; data.len()];

        for (new_idx, &old_idx) in self.sorted_indices.iter().enumerate() {
            for s in 0..stride {
                sorted[new_idx * stride + s] = data[old_idx * stride + s];
            }
        }

        sorted
    }

    /// Unsort: map from sorted order back to original order
    ///
    /// Use this when reading back data from GPU to restore original indexing.
    ///
    /// # Arguments
    /// * `data` - Sorted data from GPU
    /// * `stride` - Elements per particle
    pub fn unsort_array(&self, data: &[f64], stride: usize) -> Vec<f64> {
        let mut unsorted = vec![0.0f64; data.len()];

        for (new_idx, &old_idx) in self.sorted_indices.iter().enumerate() {
            for s in 0..stride {
                unsorted[old_idx * stride + s] = data[new_idx * stride + s];
            }
        }

        unsorted
    }

    /// Get GPU parameters for force kernel
    ///
    /// Returns a flat f64 array suitable for GPU upload:
    /// ```text
    /// [n_cells_x, n_cells_y, n_cells_z,
    ///  cell_size_x, cell_size_y, cell_size_z,
    ///  n_cells_total, 0.0]
    /// ```
    pub fn gpu_params(&self) -> Vec<f64> {
        vec![
            self.n_cells[0] as f64,
            self.n_cells[1] as f64,
            self.n_cells[2] as f64,
            self.cell_size[0],
            self.cell_size[1],
            self.cell_size[2],
            self.n_cells_total as f64,
            0.0, // padding for alignment
        ]
    }

    /// Get cell start array for GPU upload
    pub fn cell_start_u32(&self) -> &[u32] {
        &self.cell_start
    }

    /// Get cell count array for GPU upload
    pub fn cell_count_u32(&self) -> &[u32] {
        &self.cell_count
    }

    /// Get the cutoff distance
    pub fn cutoff(&self) -> f64 {
        self.rc
    }

    /// Get box dimensions
    pub fn box_dims(&self) -> [f64; 3] {
        self.box_dims
    }

    /// Estimate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let n = self.sorted_indices.len();
        let cells = self.n_cells_total;

        // sorted_indices: n * 8 bytes (usize)
        // cell_start: cells * 4 bytes (u32)
        // cell_count: cells * 4 bytes (u32)
        n * std::mem::size_of::<usize>() + cells * 8
    }

    /// Get statistics about cell occupancy
    pub fn occupancy_stats(&self) -> CellStats {
        if self.n_cells_total == 0 || self.sorted_indices.is_empty() {
            return CellStats::default();
        }

        let non_empty: Vec<u32> = self.cell_count.iter().copied().filter(|&c| c > 0).collect();

        let min_occ = non_empty.iter().copied().min().unwrap_or(0);
        let max_occ = non_empty.iter().copied().max().unwrap_or(0);
        let mean_occ = if non_empty.is_empty() {
            0.0
        } else {
            non_empty.iter().sum::<u32>() as f64 / non_empty.len() as f64
        };

        let n_empty = self.n_cells_total - non_empty.len();

        CellStats {
            n_cells_total: self.n_cells_total,
            n_empty,
            min_occupancy: min_occ,
            max_occupancy: max_occ,
            mean_occupancy: mean_occ,
        }
    }
}

/// Statistics about cell occupancy
#[derive(Clone, Debug, Default)]
pub struct CellStats {
    pub n_cells_total: usize,
    pub n_empty: usize,
    pub min_occupancy: u32,
    pub max_occupancy: u32,
    pub mean_occupancy: f64,
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_list_construction() {
        let cell_list = CellList::new(2.0, 10.0);

        // Should have 5×5×5 = 125 cells (10/2 = 5 cells per dim)
        assert_eq!(cell_list.n_cells, [5, 5, 5]);
        assert_eq!(cell_list.n_cells_total, 125);
        assert!((cell_list.cell_size[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_cell_list_minimum_cells() {
        // Even with large cutoff, should have at least 3 cells per dim
        let cell_list = CellList::new(5.0, 10.0);

        assert_eq!(cell_list.n_cells, [3, 3, 3]);
        assert_eq!(cell_list.n_cells_total, 27);
    }

    #[test]
    fn test_cell_list_rebuild() {
        let mut cell_list = CellList::new(5.0, 10.0);

        // 4 particles in different quadrants
        let positions = vec![
            1.0, 1.0, 1.0, // particle 0: cell (0,0,0)
            8.0, 8.0, 8.0, // particle 1: cell (2,2,2)
            1.0, 8.0, 1.0, // particle 2: cell (0,2,0)
            8.0, 1.0, 8.0, // particle 3: cell (2,0,2)
        ];

        cell_list.rebuild(&positions, 4);

        // Should have 4 sorted indices
        assert_eq!(cell_list.sorted_indices.len(), 4);

        // Total particles across all cells should be 4
        let total: u32 = cell_list.cell_count.iter().sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_sort_unsort_roundtrip() {
        let mut cell_list = CellList::new(5.0, 10.0);

        let positions = vec![
            1.0, 2.0, 3.0, // particle 0
            7.0, 8.0, 9.0, // particle 1
            4.0, 5.0, 6.0, // particle 2
        ];

        cell_list.rebuild(&positions, 3);

        let sorted = cell_list.sort_array(&positions, 3);
        let unsorted = cell_list.unsort_array(&sorted, 3);

        // Round-trip should recover original
        for (i, (&orig, &recovered)) in positions.iter().zip(unsorted.iter()).enumerate() {
            assert!(
                (orig - recovered).abs() < 1e-10,
                "Mismatch at index {}: {} vs {}",
                i,
                orig,
                recovered
            );
        }
    }

    #[test]
    fn test_gpu_params() {
        let cell_list = CellList::new(2.0, 10.0);
        let params = cell_list.gpu_params();

        assert_eq!(params.len(), 8);
        assert_eq!(params[0], 5.0); // n_cells_x
        assert_eq!(params[1], 5.0); // n_cells_y
        assert_eq!(params[2], 5.0); // n_cells_z
        assert!((params[3] - 2.0).abs() < 1e-10); // cell_size_x
        assert_eq!(params[6], 125.0); // n_cells_total
    }

    #[test]
    fn test_occupancy_stats() {
        let mut cell_list = CellList::new(5.0, 10.0);

        // Cluster particles in one cell
        let positions = vec![1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.2, 1.2, 1.2, 9.0, 9.0, 9.0];

        cell_list.rebuild(&positions, 4);

        let stats = cell_list.occupancy_stats();

        // 27 cells total, 25 empty (3^3 - 2 occupied)
        assert_eq!(stats.n_cells_total, 27);
        assert_eq!(stats.n_empty, 25);
        assert_eq!(stats.min_occupancy, 1);
        assert_eq!(stats.max_occupancy, 3);
    }

    #[test]
    fn test_non_cubic_box() {
        let cell_list = CellList::new_with_dims(2.0, [10.0, 8.0, 6.0]);

        assert_eq!(cell_list.n_cells, [5, 4, 3]);
        assert_eq!(cell_list.n_cells_total, 60);
    }
}
