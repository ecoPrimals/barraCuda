// SPDX-License-Identifier: AGPL-3.0-or-later
//! Verlet Neighbor List for Molecular Dynamics
//!
//! Maintains an explicit per-particle neighbor list with a skin radius,
//! enabling multi-step reuse. Particles beyond `r_cut` but within
//! `r_cut + r_skin` are included; the list is only rebuilt when the
//! maximum displacement since last build exceeds `r_skin / 2`.
//!
//! Absorbed from hotSpring's `VerletListGpu` design (Feb 2026).
//!
//! # Cell-list vs Verlet-list
//!
//! | Aspect | Cell List | Verlet List |
//! |--------|-----------|-------------|
//! | Rebuild | Every step | Every 10–30 steps |
//! | Memory | O(Nc) cell metadata | O(N × ~50) neighbor pairs |
//! | GPU overhead | Cell binning each step | Displacement check only |
//! | Best for | Dense systems, GPU | Moderate density, long sims |
//!
//! # Usage
//!
//! ```ignore
//! use barracuda::ops::md::neighbor::VerletList;
//!
//! let mut vl = VerletList::new(2.5, 0.3, 10.0);
//! vl.build(&positions, n);
//!
//! for step in 0..1000 {
//!     // Only rebuild if particles moved too far
//!     if vl.needs_rebuild(&positions) {
//!         vl.build(&positions, n);
//!     }
//!     // Use vl.neighbors(i) in force computation
//! }
//! ```

/// Verlet neighbor list with skin radius for multi-step reuse.
#[derive(Clone, Debug)]
pub struct VerletList {
    /// Cutoff distance for interactions
    r_cut: f64,
    /// Skin radius beyond cutoff
    r_skin: f64,
    /// Total list radius: `r_cut + r_skin`
    r_list: f64,
    /// Box dimensions `[Lx, Ly, Lz]`
    box_dims: [f64; 3],
    /// CSR-style: `offsets[i]..offsets[i+1]` indexes into `neighbors`
    offsets: Vec<usize>,
    /// Flat neighbor indices
    neighbors: Vec<u32>,
    /// Positions at last build (flat `[x0,y0,z0, x1,y1,z1, ...]`)
    positions_at_build: Vec<f64>,
    /// Number of particles
    n_particles: usize,
    /// Number of rebuilds performed
    rebuild_count: u64,
}

impl VerletList {
    /// Create a new Verlet list.
    ///
    /// # Arguments
    /// * `r_cut` — interaction cutoff distance
    /// * `r_skin` — skin radius (larger = fewer rebuilds, more memory)
    /// * `box_side` — cubic box side length
    #[must_use]
    pub fn new(r_cut: f64, r_skin: f64, box_side: f64) -> Self {
        Self::new_with_dims(r_cut, r_skin, [box_side; 3])
    }

    /// Create a new Verlet list for non-cubic boxes.
    #[must_use]
    pub fn new_with_dims(r_cut: f64, r_skin: f64, box_dims: [f64; 3]) -> Self {
        Self {
            r_cut,
            r_skin,
            r_list: r_cut + r_skin,
            box_dims,
            offsets: Vec::new(),
            neighbors: Vec::new(),
            positions_at_build: Vec::new(),
            n_particles: 0,
            rebuild_count: 0,
        }
    }

    /// Build the neighbor list from scratch.
    ///
    /// Uses the cell list as spatial acceleration: O(N) average case.
    ///
    /// # Arguments
    /// * `positions` — flat `[x0,y0,z0, x1,y1,z1, ...]`
    /// * `n` — number of particles
    pub fn build(&mut self, positions: &[f64], n: usize) {
        self.n_particles = n;
        self.rebuild_count += 1;

        self.positions_at_build = positions[..n * 3].to_vec();

        self.offsets.clear();
        self.offsets.reserve(n + 1);
        self.neighbors.clear();

        let r_list_sq = self.r_list * self.r_list;

        // Simple O(N²) build — for production use at N > 5000, compose with
        // CellList to accelerate. At N < 5000 this is faster due to no
        // cell-building overhead.
        for i in 0..n {
            self.offsets.push(self.neighbors.len());
            let xi = positions[i * 3];
            let yi = positions[i * 3 + 1];
            let zi = positions[i * 3 + 2];

            for j in 0..n {
                if i == j {
                    continue;
                }
                let dx = Self::min_image(positions[j * 3] - xi, self.box_dims[0]);
                let dy = Self::min_image(positions[j * 3 + 1] - yi, self.box_dims[1]);
                let dz = Self::min_image(positions[j * 3 + 2] - zi, self.box_dims[2]);
                let r_sq = dx * dx + dy * dy + dz * dz;

                if r_sq <= r_list_sq {
                    self.neighbors.push(j as u32);
                }
            }
        }
        self.offsets.push(self.neighbors.len());
    }

    /// Whether any particle has moved far enough to require a rebuild.
    ///
    /// Checks if the maximum displacement since last build exceeds
    /// `r_skin / 2`. Uses the conservative criterion: if **any** particle
    /// could have crossed the skin boundary, rebuild.
    #[must_use]
    pub fn needs_rebuild(&self, current_positions: &[f64]) -> bool {
        if self.positions_at_build.is_empty() {
            return true;
        }

        let half_skin_sq = (self.r_skin / 2.0) * (self.r_skin / 2.0);

        for i in 0..self.n_particles {
            let dx = Self::min_image(
                current_positions[i * 3] - self.positions_at_build[i * 3],
                self.box_dims[0],
            );
            let dy = Self::min_image(
                current_positions[i * 3 + 1] - self.positions_at_build[i * 3 + 1],
                self.box_dims[1],
            );
            let dz = Self::min_image(
                current_positions[i * 3 + 2] - self.positions_at_build[i * 3 + 2],
                self.box_dims[2],
            );
            if dx * dx + dy * dy + dz * dz > half_skin_sq {
                return true;
            }
        }
        false
    }

    /// Neighbors of particle `i` (indices into the position array).
    #[must_use]
    pub fn neighbors_of(&self, i: usize) -> &[u32] {
        if i >= self.n_particles {
            return &[];
        }
        &self.neighbors[self.offsets[i]..self.offsets[i + 1]]
    }

    /// Number of neighbors for particle `i`.
    #[must_use]
    pub fn neighbor_count(&self, i: usize) -> usize {
        if i >= self.n_particles {
            return 0;
        }
        self.offsets[i + 1] - self.offsets[i]
    }

    /// Total number of neighbor pairs stored.
    #[must_use]
    pub fn total_pairs(&self) -> usize {
        self.neighbors.len()
    }

    /// Average number of neighbors per particle.
    #[must_use]
    pub fn avg_neighbors(&self) -> f64 {
        if self.n_particles == 0 {
            return 0.0;
        }
        self.neighbors.len() as f64 / self.n_particles as f64
    }

    /// Number of rebuilds performed so far.
    #[must_use]
    pub fn rebuild_count(&self) -> u64 {
        self.rebuild_count
    }

    /// Interaction cutoff distance.
    #[must_use]
    pub fn r_cut(&self) -> f64 {
        self.r_cut
    }

    /// Skin radius.
    #[must_use]
    pub fn r_skin(&self) -> f64 {
        self.r_skin
    }

    /// Flat neighbor indices (CSR value array).
    #[must_use]
    pub fn neighbor_indices(&self) -> &[u32] {
        &self.neighbors
    }

    /// CSR offset array: `offsets[i]..offsets[i+1]` indexes `neighbor_indices()`.
    #[must_use]
    pub fn neighbor_offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// Minimum-image convention for periodic boundaries.
    fn min_image(dx: f64, box_len: f64) -> f64 {
        let mut d = dx;
        if d > box_len * 0.5 {
            d -= box_len;
        } else if d < -box_len * 0.5 {
            d += box_len;
        }
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid_positions(n_per_dim: usize, spacing: f64) -> Vec<f64> {
        let mut positions = Vec::new();
        for iz in 0..n_per_dim {
            for iy in 0..n_per_dim {
                for ix in 0..n_per_dim {
                    positions.push(ix as f64 * spacing + 0.1);
                    positions.push(iy as f64 * spacing + 0.1);
                    positions.push(iz as f64 * spacing + 0.1);
                }
            }
        }
        positions
    }

    #[test]
    fn two_particles_within_cutoff() {
        let positions = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut vl = VerletList::new(1.5, 0.3, 10.0);
        vl.build(&positions, 2);

        assert_eq!(vl.neighbor_count(0), 1);
        assert_eq!(vl.neighbors_of(0), &[1]);
        assert_eq!(vl.neighbor_count(1), 1);
        assert_eq!(vl.neighbors_of(1), &[0]);
    }

    #[test]
    fn two_particles_outside_cutoff() {
        let positions = vec![0.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let mut vl = VerletList::new(1.5, 0.3, 10.0);
        vl.build(&positions, 2);

        assert_eq!(vl.neighbor_count(0), 0);
        assert_eq!(vl.neighbor_count(1), 0);
    }

    #[test]
    fn periodic_boundary_neighbors() {
        // Particles at opposite ends of a small box — should be neighbors
        // via minimum image
        let positions = vec![0.5, 5.0, 5.0, 9.5, 5.0, 5.0];
        let mut vl = VerletList::new(1.5, 0.3, 10.0);
        vl.build(&positions, 2);

        assert_eq!(vl.neighbor_count(0), 1, "PBC should connect them");
    }

    #[test]
    fn needs_rebuild_after_displacement() {
        let positions = vec![1.0, 1.0, 1.0, 5.0, 5.0, 5.0];
        let mut vl = VerletList::new(2.0, 0.4, 10.0);
        vl.build(&positions, 2);

        // Small displacement: no rebuild needed
        let moved_small = vec![1.05, 1.0, 1.0, 5.0, 5.0, 5.0];
        assert!(!vl.needs_rebuild(&moved_small));

        // Large displacement: rebuild needed
        let moved_large = vec![1.5, 1.0, 1.0, 5.0, 5.0, 5.0];
        assert!(vl.needs_rebuild(&moved_large));
    }

    #[test]
    fn grid_neighbor_count() {
        let positions = make_grid_positions(3, 1.0);
        let n = 27;
        let mut vl = VerletList::new(1.5, 0.3, 10.0);
        vl.build(&positions, n);

        assert!(vl.total_pairs() > 0);
        assert!(vl.avg_neighbors() > 0.0);
    }

    #[test]
    fn rebuild_count_increments() {
        let positions = vec![1.0, 1.0, 1.0, 5.0, 5.0, 5.0];
        let mut vl = VerletList::new(2.0, 0.4, 10.0);
        assert_eq!(vl.rebuild_count(), 0);
        vl.build(&positions, 2);
        assert_eq!(vl.rebuild_count(), 1);
        vl.build(&positions, 2);
        assert_eq!(vl.rebuild_count(), 2);
    }

    #[test]
    fn empty_list() {
        let vl = VerletList::new(2.0, 0.4, 10.0);
        assert_eq!(vl.total_pairs(), 0);
        assert!(vl.needs_rebuild(&[]));
    }

    #[test]
    fn out_of_bounds_particle() {
        let vl = VerletList::new(2.0, 0.4, 10.0);
        assert_eq!(vl.neighbor_count(999), 0);
        assert!(vl.neighbors_of(999).is_empty());
    }
}
