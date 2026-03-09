// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lattice QCD / gauge field theory GPU primitives
//!
//! All math runs on GPU via WGSL shaders. CPU reference implementations exist
//! in test-gated modules for validation only.
//!
//! # WGSL Library Shaders
//!
//! | Module | Content |
//! |--------|---------|
//! | `complex_f64` | Complex f64 arithmetic (`c64_*` functions) |
//! | `su3` | SU(3) 3×3 matrix algebra |
//! | `lcg` | LCG PRNG for GPU lattice kernels |
//! | `su3_extended` | Reunitarize, `exp_cayley`, random SU(3) generation |
//!
//! # GPU Operators
//!
//! | Module | Content |
//! |--------|---------|
//! | `plaquette` | Wilson plaquette GPU op (SU(3), 4D) |
//! | `higgs_u1` | U(1) Abelian Higgs HMC force (2D) |
//! | `hmc_force_su3` | SU(3) HMC gauge force (4D, Wilson action) |
//! | `dirac` | Staggered Dirac operator (Kogut-Susskind, 4D SU(3)) |
//! | `cg` | CG vector ops (complex dot, axpy, xpay) for fermion solves |
//! | `gpu_lattice_init` | Cold/hot start lattice initialization |
//! | `gpu_wilson_action` | Per-site Wilson action (for GPU reduction) |
//! | `gpu_polyakov` | Polyakov loop (temporal Wilson line) |
//! | `gpu_hmc_leapfrog` | HMC leapfrog: momentum kick, link update, momentum gen |
//! | `gpu_kinetic_energy` | Per-link kinetic energy from momenta |
//! | `gpu_pseudofermion` | Pseudofermion heatbath noise + fermion force |
//! | `gpu_cg_solver` | GPU CG solver orchestration (D†D solve via multi-dispatch) |
//! | `gpu_hmc_trajectory` | Full dynamical fermion HMC trajectory on GPU |
//!
//! # Test-Only CPU Reference
//!
//! | Module | Content |
//! |--------|---------|
//! | `constants` | LCG constants and CPU PRNG (test reference) |
//! | `cpu_complex` | Complex64 CPU arithmetic (runtime — used by `special::plasma_dispersion`) |
//! | `cpu_su3` | SU(3) CPU matrix ops (test reference) |
//! | `wilson` | Wilson lattice CPU (test reference) |
//! | `cpu_dirac` | Dirac/CG CPU solver (test reference) |
//! | `pseudofermion` | Pseudofermion HMC CPU (test reference) |
//!
//! # Neighbor Resolution
//!
//! Lattice shaders can resolve neighbor site indices in two ways via [`NeighborMode`]:
//! - [`NeighborMode::OnTheFly`]: on-the-fly from lattice dimensions (default, suitable for small lattices)
//! - [`NeighborMode::PrecomputedBuffer`]: precomputed table (faster for large lattices, Ising/Potts/lattice gas)

// WGSL library preambles
pub mod complex_f64;
pub mod lcg;
pub mod su3;
pub mod su3_extended;

// GPU operators
pub mod absorbed_shaders;
pub mod cg;
pub mod dirac;
pub mod gpu_cg_resident;
pub mod gpu_cg_solver;
pub mod gpu_hmc_leapfrog;
pub mod gpu_hmc_trajectory;
pub mod gpu_kinetic_energy;
pub mod gpu_lattice_init;
pub mod gpu_polyakov;
pub mod gpu_pseudofermion;
pub mod gpu_wilson_action;
pub mod higgs_u1;
pub mod hmc_force_su3;
pub mod omelyan_integrator;
pub mod plaquette;

/// LCG PRNG and lattice constants (used at runtime by `HostRng`).
pub mod constants;

// Complex64 is used at runtime by special::plasma_dispersion (ISSUE-006 absorption).
pub mod cpu_complex;
#[cfg(test)]
pub mod cpu_dirac;
#[cfg(test)]
pub mod cpu_su3;
#[cfg(test)]
pub mod pseudofermion;
#[cfg(test)]
pub mod wilson;

// ── Neighbor resolution for lattice shaders ────────────────────────────────────

use std::sync::Arc;

/// How lattice shaders resolve neighbor site indices.
#[derive(Debug, Clone)]
pub enum NeighborMode {
    /// Compute neighbors on-the-fly (modular arithmetic in shader).
    /// Suitable for small lattices or one-off calculations.
    OnTheFly,
    /// Precomputed neighbor index table passed as buffer.
    /// Faster for large lattices (avoids recomputing modular arithmetic every step).
    /// Layout: `table[site * neighbors_per_site + dir]` gives neighbor index.
    PrecomputedBuffer(Vec<u32>),
}

/// SU(3) Wilson gauge action density from average plaquette.
///
/// For SU(3) in 4 dimensions there are 6 plaquette orientations per site.
/// The per-site Wilson action density (without β factor) is:
///   `a_d = 6 × (1 − ⟨P⟩)`
/// where ⟨P⟩ = (1/3) Re Tr `U_p` averaged over all orientations and sites.
/// A cold (identity) configuration gives ⟨P⟩ = 1 → `a_d` = 0.
#[inline]
#[must_use]
pub fn action_density(avg_plaquette: f64) -> f64 {
    6.0 * (1.0 - avg_plaquette)
}

impl NeighborMode {
    /// Generate a precomputed neighbor table for a 2D L×L periodic lattice.
    ///
    /// 4 neighbors per site: +x, -x, +y, -y (up, down, left, right).
    /// Row-major indexing: `idx = y * L + x`.
    #[must_use]
    pub fn precompute_periodic_2d(l: usize) -> Self {
        let n_sites = l * l;
        let mut table = Vec::with_capacity(n_sites * 4);
        let l_u = l as u32;

        for y in 0..l {
            for x in 0..l {
                let idx = |x: u32, y: u32| -> u32 { y * l_u + x };
                let x = x as u32;
                let y = y as u32;
                // +x, -x, +y, -y (periodic BC)
                table.push(idx((x + 1) % l_u, y));
                table.push(idx((x + l_u - 1) % l_u, y));
                table.push(idx(x, (y + 1) % l_u));
                table.push(idx(x, (y + l_u - 1) % l_u));
            }
        }

        Self::PrecomputedBuffer(table)
    }

    /// Generate a precomputed neighbor table for a 3D L×L×L periodic lattice.
    ///
    /// 6 neighbors per site: +x, -x, +y, -y, +z, -z.
    /// Indexing: `idx = z * L² + y * L + x`.
    #[must_use]
    pub fn precompute_periodic_3d(l: usize) -> Self {
        let n_sites = l * l * l;
        let mut table = Vec::with_capacity(n_sites * 6);
        let l_u = l as u32;

        for z in 0..l {
            for y in 0..l {
                for x in 0..l {
                    let idx = |x: u32, y: u32, z: u32| -> u32 { z * l_u * l_u + y * l_u + x };
                    let x = x as u32;
                    let y = y as u32;
                    let z = z as u32;
                    // +x, -x, +y, -y, +z, -z (periodic BC)
                    table.push(idx((x + 1) % l_u, y, z));
                    table.push(idx((x + l_u - 1) % l_u, y, z));
                    table.push(idx(x, (y + 1) % l_u, z));
                    table.push(idx(x, (y + l_u - 1) % l_u, z));
                    table.push(idx(x, y, (z + 1) % l_u));
                    table.push(idx(x, y, (z + l_u - 1) % l_u));
                }
            }
        }

        Self::PrecomputedBuffer(table)
    }

    /// Build a precomputed neighbor table for a 4D periodic lattice.
    ///
    /// # Index Convention
    ///
    /// Uses **x-fastest** (row-major / C-order) with t as the outermost dimension:
    ///
    /// ```text
    /// index = t * (Nz * Ny * Nx) + z * (Ny * Nx) + y * Nx + x
    /// ```
    ///
    /// This matches the hotSpring lattice QCD convention. For a 4^4 lattice,
    /// site (x=1, y=0, z=0, t=0) has index 1, and site (x=0, y=0, z=0, t=1)
    /// has index 64.
    ///
    /// **Note for springs**: hotSpring and barraCuda both use x-fastest.
    /// If your physics uses z-fastest ordering, transpose your data before
    /// passing to precomputed neighbor tables, or use `OnTheFly` mode.
    ///
    /// # Direction ordering
    ///
    /// 8 neighbors per site in this order: +x, -x, +y, -y, +z, -z, +t, -t.
    /// `table[site * 8 + dir]` gives the neighbor index for direction `dir`.
    #[must_use]
    pub fn precompute_periodic_4d(dims: [u32; 4]) -> Self {
        let [nx, ny, nz, nt] = dims;
        let n_sites = (nx * ny * nz * nt) as usize;
        let mut table = Vec::with_capacity(n_sites * 8);

        for t in 0..nt {
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let idx = |x: u32, y: u32, z: u32, t: u32| -> u32 {
                            t * nz * ny * nx + z * ny * nx + y * nx + x
                        };
                        table.push(idx((x + 1) % nx, y, z, t));
                        table.push(idx((x + nx - 1) % nx, y, z, t));
                        table.push(idx(x, (y + 1) % ny, z, t));
                        table.push(idx(x, (y + ny - 1) % ny, z, t));
                        table.push(idx(x, y, (z + 1) % nz, t));
                        table.push(idx(x, y, (z + nz - 1) % nz, t));
                        table.push(idx(x, y, z, (t + 1) % nt));
                        table.push(idx(x, y, z, (t + nt - 1) % nt));
                    }
                }
            }
        }

        Self::PrecomputedBuffer(table)
    }

    /// Create a GPU buffer from the precomputed table (for `PrecomputedBuffer` variant).
    /// Returns `None` for `OnTheFly`.
    #[must_use]
    pub fn create_gpu_buffer(
        &self,
        device: &crate::device::WgpuDevice,
    ) -> Option<Arc<wgpu::Buffer>> {
        let table = match self {
            Self::OnTheFly => return None,
            Self::PrecomputedBuffer(t) => t,
        };
        let buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Neighbor Table"),
                contents: bytemuck::cast_slice(table),
                usage: wgpu::BufferUsages::STORAGE,
            });
        Some(Arc::new(buffer))
    }
}

#[cfg(test)]
mod action_density_tests {
    use super::action_density;

    #[test]
    fn cold_start_zero() {
        assert!((action_density(1.0)).abs() < 1e-15);
    }

    #[test]
    fn strong_coupling() {
        let ad = action_density(0.0);
        assert!((ad - 6.0).abs() < 1e-15);
    }

    #[test]
    fn typical_value() {
        let ad = action_density(0.55);
        assert!((ad - 2.7).abs() < 1e-14);
    }
}

#[cfg(test)]
mod neighbor_tests {
    use super::NeighborMode;

    #[test]
    fn precompute_2d_table_size() {
        // L×L lattice: N sites × 4 neighbors
        let mode = NeighborMode::precompute_periodic_2d(8);
        let NeighborMode::PrecomputedBuffer(table) = &mode else {
            panic!("expected PrecomputedBuffer");
        };
        assert_eq!(table.len(), 8 * 8 * 4, "8×8 lattice × 4 neighbors = 256");
    }

    #[test]
    fn precompute_2d_periodic_boundary() {
        // Site (0,0): -x wraps to (L-1,0), -y wraps to (0,L-1)
        let mode = NeighborMode::precompute_periodic_2d(4);
        let NeighborMode::PrecomputedBuffer(table) = &mode else {
            panic!("expected PrecomputedBuffer");
        };
        // Site 0 = (0,0): dirs [+x, -x, +y, -y]
        // +x -> (1,0) = 1, -x -> (3,0) = 3, +y -> (0,1) = 4, -y -> (0,3) = 12
        const NEIGHBORS_2D: usize = 4;
        let site = 0;
        let nbr_plus_x = table[site * NEIGHBORS_2D];
        let nbr_minus_x = table[site * NEIGHBORS_2D + 1];
        let nbr_plus_y = table[site * NEIGHBORS_2D + 2];
        let nbr_minus_y = table[site * NEIGHBORS_2D + 3];
        assert_eq!(nbr_plus_x, 1, "site (0,0) +x -> (1,0)");
        assert_eq!(nbr_minus_x, 3, "site (0,0) -x wraps to (3,0)");
        assert_eq!(nbr_plus_y, 4, "site (0,0) +y -> (0,1)");
        assert_eq!(nbr_minus_y, 12, "site (0,0) -y wraps to (0,3)");
    }

    #[test]
    fn precompute_3d_table_size() {
        // L×L×L lattice: N sites × 6 neighbors
        let mode = NeighborMode::precompute_periodic_3d(4);
        let NeighborMode::PrecomputedBuffer(table) = &mode else {
            panic!("expected PrecomputedBuffer");
        };
        assert_eq!(table.len(), 4 * 4 * 4 * 6, "4³ lattice × 6 neighbors = 384");
    }

    #[test]
    fn precompute_3d_periodic_wrap() {
        // Site (0,0,0): -x -> (L-1,0,0), -y -> (0,L-1,0), -z -> (0,0,L-1)
        let mode = NeighborMode::precompute_periodic_3d(3);
        let NeighborMode::PrecomputedBuffer(table) = &mode else {
            panic!("expected PrecomputedBuffer");
        };
        // idx = z*9 + y*3 + x; site (0,0,0) = 0
        // -x -> (2,0,0) = 2, -y -> (0,2,0) = 6, -z -> (0,0,2) = 18
        const NEIGHBORS_3D: usize = 6;
        let site = 0;
        let nbr_minus_x = table[site * NEIGHBORS_3D + 1];
        let nbr_minus_y = table[site * NEIGHBORS_3D + 3];
        let nbr_minus_z = table[site * NEIGHBORS_3D + 5];
        assert_eq!(nbr_minus_x, 2, "site (0,0,0) -x wraps to (2,0,0)");
        assert_eq!(nbr_minus_y, 6, "site (0,0,0) -y wraps to (0,2,0)");
        assert_eq!(nbr_minus_z, 18, "site (0,0,0) -z wraps to (0,0,2)");
    }

    #[test]
    fn precompute_4d_table_size() {
        // 4^4 lattice: 256 sites * 8 neighbors = 2048 entries
        let mode = NeighborMode::precompute_periodic_4d([4, 4, 4, 4]);
        let NeighborMode::PrecomputedBuffer(table) = &mode else {
            panic!("expected PrecomputedBuffer");
        };
        assert_eq!(table.len(), 2048);
    }

    #[test]
    fn precompute_4d_periodic_boundary() {
        let mode = NeighborMode::precompute_periodic_4d([4, 4, 4, 4]);
        let NeighborMode::PrecomputedBuffer(table) = &mode else {
            panic!("expected PrecomputedBuffer");
        };
        // Site (0,0,0,0): +x -> (1,0,0,0), -x wraps to (3,0,0,0)
        let idx = |x: u32, y: u32, z: u32, t: u32| -> u32 { t * 4 * 4 * 4 + z * 4 * 4 + y * 4 + x };
        const NEIGHBORS_4D: usize = 8;
        let site = 0;
        let nbr_plus_x = table[site * NEIGHBORS_4D];
        let nbr_minus_x = table[site * NEIGHBORS_4D + 1];
        assert_eq!(nbr_plus_x, idx(1, 0, 0, 0));
        assert_eq!(nbr_minus_x, idx(3, 0, 0, 0));
    }
}
