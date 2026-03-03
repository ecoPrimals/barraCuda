//! Charge Spreading for PPPM
//!
//! Spreads point charges onto a regular mesh using B-spline interpolation.
//! This is the first step of the PPPM algorithm.
//!
//! # Algorithm
//!
//! For each particle i with charge q_i at position r_i:
//! 1. Compute B-spline coefficients for the stencil
//! 2. For each mesh node in the p×p×p stencil:
//!    ρ(mesh) += q_i × M(u_x) × M(u_y) × M(u_z)
//!
//! where M is the cardinal B-spline and u is the fractional mesh coordinate.
//!
//! # Implementation
//!
//! This is done on CPU for correctness. GPU acceleration would require
//! atomic operations for thread-safe mesh updates.

/// WGSL kernel for charge-to-mesh spreading (f64).
pub const WGSL_CHARGE_SPREAD_F64: &str = include_str!("charge_spread_f64.wgsl");

use super::bspline::BsplineCoeffs;
use super::PppmParams;

/// Charge mesh for PPPM
///
/// Stores the mesh charge density ρ(k) = Σ_i q_i M(r_i - r_k)
#[derive(Clone, Debug)]
pub struct ChargeMesh {
    /// Mesh dimensions [Kx, Ky, Kz]
    pub dims: [usize; 3],

    /// Mesh values (row-major: index = iz + Kz*(iy + Ky*ix))
    pub values: Vec<f64>,

    /// Box dimensions (for coordinate conversion)
    pub box_dims: [f64; 3],
}

impl ChargeMesh {
    /// Create empty charge mesh
    pub fn new(dims: [usize; 3], box_dims: [f64; 3]) -> Self {
        let size = dims[0] * dims[1] * dims[2];
        Self {
            dims,
            values: vec![0.0; size],
            box_dims,
        }
    }

    /// Reset mesh to zero
    pub fn clear(&mut self) {
        self.values.fill(0.0);
    }

    /// Get linear index from 3D indices
    #[inline]
    pub fn index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        iz + self.dims[2] * (iy + self.dims[1] * ix)
    }

    /// Get value at mesh point
    #[inline]
    pub fn get(&self, ix: usize, iy: usize, iz: usize) -> f64 {
        self.values[self.index(ix, iy, iz)]
    }

    /// Add value at mesh point
    #[inline]
    pub fn add(&mut self, ix: usize, iy: usize, iz: usize, value: f64) {
        let idx = self.index(ix, iy, iz);
        self.values[idx] += value;
    }

    /// Get mesh spacing
    pub fn spacing(&self) -> [f64; 3] {
        [
            self.box_dims[0] / self.dims[0] as f64,
            self.box_dims[1] / self.dims[1] as f64,
            self.box_dims[2] / self.dims[2] as f64,
        ]
    }

    /// Total mesh elements
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if mesh is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Spread charges onto mesh
///
/// # Arguments
/// * `positions` - Particle positions [x, y, z] for each particle
/// * `charges` - Particle charges
/// * `params` - PPPM parameters
///
/// # Returns
/// Charge mesh with spread charges
pub fn spread_charges(positions: &[[f64; 3]], charges: &[f64], params: &PppmParams) -> ChargeMesh {
    assert_eq!(positions.len(), charges.len());

    let mut mesh = ChargeMesh::new(params.mesh_dims, params.box_dims);

    let order = params.interpolation_order;

    for (pos, &charge) in positions.iter().zip(charges.iter()) {
        // Wrap position to box (periodic boundary conditions)
        let wrapped_pos = wrap_position(*pos, params.box_dims);

        // Compute B-spline coefficients
        let coeffs = BsplineCoeffs::compute(order, wrapped_pos, params.mesh_dims, params.box_dims);

        // Spread charge to stencil nodes
        for ix in 0..order {
            for iy in 0..order {
                for iz in 0..order {
                    let [mx, my, mz] = coeffs.mesh_index(ix, iy, iz, params.mesh_dims);
                    let weight = coeffs.weight(ix, iy, iz);
                    mesh.add(mx, my, mz, charge * weight);
                }
            }
        }
    }

    mesh
}

/// Spread charges onto mesh, also returning B-spline coefficients
///
/// This version returns the coefficients for reuse in force interpolation,
/// avoiding recomputation.
///
/// # Arguments
/// * `positions` - Particle positions [x, y, z] for each particle
/// * `charges` - Particle charges
/// * `params` - PPPM parameters
///
/// # Returns
/// (mesh, coefficients) where coefficients[i] is for particle i
pub fn spread_charges_with_coeffs(
    positions: &[[f64; 3]],
    charges: &[f64],
    params: &PppmParams,
) -> (ChargeMesh, Vec<BsplineCoeffs>) {
    assert_eq!(positions.len(), charges.len());

    let mut mesh = ChargeMesh::new(params.mesh_dims, params.box_dims);

    let order = params.interpolation_order;
    let mut all_coeffs = Vec::with_capacity(positions.len());

    for (pos, &charge) in positions.iter().zip(charges.iter()) {
        // Wrap position to box (periodic boundary conditions)
        let wrapped_pos = wrap_position(*pos, params.box_dims);

        // Compute B-spline coefficients
        let coeffs = BsplineCoeffs::compute(order, wrapped_pos, params.mesh_dims, params.box_dims);

        // Spread charge to stencil nodes
        for ix in 0..order {
            for iy in 0..order {
                for iz in 0..order {
                    let [mx, my, mz] = coeffs.mesh_index(ix, iy, iz, params.mesh_dims);
                    let weight = coeffs.weight(ix, iy, iz);
                    mesh.add(mx, my, mz, charge * weight);
                }
            }
        }

        all_coeffs.push(coeffs);
    }

    (mesh, all_coeffs)
}

/// Wrap position to box using periodic boundary conditions
fn wrap_position(pos: [f64; 3], box_dims: [f64; 3]) -> [f64; 3] {
    [
        pos[0] - (pos[0] / box_dims[0]).floor() * box_dims[0],
        pos[1] - (pos[1] / box_dims[1]).floor() * box_dims[1],
        pos[2] - (pos[2] / box_dims[2]).floor() * box_dims[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::md::electrostatics::PppmAccuracy;

    #[test]
    fn test_charge_mesh_indexing() {
        let mesh = ChargeMesh::new([4, 4, 4], [10.0, 10.0, 10.0]);

        // Check linear indexing
        assert_eq!(mesh.index(0, 0, 0), 0);
        assert_eq!(mesh.index(0, 0, 1), 1);
        assert_eq!(mesh.index(0, 1, 0), 4);
        assert_eq!(mesh.index(1, 0, 0), 16);
    }

    #[test]
    fn test_charge_conservation() {
        let params = PppmParams::auto(10, 10.0, PppmAccuracy::Medium);

        // Random-ish positions
        let positions: Vec<[f64; 3]> = (0..10)
            .map(|i| {
                let t = i as f64 / 10.0;
                [t * 9.0 + 0.5, t * 7.0 + 1.0, t * 8.0 + 0.5]
            })
            .collect();

        // Unit charges
        let charges: Vec<f64> = vec![1.0; 10];

        let mesh = spread_charges(&positions, &charges, &params);

        // Total charge on mesh should equal total particle charge
        let mesh_total: f64 = mesh.values.iter().sum();
        let particle_total: f64 = charges.iter().sum();

        assert!((mesh_total - particle_total).abs() < 1e-10);
    }

    #[test]
    fn test_single_particle_spread() {
        let params = PppmParams::custom(
            1,
            [10.0, 10.0, 10.0],
            [8, 8, 8],
            2.0,
            2.5,
            4, // order 4
        );

        // Particle at box center
        let positions = vec![[5.0, 5.0, 5.0]];
        let charges = vec![1.0];

        let mesh = spread_charges(&positions, &charges, &params);

        // Charge should sum to 1
        let total: f64 = mesh.values.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);

        // Non-zero values should be in a localized region (order^3 = 64 nodes)
        let nonzero_count = mesh.values.iter().filter(|&&v| v.abs() > 1e-14).count();
        assert!(nonzero_count <= 64); // At most order^3 nodes
        assert!(nonzero_count > 0);
    }

    #[test]
    fn test_periodic_wrapping() {
        let params = PppmParams::custom(1, [10.0, 10.0, 10.0], [8, 8, 8], 2.0, 2.5, 4);

        // Particle outside box (should wrap)
        let positions = vec![[15.0, -3.0, 25.0]]; // Wraps to [5, 7, 5]
        let charges = vec![1.0];

        let mesh = spread_charges(&positions, &charges, &params);

        // Should still conserve charge
        let total: f64 = mesh.values.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_wrap_position() {
        let box_dims = [10.0, 10.0, 10.0];

        // Position inside box
        let p1 = wrap_position([5.0, 5.0, 5.0], box_dims);
        assert!((p1[0] - 5.0).abs() < 1e-10);

        // Position outside (positive)
        let p2 = wrap_position([15.0, 25.0, 35.0], box_dims);
        assert!((p2[0] - 5.0).abs() < 1e-10);
        assert!((p2[1] - 5.0).abs() < 1e-10);
        assert!((p2[2] - 5.0).abs() < 1e-10);

        // Position outside (negative)
        let p3 = wrap_position([-3.0, -13.0, -23.0], box_dims);
        assert!((p3[0] - 7.0).abs() < 1e-10);
        assert!((p3[1] - 7.0).abs() < 1e-10);
        assert!((p3[2] - 7.0).abs() < 1e-10);
    }
}
