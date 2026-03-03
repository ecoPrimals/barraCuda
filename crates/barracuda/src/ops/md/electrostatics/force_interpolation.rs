//! Force Interpolation for PPPM
//!
//! Interpolates the mesh potential back to particle positions to compute
//! the k-space contribution to forces.
//!
//! # Algorithm
//!
//! For each particle i at position r_i:
//! F_i = -q_i × ∇φ(r_i)
//!
//! where ∇φ is computed from the mesh potential using B-spline derivatives:
//! ∇φ(r_i) = Σ_k φ(r_k) × ∇M(r_i - r_k)
//!
//! The gradient of the B-spline is:
//! ∂M(r)/∂x = M'(u_x) × M(u_y) × M(u_z) × (K_x / L_x)

/// WGSL kernel for mesh-to-particle force interpolation (f64).
pub const WGSL_FORCE_INTERPOLATION_F64: &str = include_str!("force_interpolation_f64.wgsl");

use super::bspline::BsplineCoeffs;
use super::PppmParams;

/// Mesh potential for force interpolation
#[derive(Clone, Debug)]
pub struct PotentialMesh {
    /// Mesh dimensions [Kx, Ky, Kz]
    pub dims: [usize; 3],

    /// Potential values at mesh nodes
    pub values: Vec<f64>,

    /// Box dimensions
    pub box_dims: [f64; 3],
}

impl PotentialMesh {
    /// Create from raw values
    pub fn from_values(dims: [usize; 3], values: Vec<f64>, box_dims: [f64; 3]) -> Self {
        assert_eq!(values.len(), dims[0] * dims[1] * dims[2]);
        Self {
            dims,
            values,
            box_dims,
        }
    }

    /// Get potential at mesh point
    #[inline]
    pub fn get(&self, ix: usize, iy: usize, iz: usize) -> f64 {
        let idx = iz + self.dims[2] * (iy + self.dims[1] * ix);
        self.values[idx]
    }

    /// Get mesh spacing
    pub fn spacing(&self) -> [f64; 3] {
        [
            self.box_dims[0] / self.dims[0] as f64,
            self.box_dims[1] / self.dims[1] as f64,
            self.box_dims[2] / self.dims[2] as f64,
        ]
    }
}

/// Interpolate forces from mesh potential to particles
///
/// # Arguments
/// * `potential` - Potential mesh from k-space solve
/// * `charges` - Particle charges
/// * `coeffs` - B-spline coefficients (from charge spreading)
/// * `params` - PPPM parameters
///
/// # Returns
/// Forces [fx, fy, fz] for each particle
pub fn interpolate_forces(
    potential: &PotentialMesh,
    charges: &[f64],
    coeffs: &[BsplineCoeffs],
    params: &PppmParams,
) -> Vec<[f64; 3]> {
    assert_eq!(charges.len(), coeffs.len());

    let order = params.interpolation_order;
    let n_particles = charges.len();
    let mut forces = vec![[0.0, 0.0, 0.0]; n_particles];

    // Scale factors for gradient: ∂/∂r = (K/L) × ∂/∂u
    let scale = [
        params.mesh_dims[0] as f64 / params.box_dims[0],
        params.mesh_dims[1] as f64 / params.box_dims[1],
        params.mesh_dims[2] as f64 / params.box_dims[2],
    ];

    for (i, (coeff, &charge)) in coeffs.iter().zip(charges.iter()).enumerate() {
        let mut grad_phi = [0.0, 0.0, 0.0];

        // Sum over stencil nodes
        for ix in 0..order {
            for iy in 0..order {
                for iz in 0..order {
                    let [mx, my, mz] = coeff.mesh_index(ix, iy, iz, params.mesh_dims);
                    let phi = potential.get(mx, my, mz);
                    let [gx, gy, gz] = coeff.gradient_weights(ix, iy, iz);

                    grad_phi[0] += phi * gx * scale[0];
                    grad_phi[1] += phi * gy * scale[1];
                    grad_phi[2] += phi * gz * scale[2];
                }
            }
        }

        // F = -q × ∇φ
        forces[i][0] = -charge * grad_phi[0];
        forces[i][1] = -charge * grad_phi[1];
        forces[i][2] = -charge * grad_phi[2];
    }

    forces
}

/// Interpolate forces with positions (computes B-spline coefficients internally)
///
/// Use this when B-spline coefficients weren't saved from charge spreading.
///
/// # Arguments
/// * `potential` - Potential mesh
/// * `positions` - Particle positions
/// * `charges` - Particle charges
/// * `params` - PPPM parameters
///
/// # Returns
/// Forces [fx, fy, fz] for each particle
pub fn interpolate_forces_from_positions(
    potential: &PotentialMesh,
    positions: &[[f64; 3]],
    charges: &[f64],
    params: &PppmParams,
) -> Vec<[f64; 3]> {
    assert_eq!(positions.len(), charges.len());

    let order = params.interpolation_order;
    let n_particles = positions.len();
    let mut forces = vec![[0.0, 0.0, 0.0]; n_particles];

    // Scale factors for gradient
    let scale = [
        params.mesh_dims[0] as f64 / params.box_dims[0],
        params.mesh_dims[1] as f64 / params.box_dims[1],
        params.mesh_dims[2] as f64 / params.box_dims[2],
    ];

    for (i, (pos, &charge)) in positions.iter().zip(charges.iter()).enumerate() {
        // Wrap position
        let wrapped_pos = wrap_position(*pos, params.box_dims);

        // Compute B-spline coefficients
        let coeff = BsplineCoeffs::compute(order, wrapped_pos, params.mesh_dims, params.box_dims);

        let mut grad_phi = [0.0, 0.0, 0.0];

        // Sum over stencil nodes
        for ix in 0..order {
            for iy in 0..order {
                for iz in 0..order {
                    let [mx, my, mz] = coeff.mesh_index(ix, iy, iz, params.mesh_dims);
                    let phi = potential.get(mx, my, mz);
                    let [gx, gy, gz] = coeff.gradient_weights(ix, iy, iz);

                    grad_phi[0] += phi * gx * scale[0];
                    grad_phi[1] += phi * gy * scale[1];
                    grad_phi[2] += phi * gz * scale[2];
                }
            }
        }

        // F = -q × ∇φ
        forces[i][0] = -charge * grad_phi[0];
        forces[i][1] = -charge * grad_phi[1];
        forces[i][2] = -charge * grad_phi[2];
    }

    forces
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

    #[test]
    fn test_uniform_potential_zero_force() {
        let params = PppmParams::custom(2, [10.0, 10.0, 10.0], [8, 8, 8], 2.0, 2.5, 4);

        // Uniform potential (constant everywhere)
        let size = 8 * 8 * 8;
        let values = vec![1.0; size];
        let potential = PotentialMesh::from_values([8, 8, 8], values, [10.0, 10.0, 10.0]);

        let positions = vec![[5.0, 5.0, 5.0], [2.0, 3.0, 4.0]];
        let charges = vec![1.0, 1.0];

        let forces = interpolate_forces_from_positions(&potential, &positions, &charges, &params);

        // Gradient of constant is zero → force should be near zero
        for force in &forces {
            assert!(force[0].abs() < 1e-10, "fx = {}", force[0]);
            assert!(force[1].abs() < 1e-10, "fy = {}", force[1]);
            assert!(force[2].abs() < 1e-10, "fz = {}", force[2]);
        }
    }

    #[test]
    fn test_linear_potential_constant_force() {
        let params = PppmParams::custom(1, [10.0, 10.0, 10.0], [8, 8, 8], 2.0, 2.5, 4);

        // Linear potential: φ(x,y,z) = x
        let dims = [8, 8, 8];
        let mut values = vec![0.0; dims[0] * dims[1] * dims[2]];
        for ix in 0..dims[0] {
            let x = (ix as f64 + 0.5) * 10.0 / 8.0;
            for iy in 0..dims[1] {
                for iz in 0..dims[2] {
                    let idx = iz + dims[2] * (iy + dims[1] * ix);
                    values[idx] = x;
                }
            }
        }
        let potential = PotentialMesh::from_values(dims, values, [10.0, 10.0, 10.0]);

        let positions = vec![[5.0, 5.0, 5.0]];
        let charges = vec![1.0];

        let forces = interpolate_forces_from_positions(&potential, &positions, &charges, &params);

        // ∇φ = (1, 0, 0), so F = -q(1, 0, 0) = (-1, 0, 0)
        assert!(forces[0][0] < 0.0, "fx should be negative");
        assert!(forces[0][1].abs() < 0.5, "fy should be near zero");
        assert!(forces[0][2].abs() < 0.5, "fz should be near zero");
    }

    #[test]
    fn test_force_interpolation_consistency() {
        let params = PppmParams::custom(2, [10.0, 10.0, 10.0], [8, 8, 8], 2.0, 2.5, 4);

        // Create some potential pattern
        let dims = [8, 8, 8];
        let mut values = vec![0.0; dims[0] * dims[1] * dims[2]];
        for ix in 0..dims[0] {
            for iy in 0..dims[1] {
                for iz in 0..dims[2] {
                    let idx = iz + dims[2] * (iy + dims[1] * ix);
                    // Sinusoidal pattern
                    let x = ix as f64 * std::f64::consts::PI / 4.0;
                    values[idx] = x.sin();
                }
            }
        }
        let potential = PotentialMesh::from_values(dims, values, [10.0, 10.0, 10.0]);

        let positions = vec![[5.0, 5.0, 5.0], [2.5, 5.0, 5.0]];
        let charges = vec![1.0, 1.0];

        // Compute via both methods
        let forces1 = interpolate_forces_from_positions(&potential, &positions, &charges, &params);

        // Compute B-spline coefficients manually
        let coeffs: Vec<_> = positions
            .iter()
            .map(|pos| {
                let wrapped = wrap_position(*pos, params.box_dims);
                BsplineCoeffs::compute(
                    params.interpolation_order,
                    wrapped,
                    params.mesh_dims,
                    params.box_dims,
                )
            })
            .collect();

        let forces2 = interpolate_forces(&potential, &charges, &coeffs, &params);

        // Should match
        for i in 0..positions.len() {
            assert!((forces1[i][0] - forces2[i][0]).abs() < 1e-10);
            assert!((forces1[i][1] - forces2[i][1]).abs() < 1e-10);
            assert!((forces1[i][2] - forces2[i][2]).abs() < 1e-10);
        }
    }
}
