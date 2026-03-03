// SPDX-License-Identifier: AGPL-3.0-or-later
//! B-spline Functions for PPPM
//!
//! Cardinal B-splines are used for charge spreading and force interpolation
//! in the PPPM algorithm. They provide smooth, local interpolation between
//! particles and mesh nodes.
//!
//! # Mathematical Background
//!
//! The cardinal B-spline of order p is defined recursively:
//! ```text
//! M_1(x) = 1 if 0 ≤ x < 1, else 0
//! M_p(x) = (x/(p-1)) M_{p-1}(x) + ((p-x)/(p-1)) M_{p-1}(x-1)
//! ```
//!
//! Properties:
//! - Non-negative: M_p(x) ≥ 0
//! - Compact support: M_p(x) = 0 for x ≤ 0 or x ≥ p
//! - Partition of unity: Σ M_p(x-n) = 1 for all x
//! - Smooth: M_p ∈ C^{p-2} (p-2 continuous derivatives)
//!
//! # Usage in PPPM
//!
//! 1. **Charge spreading**: Each particle contributes to p³ mesh nodes
//! 2. **Force interpolation**: Each mesh node contributes to nearby particles
//!
//! The spreading/interpolation stencil is p × p × p nodes centered on the
//! particle's mesh cell.

/// WGSL kernel for cardinal B-spline evaluation (f64).
pub const WGSL_BSPLINE_F64: &str = include_str!("bspline_f64.wgsl");

use std::f64::consts::PI;

/// Cardinal B-spline of order p evaluated at x
///
/// Uses explicit formula for stability:
/// M_p(x) = 1/(p-1)! × Σ_{k=0}^{p} (-1)^k × C(p,k) × max(x-k, 0)^{p-1}
///
/// # Arguments
/// * `order` - B-spline order (typically 4-7 for PPPM)
/// * `x` - Evaluation point (should be in [0, order] for non-zero result)
///
/// # Returns
/// B-spline value M_order(x)
pub fn bspline(order: usize, x: f64) -> f64 {
    if order == 0 {
        return 0.0;
    }
    if order == 1 {
        return if (0.0..1.0).contains(&x) { 1.0 } else { 0.0 };
    }

    let order_f = order as f64;
    if x <= 0.0 || x >= order_f {
        return 0.0;
    }

    let p = order;
    let pm1 = (p - 1) as i32;

    // Compute factorial (p-1)!
    let mut factorial = 1.0;
    for i in 2..=pm1 {
        factorial *= i as f64;
    }

    // Sum over k
    let mut sum = 0.0;
    for k in 0..=p {
        let xmk = x - k as f64;
        if xmk > 0.0 {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            let binom = binomial(p, k) as f64;
            sum += sign * binom * xmk.powi(pm1);
        }
    }

    sum / factorial
}

/// Binomial coefficient C(n, k)
fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k); // Use symmetry
    let mut result = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Derivative of cardinal B-spline
///
/// dM_p(x)/dx = M_{p-1}(x) - M_{p-1}(x-1)
///
/// # Arguments
/// * `order` - B-spline order
/// * `x` - Evaluation point
///
/// # Returns
/// Derivative dM_order(x)/dx
pub fn bspline_deriv(order: usize, x: f64) -> f64 {
    if order <= 1 {
        return 0.0;
    }

    bspline(order - 1, x) - bspline(order - 1, x - 1.0)
}

/// Precomputed B-spline coefficients for a particle
///
/// Stores the B-spline values for all nodes in the interpolation stencil.
#[derive(Clone, Debug)]
pub struct BsplineCoeffs {
    /// B-spline order
    pub order: usize,

    /// Grid index of leftmost node in stencil (for each dimension)
    pub base_idx: [i32; 3],

    /// B-spline values W_k for k = 0..order
    /// Stored as [x_coeffs, y_coeffs, z_coeffs]
    pub coeffs: [Vec<f64>; 3],

    /// B-spline derivatives for force interpolation
    pub derivs: [Vec<f64>; 3],
}

impl BsplineCoeffs {
    /// Compute B-spline coefficients for a particle at position (x, y, z)
    ///
    /// Uses the standard PPPM charge assignment scheme. For a particle at
    /// continuous position u and order p:
    /// - Base index n0 = floor(u) - (p-2)/2  (for even p)
    ///   = floor(u) - (p-1)/2  (for odd p)
    /// - Weight W_k = M_p(u - n0 - k) for k = 0, ..., p-1
    ///
    /// The weights satisfy: Σ_k W_k = 1 (partition of unity)
    ///
    /// # Arguments
    /// * `order` - B-spline order (typically 4-7)
    /// * `pos` - Particle position [x, y, z] in box coordinates
    /// * `mesh_dims` - Mesh dimensions [Kx, Ky, Kz]
    /// * `box_dims` - Box dimensions [Lx, Ly, Lz]
    pub fn compute(order: usize, pos: [f64; 3], mesh_dims: [usize; 3], box_dims: [f64; 3]) -> Self {
        let mut coeffs = [
            Vec::with_capacity(order),
            Vec::with_capacity(order),
            Vec::with_capacity(order),
        ];
        let mut derivs = [
            Vec::with_capacity(order),
            Vec::with_capacity(order),
            Vec::with_capacity(order),
        ];
        let mut base_idx = [0i32; 3];

        let p = order;

        for d in 0..3 {
            // Convert position to mesh units (scaled to [0, K))
            let u = pos[d] / box_dims[d] * mesh_dims[d] as f64;

            // Base index: first node in the p-node stencil
            // M_p(u - n) is nonzero when 0 < u - n < p, i.e., (u - p) < n < u
            // The first contributing node is ceil(u - p + epsilon) = floor(u) - p + 1
            base_idx[d] = u.floor() as i32 - p as i32 + 1;

            // Compute weights: W_k = M_p(u - (base_idx + k))
            // The argument (u - node_index) should be in (0, p) for M_p's support
            for k in 0..p {
                let node_pos = (base_idx[d] + k as i32) as f64;
                let arg = u - node_pos;
                coeffs[d].push(bspline(p, arg));
                derivs[d].push(bspline_deriv(p, arg));
            }
        }

        Self {
            order,
            base_idx,
            coeffs,
            derivs,
        }
    }

    /// Get the mesh index for stencil position (ix, iy, iz) with periodic wrapping
    ///
    /// # Arguments
    /// * `ix, iy, iz` - Stencil indices (0 to order-1)
    /// * `mesh_dims` - Mesh dimensions for periodic wrapping
    pub fn mesh_index(&self, ix: usize, iy: usize, iz: usize, mesh_dims: [usize; 3]) -> [usize; 3] {
        [
            ((self.base_idx[0] + ix as i32).rem_euclid(mesh_dims[0] as i32)) as usize,
            ((self.base_idx[1] + iy as i32).rem_euclid(mesh_dims[1] as i32)) as usize,
            ((self.base_idx[2] + iz as i32).rem_euclid(mesh_dims[2] as i32)) as usize,
        ]
    }

    /// Get the weight for stencil position (ix, iy, iz)
    ///
    /// This is the product of 1D B-spline values: w = W_x × W_y × W_z
    pub fn weight(&self, ix: usize, iy: usize, iz: usize) -> f64 {
        self.coeffs[0][ix] * self.coeffs[1][iy] * self.coeffs[2][iz]
    }

    /// Get the gradient weights for stencil position (ix, iy, iz)
    ///
    /// Returns [dw/dx, dw/dy, dw/dz] for force interpolation.
    /// The derivatives need to be scaled by mesh_dims/box_dims.
    pub fn gradient_weights(&self, ix: usize, iy: usize, iz: usize) -> [f64; 3] {
        [
            self.derivs[0][ix] * self.coeffs[1][iy] * self.coeffs[2][iz],
            self.coeffs[0][ix] * self.derivs[1][iy] * self.coeffs[2][iz],
            self.coeffs[0][ix] * self.coeffs[1][iy] * self.derivs[2][iz],
        ]
    }

    /// Sum of all weights (should be 1.0 for correctly computed coefficients)
    pub fn weight_sum(&self) -> f64 {
        let mut sum = 0.0;
        for ix in 0..self.order {
            for iy in 0..self.order {
                for iz in 0..self.order {
                    sum += self.weight(ix, iy, iz);
                }
            }
        }
        sum
    }

    /// Sum of 1D weights (should each be 1.0)
    pub fn weight_sum_1d(&self) -> [f64; 3] {
        [
            self.coeffs[0].iter().sum(),
            self.coeffs[1].iter().sum(),
            self.coeffs[2].iter().sum(),
        ]
    }
}

/// Influence function G(k) for PPPM
///
/// The influence function corrects for the finite mesh and interpolation
/// scheme to achieve accurate forces.
///
/// G(k) = 4π/k² × [reference charge / actual charge]²
///
/// where the ratio accounts for B-spline aliasing.
pub fn influence_function(
    kx: f64,
    ky: f64,
    kz: f64,
    order: usize,
    box_dims: [f64; 3],
    mesh_dims: [usize; 3],
) -> f64 {
    let k_sq = kx * kx + ky * ky + kz * kz;

    // Handle k = 0 (no contribution)
    if k_sq < 1e-14 {
        return 0.0;
    }

    // Mesh spacing
    let hx = box_dims[0] / mesh_dims[0] as f64;
    let hy = box_dims[1] / mesh_dims[1] as f64;
    let hz = box_dims[2] / mesh_dims[2] as f64;

    // B-spline Fourier transform squared
    let bx = bspline_ft_squared(kx * hx / (2.0 * PI), order);
    let by = bspline_ft_squared(ky * hy / (2.0 * PI), order);
    let bz = bspline_ft_squared(kz * hz / (2.0 * PI), order);

    // Aliasing sum correction
    // Sum over periodic images: Σ_m |M̃(k + 2πm/h)|²
    // For most k, only m=0 contributes significantly
    let denominator = bx * by * bz;

    if denominator < 1e-14 {
        return 0.0;
    }

    // G(k) = 4π/k² × 1/|M̃(k)|²
    4.0 * PI / k_sq / denominator
}

/// B-spline Fourier transform magnitude squared
///
/// |M̃_p(ξ)|² = (sin(πξ)/(πξ))^(2p) for ξ ≠ 0
///           = 1 for ξ = 0
fn bspline_ft_squared(xi: f64, order: usize) -> f64 {
    if xi.abs() < 1e-10 {
        return 1.0;
    }

    let sinc = (PI * xi).sin() / (PI * xi);
    sinc.powi(2 * order as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bspline_order_1() {
        // Box function
        assert!((bspline(1, 0.5) - 1.0).abs() < 1e-10);
        assert!(bspline(1, -0.1).abs() < 1e-10);
        assert!(bspline(1, 1.1).abs() < 1e-10);
    }

    #[test]
    fn test_bspline_order_2() {
        // Linear (tent) function, peak at x=1
        assert!((bspline(2, 1.0) - 1.0).abs() < 1e-10); // Peak at center
        assert!(bspline(2, 0.0).abs() < 1e-10);
        assert!(bspline(2, 2.0).abs() < 1e-10);
        // Linear interpolation
        assert!((bspline(2, 0.5) - 0.5).abs() < 1e-10);
        assert!((bspline(2, 1.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_bspline_order_4() {
        // Cubic B-spline should be symmetric around center (x=2)
        let mid = 2.0;
        let v1 = bspline(4, mid - 0.5);
        let v2 = bspline(4, mid + 0.5);
        assert!((v1 - v2).abs() < 1e-10);

        // Peak at center
        assert!(bspline(4, mid) > bspline(4, mid + 0.5));
    }

    #[test]
    fn test_bspline_partition_of_unity() {
        // Sum of B-splines over integer shifts should equal 1
        for order in 2..=6 {
            let x = 2.37; // Arbitrary point
            let mut sum = 0.0;
            for n in -5..15 {
                sum += bspline(order, x - n as f64);
            }
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "Partition of unity failed for order {}: sum = {}",
                order,
                sum
            );
        }
    }

    #[test]
    fn test_bspline_coeffs_sum_to_one() {
        // Test that 1D weights sum to 1 for various positions
        for order in 2..=6 {
            for i in 0..10 {
                let pos = [
                    (i as f64) * 0.73 + 0.1,
                    (i as f64) * 0.91 + 0.2,
                    (i as f64) * 0.67 + 0.3,
                ];
                let mesh_dims = [16, 16, 16];
                let box_dims = [10.0, 10.0, 10.0];

                let coeffs = BsplineCoeffs::compute(order, pos, mesh_dims, box_dims);
                let sums = coeffs.weight_sum_1d();

                for (d, s) in sums.iter().enumerate() {
                    assert!(
                        (s - 1.0).abs() < 1e-10,
                        "Order {} dim {} at {:?}: 1D sum = {} (expected 1)",
                        order,
                        d,
                        pos,
                        s
                    );
                }
            }
        }
    }

    #[test]
    fn test_bspline_coeffs_3d_sum() {
        let pos = [5.0, 5.0, 5.0];
        let mesh_dims = [16, 16, 16];
        let box_dims = [10.0, 10.0, 10.0];

        for order in 2..=6 {
            let coeffs = BsplineCoeffs::compute(order, pos, mesh_dims, box_dims);
            let sum = coeffs.weight_sum();

            assert!(
                (sum - 1.0).abs() < 1e-10,
                "Order {}: 3D sum = {} (expected 1)",
                order,
                sum
            );
        }
    }

    #[test]
    fn test_influence_function() {
        let order = 4;
        let box_dims = [10.0, 10.0, 10.0];
        let mesh_dims = [16, 16, 16];

        // k = 0 should give 0
        assert!(influence_function(0.0, 0.0, 0.0, order, box_dims, mesh_dims).abs() < 1e-10);

        // Non-zero k should give positive value
        let g = influence_function(1.0, 0.0, 0.0, order, box_dims, mesh_dims);
        assert!(g > 0.0);

        // Should decrease with increasing k (roughly as 1/k²)
        let g1 = influence_function(1.0, 0.0, 0.0, order, box_dims, mesh_dims);
        let g2 = influence_function(2.0, 0.0, 0.0, order, box_dims, mesh_dims);
        assert!(g2 < g1);
    }
}
