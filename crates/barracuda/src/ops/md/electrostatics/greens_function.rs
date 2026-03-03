//! Green's Function for PPPM
//!
//! The Green's function G(k) transforms charge density in k-space to potential:
//! φ̃(k) = G(k) × ρ̃(k)
//!
//! For Coulomb interactions with Ewald splitting:
//! G(k) = 4π/k² × exp(-k²/(4α²)) × influence_correction
//!
//! The influence correction accounts for B-spline interpolation artifacts.

use std::f64::consts::PI;

use super::bspline::influence_function;
use super::PppmParams;

/// Precomputed Green's function table
///
/// Stores G(k) for all k-vectors on the mesh.
#[derive(Clone, Debug)]
pub struct GreensFunction {
    /// Mesh dimensions [Kx, Ky, Kz]
    pub dims: [usize; 3],

    /// Green's function values (same layout as charge mesh)
    pub values: Vec<f64>,

    /// k-vectors for each mesh point (for debugging/analysis)
    /// Not stored by default to save memory
    #[cfg(feature = "debug-pppm")]
    pub k_vectors: Vec<[f64; 3]>,
}

impl GreensFunction {
    /// Precompute Green's function for given PPPM parameters
    ///
    /// # Arguments
    /// * `params` - PPPM configuration
    ///
    /// # Returns
    /// Green's function table
    pub fn new(params: &PppmParams) -> Self {
        let dims = params.mesh_dims;
        let size = dims[0] * dims[1] * dims[2];
        let mut values = vec![0.0; size];

        let alpha = params.alpha;
        let order = params.interpolation_order;

        // k-space resolution
        let dk = [
            2.0 * PI / params.box_dims[0],
            2.0 * PI / params.box_dims[1],
            2.0 * PI / params.box_dims[2],
        ];

        // Compute G(k) for each k-vector
        for ix in 0..dims[0] {
            // k_x = ix * dk_x for ix < Kx/2, else (ix - Kx) * dk_x
            let kx = if ix <= dims[0] / 2 {
                ix as f64 * dk[0]
            } else {
                (ix as i64 - dims[0] as i64) as f64 * dk[0]
            };

            for iy in 0..dims[1] {
                let ky = if iy <= dims[1] / 2 {
                    iy as f64 * dk[1]
                } else {
                    (iy as i64 - dims[1] as i64) as f64 * dk[1]
                };

                for iz in 0..dims[2] {
                    let kz = if iz <= dims[2] / 2 {
                        iz as f64 * dk[2]
                    } else {
                        (iz as i64 - dims[2] as i64) as f64 * dk[2]
                    };

                    let idx = iz + dims[2] * (iy + dims[1] * ix);
                    values[idx] = compute_g(kx, ky, kz, alpha, order, params.box_dims, dims);
                }
            }
        }

        Self {
            dims,
            values,
            #[cfg(feature = "debug-pppm")]
            k_vectors: Vec::new(),
        }
    }

    /// Get Green's function value at mesh indices
    #[inline]
    pub fn get(&self, ix: usize, iy: usize, iz: usize) -> f64 {
        let idx = iz + self.dims[2] * (iy + self.dims[1] * ix);
        self.values[idx]
    }

    /// Apply Green's function in k-space
    ///
    /// Multiplies each element: φ̃(k) = G(k) × ρ̃(k)
    ///
    /// # Arguments
    /// * `rho_k` - Charge density in k-space (complex, interleaved re/im)
    ///
    /// # Returns
    /// Potential in k-space (complex, interleaved re/im)
    pub fn apply(&self, rho_k: &[f64]) -> Vec<f64> {
        let size = self.dims[0] * self.dims[1] * self.dims[2];
        assert_eq!(rho_k.len(), size * 2); // Complex values

        let mut phi_k = vec![0.0; size * 2];

        for i in 0..size {
            let g = self.values[i];
            let re_idx = i * 2;
            let im_idx = i * 2 + 1;

            // φ̃ = G × ρ̃ (G is real, so just scale both components)
            phi_k[re_idx] = g * rho_k[re_idx];
            phi_k[im_idx] = g * rho_k[im_idx];
        }

        phi_k
    }

    /// Apply Green's function in-place
    ///
    /// # Arguments
    /// * `rho_k` - Charge density in k-space (modified in place to potential)
    pub fn apply_inplace(&self, rho_k: &mut [f64]) {
        let size = self.dims[0] * self.dims[1] * self.dims[2];
        assert_eq!(rho_k.len(), size * 2);

        for i in 0..size {
            let g = self.values[i];
            let re_idx = i * 2;
            let im_idx = i * 2 + 1;

            rho_k[re_idx] *= g;
            rho_k[im_idx] *= g;
        }
    }

    /// Compute k-space energy contribution
    ///
    /// E_k = (1/2V) Σ_k |ρ̃(k)|² G(k)
    ///
    /// # Arguments
    /// * `rho_k` - Charge density in k-space (complex, interleaved)
    ///
    /// # Returns
    /// k-space energy contribution
    pub fn kspace_energy(&self, rho_k: &[f64], volume: f64) -> f64 {
        let size = self.dims[0] * self.dims[1] * self.dims[2];
        assert_eq!(rho_k.len(), size * 2);

        let mut energy = 0.0;

        for i in 0..size {
            let g = self.values[i];
            let re = rho_k[i * 2];
            let im = rho_k[i * 2 + 1];
            let rho_sq = re * re + im * im;

            energy += g * rho_sq;
        }

        // Factor of 1/2 from double counting, 1/V from normalization
        0.5 * energy / volume
    }
}

/// Compute Green's function for a single k-vector
///
/// G(k) = 4π/k² × exp(-k²/(4α²)) × influence(k)
fn compute_g(
    kx: f64,
    ky: f64,
    kz: f64,
    alpha: f64,
    order: usize,
    box_dims: [f64; 3],
    mesh_dims: [usize; 3],
) -> f64 {
    let k_sq = kx * kx + ky * ky + kz * kz;

    // k = 0 gives no contribution
    if k_sq < 1e-14 {
        return 0.0;
    }

    // Ewald damping: exp(-k²/(4α²))
    let ewald_factor = (-k_sq / (4.0 * alpha * alpha)).exp();

    // Bare Coulomb: 4π/k²
    let coulomb = 4.0 * PI / k_sq;

    // B-spline influence correction
    let influence = influence_function(kx, ky, kz, order, box_dims, mesh_dims);

    // Combined Green's function
    // Note: influence_function already includes 4π/k², so we just use the ratio
    coulomb * ewald_factor * (influence / coulomb).clamp(0.0, 10.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::md::electrostatics::PppmAccuracy;

    #[test]
    fn test_greens_function_creation() {
        let params = PppmParams::auto(1000, 10.0, PppmAccuracy::Medium);
        let g = GreensFunction::new(&params);

        // G(0,0,0) should be 0
        assert!(g.get(0, 0, 0).abs() < 1e-10);

        // G should be non-negative everywhere
        for v in &g.values {
            assert!(*v >= 0.0);
        }
    }

    #[test]
    fn test_greens_function_symmetry() {
        let params = PppmParams::auto(100, 10.0, PppmAccuracy::Medium);
        let g = GreensFunction::new(&params);

        // Should have inversion symmetry: G(k) = G(-k)
        let dims = g.dims;
        for ix in 1..dims[0] {
            let ix_neg = dims[0] - ix;
            for iy in 1..dims[1] {
                let iy_neg = dims[1] - iy;
                for iz in 1..dims[2] {
                    let iz_neg = dims[2] - iz;

                    let g1 = g.get(ix, iy, iz);
                    let g2 = g.get(ix_neg, iy_neg, iz_neg);

                    assert!(
                        (g1 - g2).abs() < 1e-10,
                        "G({},{},{}) = {} != G({},{},{}) = {}",
                        ix,
                        iy,
                        iz,
                        g1,
                        ix_neg,
                        iy_neg,
                        iz_neg,
                        g2
                    );
                }
            }
        }
    }

    #[test]
    fn test_greens_function_decay() {
        let params = PppmParams::auto(100, 10.0, PppmAccuracy::Medium);
        let g = GreensFunction::new(&params);

        // G should decay with increasing |k|
        let g_small_k = g.get(1, 0, 0);
        let g_large_k = g.get(4, 0, 0);

        assert!(g_large_k < g_small_k);
    }

    #[test]
    fn test_apply_greens_function() {
        let params = PppmParams::custom(100, [10.0, 10.0, 10.0], [8, 8, 8], 2.0, 2.5, 4);
        let g = GreensFunction::new(&params);

        // Create test charge density (all zeros except one point)
        let size = 8 * 8 * 8;
        let mut rho_k = vec![0.0; size * 2];
        rho_k[2] = 1.0; // Set rho(1,0,0) = 1+0i

        let phi_k = g.apply(&rho_k);

        // φ(1,0,0) should be G(1,0,0)
        assert!((phi_k[2] - g.get(1, 0, 0)).abs() < 1e-10);
    }

    #[test]
    fn test_kspace_energy() {
        let params = PppmParams::custom(100, [10.0, 10.0, 10.0], [8, 8, 8], 2.0, 2.5, 4);
        let g = GreensFunction::new(&params);

        // Test with simple charge distribution
        let size = 8 * 8 * 8;
        let mut rho_k = vec![0.0; size * 2];
        rho_k[2] = 1.0; // Single non-zero mode

        let volume = 10.0 * 10.0 * 10.0;
        let energy = g.kspace_energy(&rho_k, volume);

        // Energy should be positive
        assert!(energy > 0.0);
    }
}
