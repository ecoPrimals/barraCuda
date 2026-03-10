// SPDX-License-Identifier: AGPL-3.0-only
//! PPPM Parameter Configuration
//!
//! Provides automatic parameter selection for desired accuracy.

use std::f64::consts::PI;

/// Accuracy level for PPPM electrostatics
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PppmAccuracy {
    /// Low accuracy (~1e-3 relative force error), fastest
    Low,
    /// Medium accuracy (~1e-5 relative force error), balanced
    Medium,
    /// High accuracy (~1e-7 relative force error), slowest
    High,
    /// Custom accuracy with specified relative force error
    Custom(f64),
}

impl PppmAccuracy {
    /// Get the target relative force error
    #[must_use]
    pub fn target_error(&self) -> f64 {
        match self {
            PppmAccuracy::Low => 1e-3,
            PppmAccuracy::Medium => 1e-5,
            PppmAccuracy::High => 1e-7,
            PppmAccuracy::Custom(e) => *e,
        }
    }
}

/// PPPM configuration parameters
///
/// All parameters are in reduced units (charges in units of elementary charge,
/// lengths in units of Wigner-Seitz radius for plasmas, etc.)
#[derive(Clone, Debug)]
pub struct PppmParams {
    /// Box dimensions [Lx, Ly, Lz]
    pub box_dims: [f64; 3],

    /// Number of particles
    pub n_particles: usize,

    /// Mesh dimensions [Kx, Ky, Kz] — must be powers of 2 for FFT
    pub mesh_dims: [usize; 3],

    /// Ewald splitting parameter α (inverse length)
    pub alpha: f64,

    /// Real-space cutoff rc
    pub real_cutoff: f64,

    /// B-spline interpolation order (typically 4-7)
    pub interpolation_order: usize,

    /// Coulomb constant (1.0 in Gaussian units, 8.9875e9 in SI)
    pub coulomb_constant: f64,

    /// Target accuracy level
    pub accuracy: PppmAccuracy,
}

impl PppmParams {
    /// Create PPPM parameters with automatic optimization
    ///
    /// # Arguments
    /// * `n_particles` - Number of particles
    /// * `box_side` - Box side length (cubic box)
    /// * `accuracy` - Desired accuracy level
    ///
    /// # Returns
    /// Optimized PPPM parameters
    #[must_use]
    pub fn auto(n_particles: usize, box_side: f64, accuracy: PppmAccuracy) -> Self {
        Self::auto_with_dims(n_particles, [box_side; 3], accuracy)
    }

    /// Create PPPM parameters for non-cubic box
    ///
    /// # Arguments
    /// * `n_particles` - Number of particles
    /// * `box_dims` - Box dimensions [Lx, Ly, Lz]
    /// * `accuracy` - Desired accuracy level
    pub fn auto_with_dims(n_particles: usize, box_dims: [f64; 3], accuracy: PppmAccuracy) -> Self {
        let _target_error = accuracy.target_error();

        // Heuristic: interpolation order based on accuracy
        let interpolation_order = match accuracy {
            PppmAccuracy::Low => 4,
            PppmAccuracy::Medium => 5,
            PppmAccuracy::High => 7,
            PppmAccuracy::Custom(e) if e > 1e-4 => 4,
            PppmAccuracy::Custom(e) if e > 1e-6 => 5,
            PppmAccuracy::Custom(_) => 7,
        };

        // Minimum box dimension
        let l_min = box_dims.iter().copied().fold(f64::INFINITY, f64::min);

        // Real-space cutoff: use 1/4 of box for good balance
        let real_cutoff = l_min / 4.0;

        // Alpha: balance real-space and k-space work
        // Larger alpha → fewer k-modes but more real-space work
        // Rule of thumb: α ≈ 5/rc for decent balance
        let alpha = 5.0 / real_cutoff;

        // Mesh size: based on box size and accuracy
        // Higher accuracy → finer mesh
        let base_mesh = match accuracy {
            PppmAccuracy::Low => 16,
            PppmAccuracy::Medium => 32,
            PppmAccuracy::High => 64,
            PppmAccuracy::Custom(e) if e > 1e-4 => 16,
            PppmAccuracy::Custom(e) if e > 1e-6 => 32,
            PppmAccuracy::Custom(_) => 64,
        };

        // Scale mesh with box size (keep roughly same resolution)
        let mesh_scale = (l_min / 10.0).max(1.0);
        let mesh_size = next_power_of_2((base_mesh as f64 * mesh_scale) as usize);

        // Mesh dimensions (can be different for non-cubic boxes)
        let mesh_dims = [
            next_power_of_2((mesh_size as f64 * box_dims[0] / l_min) as usize),
            next_power_of_2((mesh_size as f64 * box_dims[1] / l_min) as usize),
            next_power_of_2((mesh_size as f64 * box_dims[2] / l_min) as usize),
        ];

        Self {
            box_dims,
            n_particles,
            mesh_dims,
            alpha,
            real_cutoff,
            interpolation_order,
            coulomb_constant: 1.0, // Gaussian units default
            accuracy,
        }
    }

    /// Create custom PPPM parameters
    #[must_use]
    pub fn custom(
        n_particles: usize,
        box_dims: [f64; 3],
        mesh_dims: [usize; 3],
        alpha: f64,
        real_cutoff: f64,
        interpolation_order: usize,
    ) -> Self {
        Self {
            box_dims,
            n_particles,
            mesh_dims,
            alpha,
            real_cutoff,
            interpolation_order,
            coulomb_constant: 1.0,
            accuracy: PppmAccuracy::Custom(1e-5),
        }
    }

    /// Set Coulomb constant (for unit system conversion)
    #[must_use]
    pub fn with_coulomb_constant(mut self, k: f64) -> Self {
        self.coulomb_constant = k;
        self
    }

    /// Estimate memory usage in bytes
    #[must_use]
    pub fn estimated_memory(&self) -> usize {
        let mesh_elements = self.mesh_dims[0] * self.mesh_dims[1] * self.mesh_dims[2];

        // Complex mesh (2 f64 per element)
        let mesh_mem = mesh_elements * 2 * 8;

        // Green's function table (1 f64 per k-vector)
        let green_mem = mesh_elements * 8;

        // B-spline coefficients (order * n_particles)
        let bspline_mem = self.interpolation_order * self.n_particles * 8 * 3;

        mesh_mem * 2 + green_mem + bspline_mem
    }

    /// Estimate k-space cutoff (in units of 2π/L)
    #[must_use]
    pub fn k_cutoff(&self) -> f64 {
        // k_max = π * mesh / L (Nyquist)
        let k_max_x = PI * self.mesh_dims[0] as f64 / self.box_dims[0];
        let k_max_y = PI * self.mesh_dims[1] as f64 / self.box_dims[1];
        let k_max_z = PI * self.mesh_dims[2] as f64 / self.box_dims[2];

        k_max_x.min(k_max_y).min(k_max_z)
    }

    /// Estimated real-space force error
    ///
    /// Based on erfc decay: error ∝ erfc(α*rc) / rc²
    #[must_use]
    pub fn estimated_real_error(&self) -> f64 {
        let x = self.alpha * self.real_cutoff;
        // erfc(x) ≈ exp(-x²) / (x * √π) for large x
        (-x * x).exp() / (x * PI.sqrt()) / (self.real_cutoff * self.real_cutoff)
    }

    /// Estimated k-space force error
    ///
    /// Based on mesh aliasing and interpolation order
    #[must_use]
    pub fn estimated_kspace_error(&self) -> f64 {
        let h = self.box_dims[0] / self.mesh_dims[0] as f64; // mesh spacing
        let order = self.interpolation_order as f64;

        // Interpolation error scales as h^order
        h.powf(order) * (2.0 * PI / self.box_dims[0]).powi(2)
    }

    /// Print parameter summary
    pub fn print_summary(&self) {
        tracing::info!(
            "PPPM Parameters: Box: [{:.3}, {:.3}, {:.3}], Particles: {}",
            self.box_dims[0],
            self.box_dims[1],
            self.box_dims[2],
            self.n_particles
        );
        tracing::info!(
            "  Mesh: {}×{}×{}, α = {:.4}, rc = {:.4}, order: {}",
            self.mesh_dims[0],
            self.mesh_dims[1],
            self.mesh_dims[2],
            self.alpha,
            self.real_cutoff,
            self.interpolation_order
        );
        tracing::info!(
            "  Est. memory: {:.1} MB, real error: {:.2e}, k-space error: {:.2e}",
            self.estimated_memory() as f64 / 1e6,
            self.estimated_real_error(),
            self.estimated_kspace_error()
        );
    }
}

/// Round up to next power of 2
fn next_power_of_2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p *= 2;
    }
    p.max(8) // Minimum mesh size of 8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pppm_params_auto() {
        let params = PppmParams::auto(1000, 10.0, PppmAccuracy::Medium);

        // Verify reasonable defaults
        assert_eq!(params.box_dims, [10.0, 10.0, 10.0]);
        assert_eq!(params.n_particles, 1000);
        assert!(params.mesh_dims[0].is_power_of_two());
        assert!(params.mesh_dims[1].is_power_of_two());
        assert!(params.mesh_dims[2].is_power_of_two());
        assert!(params.alpha > 0.0);
        assert!(params.real_cutoff > 0.0);
        assert!(params.real_cutoff < 10.0); // Must be less than box
    }

    #[test]
    fn test_pppm_params_accuracy_levels() {
        let low = PppmParams::auto(1000, 10.0, PppmAccuracy::Low);
        let high = PppmParams::auto(1000, 10.0, PppmAccuracy::High);

        // Higher accuracy should use finer mesh
        assert!(high.mesh_dims[0] >= low.mesh_dims[0]);
        // Higher accuracy should use higher interpolation order
        assert!(high.interpolation_order >= low.interpolation_order);
    }

    #[test]
    fn test_pppm_params_memory_estimate() {
        let params = PppmParams::auto(10_000, 20.0, PppmAccuracy::Medium);
        let mem = params.estimated_memory();

        // Should be reasonable (< 1 GB for typical parameters)
        assert!(mem < 1_000_000_000);
        assert!(mem > 0);
    }

    #[test]
    fn test_next_power_of_2() {
        assert_eq!(next_power_of_2(1), 8); // Minimum is 8
        assert_eq!(next_power_of_2(7), 8);
        assert_eq!(next_power_of_2(8), 8);
        assert_eq!(next_power_of_2(9), 16);
        assert_eq!(next_power_of_2(31), 32);
        assert_eq!(next_power_of_2(32), 32);
        assert_eq!(next_power_of_2(33), 64);
    }

    #[test]
    fn test_non_cubic_box() {
        let params = PppmParams::auto_with_dims(1000, [10.0, 20.0, 15.0], PppmAccuracy::Medium);

        // Should have non-equal mesh dimensions
        assert_eq!(params.box_dims, [10.0, 20.0, 15.0]);
        // Real cutoff should be based on smallest dimension
        assert!(params.real_cutoff <= 10.0 / 4.0 + 0.01);
    }
}
