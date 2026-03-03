//! RBF kernel functions

/// Radial basis function kernel types
#[derive(Debug, Clone, Copy)]
pub enum RBFKernel {
    /// Thin plate spline: φ(r²) = 0.5·r²·ln(r²)
    ///
    /// Used in Diaw et al. for nuclear physics surrogate modeling.
    /// Exact interpolation of training data with good smoothness.
    ThinPlateSpline,

    /// Gaussian: φ(r) = exp(-ε²r²)
    ///
    /// Popular general-purpose kernel. Requires shape parameter ε.
    Gaussian { epsilon: f64 },

    /// Multiquadric: φ(r) = √(1 + ε²r²)
    ///
    /// Good for scattered data interpolation.
    Multiquadric { epsilon: f64 },

    /// Inverse multiquadric: φ(r) = 1/√(1 + ε²r²)
    ///
    /// Produces smoother interpolants than multiquadric.
    InverseMultiquadric { epsilon: f64 },

    /// Cubic: φ(r) = r³
    ///
    /// Simple kernel for 1D/2D interpolation.
    Cubic,

    /// Quintic: φ(r) = r⁵
    ///
    /// Higher-order smoothness than cubic.
    Quintic,
}

impl RBFKernel {
    /// Evaluate kernel at distance r
    pub fn eval(&self, r: f64) -> f64 {
        match self {
            RBFKernel::ThinPlateSpline => {
                if r < 1e-14 {
                    0.0 // Limit as r → 0
                } else {
                    let r2 = r * r;
                    0.5 * r2 * r2.ln()
                }
            }
            RBFKernel::Gaussian { epsilon } => {
                let er = epsilon * r;
                (-er * er).exp()
            }
            RBFKernel::Multiquadric { epsilon } => {
                let er2 = (epsilon * r).powi(2);
                (1.0 + er2).sqrt()
            }
            RBFKernel::InverseMultiquadric { epsilon } => {
                let er2 = (epsilon * r).powi(2);
                1.0 / (1.0 + er2).sqrt()
            }
            RBFKernel::Cubic => r.powi(3),
            RBFKernel::Quintic => r.powi(5),
        }
    }

    /// Name of the kernel
    pub fn name(&self) -> &str {
        match self {
            RBFKernel::ThinPlateSpline => "ThinPlateSpline",
            RBFKernel::Gaussian { .. } => "Gaussian",
            RBFKernel::Multiquadric { .. } => "Multiquadric",
            RBFKernel::InverseMultiquadric { .. } => "InverseMultiquadric",
            RBFKernel::Cubic => "Cubic",
            RBFKernel::Quintic => "Quintic",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tps_kernel_at_zero() {
        let kernel = RBFKernel::ThinPlateSpline;
        assert_eq!(kernel.eval(0.0), 0.0);
    }

    #[test]
    fn test_tps_kernel_nonzero() {
        let kernel = RBFKernel::ThinPlateSpline;
        let r = 2.0;
        let expected = 0.5 * 4.0 * (4.0_f64).ln(); // 0.5 * r² * ln(r²)
        assert!((kernel.eval(r) - expected).abs() < 1e-12);
    }

    #[test]
    fn test_gaussian_kernel() {
        let kernel = RBFKernel::Gaussian { epsilon: 1.0 };
        assert!((kernel.eval(0.0) - 1.0).abs() < 1e-14);
        assert!((kernel.eval(1.0) - (-1.0_f64).exp()).abs() < 1e-14);
    }

    #[test]
    fn test_multiquadric_kernel() {
        let kernel = RBFKernel::Multiquadric { epsilon: 1.0 };
        assert!((kernel.eval(0.0) - 1.0).abs() < 1e-14);
        assert!((kernel.eval(1.0) - 2.0_f64.sqrt()).abs() < 1e-14);
    }

    #[test]
    fn test_cubic_kernel() {
        let kernel = RBFKernel::Cubic;
        assert_eq!(kernel.eval(0.0), 0.0);
        assert_eq!(kernel.eval(2.0), 8.0);
    }
}
