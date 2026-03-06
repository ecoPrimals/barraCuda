// SPDX-License-Identifier: AGPL-3.0-or-later
//! Numerical methods for differentiation, integration, and ODEs
//!
//! This module provides standard numerical methods for computing
//! derivatives, integrals, and solving differential equations.
//!
//! # Methods
//!
//! - **`gradient_1d`**: Finite-difference gradients (3-point stencil)
//! - **trapz**: Trapezoidal integration
//! - **`trapz_product`**: Weighted product integration
//! - **`rk45_solve`**: Adaptive Runge-Kutta ODE solver
//!
//! # Examples
//!
//! ```
//! use barracuda::numerical::{gradient_1d, trapz};
//!
//! // Compute gradient of y = x²
//! let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
//! let y: Vec<f64> = x.iter().map(|xi| xi * xi).collect();
//! let dy_dx = gradient_1d(&y, 1.0);  // dx = 1.0
//!
//! // Gradient should be ≈ 2x
//! assert!((dy_dx[2] - 4.0).abs() < 0.1);  // at x=2, dy/dx ≈ 4
//!
//! // Integrate y = x from 0 to 4
//! let integral = trapz(&y, &x)?;
//! assert!((integral - 21.33).abs() < 1.0);  // ∫₀⁴ x² dx = 64/3 ≈ 21.33 (trapezoidal gives ~22)
//! # Ok::<(), barracuda::error::BarracudaError>(())
//! ```
//!
//! ```
//! use barracuda::numerical::rk45::{rk45_solve, Rk45Config};
//!
//! // Solve dy/dt = -y (exponential decay)
//! let f = |_t: f64, y: &[f64]| vec![-y[0]];
//! let config = Rk45Config::new(1e-6, 1e-9);
//!
//! let result = rk45_solve(&f, 0.0, 1.0, &[1.0], &config).unwrap();
//! // y(1) ≈ e^(-1) ≈ 0.368
//! ```

pub mod gradient;
pub mod hessian;
pub mod integrate;
pub mod ode_bio;
pub mod ode_generic;
pub mod rk45;

pub use gradient::gradient_1d;
pub use hessian::numerical_hessian;
pub use integrate::{trapz, trapz_product};
pub use ode_bio::{
    BistableOde, BistableParams, CapacitorOde, CapacitorParams, CooperationOde, CooperationParams,
    MultiSignalOde, MultiSignalParams, PhageDefenseOde, PhageDefenseParams, QsBiofilmParams,
};
pub use ode_generic::{BatchedOdeRK4, OdeSystem};
pub use rk45::{Rk45Config, Rk45Result, rk45_at, rk45_solve};

/// WGSL shader: parallel central-difference Hessian column computation
/// WGSL kernel for Hessian column extraction via finite differences.
#[cfg(feature = "gpu")]
#[must_use]
pub fn wgsl_hessian_column() -> &'static str {
    static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        crate::shaders::precision::downcast_f64_to_f32(include_str!(
            "../shaders/numerical/hessian_column_f64.wgsl"
        ))
    });
    std::sync::LazyLock::force(&SHADER).as_str()
}
