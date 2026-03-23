// SPDX-License-Identifier: AGPL-3.0-or-later
//! Special mathematical functions
//!
//! This module provides special functions commonly needed in scientific
//! computing: gamma function, factorials, orthogonal polynomials, Bessel
//! functions, error functions, and spherical harmonics.
//!
//! # Architecture
//!
//! Each function has two paths:
//! - **CPU (f64)**: High precision, single values or small batches
//! - **GPU (f32)**: Batch processing via WGSL shaders
//!
//! # Functions
//!
//! | Function | CPU | GPU | Reference |
//! |----------|-----|-----|-----------|
//! | `gamma(x)` | ✅ | — | Lanczos approximation |
//! | `ln_gamma(x)` | ✅ | ✅ | Log-gamma |
//! | `digamma(x)` | ✅ | — | ψ(x) = Γ'(x)/Γ(x) |
//! | `beta(a, b)` | ✅ | — | B(a,b) = Γ(a)Γ(b)/Γ(a+b) |
//! | `ln_beta(a, b)` | ✅ | — | ln(B(a,b)) |
//! | `regularized_gamma_p(a, x)` | ✅ | — | P(a,x) = γ(a,x)/Γ(a) |
//! | `regularized_gamma_q(a, x)` | ✅ | — | Q(a,x) = 1 - P(a,x) |
//! | `chi_squared_cdf(x, k)` | ✅ | — | Chi² CDF via gamma |
//! | `chi_squared_quantile(p, k)` | ✅ | — | Chi² inverse CDF |
//! | `chi_squared_test(obs, exp)` | ✅ | — | Goodness-of-fit test |
//! | `factorial(n)` | ✅ | — | Stirling for large n |
//! | `laguerre(n, α, x)` | ✅ | — | Recurrence relation |
//! | `erf(x)` | ✅ | ✅ | A&S 7.1.26 |
//! | `erfc(x)` | ✅ | ✅ | A&S 7.1.23 |
//! | `bessel_j0(x)` | ✅ | ✅ | A&S 9.4.1-9.4.3 |
//! | `bessel_j1(x)` | ✅ | ✅ | A&S 9.4.4-9.4.6 |
//! | `bessel_i0(x)` | ✅ | ✅ | A&S 9.8.1-9.8.2 |
//! | `bessel_k0(x)` | ✅ | ✅ | A&S 9.8.5-9.8.6 |
//! | `hermite(n, x)` | ✅ | ✅ | Physicist's Hermite Hₙ(x) |
//! | `legendre(n, x)` | ✅ | ✅ | Legendre Pₙ(x) |
//! | `assoc_legendre(n, m, x)` | ✅ | — | Associated Pₙᵐ(x) |
//! | `plasma_dispersion_z(z)` | ✅ | ✅ | Z(z) power series / asymptotic |
//! | `plasma_dispersion_w(z)` | ✅ | ✅ | W(z) = 1+z·Z(z), stable branch |
//! | `tridiagonal_ql(d, e)` | ✅ | — | QL with Wilkinson shifts |
//! | `anderson_diagonalize(w, t)` | ✅ | — | Anderson tight-binding eigensolver |
//!
//! # Precision
//!
//! - CPU functions: f64, match scipy.special to machine precision
//! - GPU functions: f32, |ε| < 1e-5 for most inputs
//!
//! # Examples
//!
//! ```
//! use barracuda::special::{gamma, factorial, erf, bessel_j0};
//! use std::f64::consts::PI;
//!
//! # fn main() -> barracuda::error::Result<()> {
//! // Γ(n) = (n-1)! for integers
//! assert!((gamma(5.0)? - 24.0).abs() < 1e-12);
//!
//! // Γ(1/2) = √π
//! assert!((gamma(0.5)? - PI.sqrt()).abs() < 1e-12);
//!
//! // Error function
//! assert!((erf(0.0) - 0.0).abs() < 1e-14);
//! assert!((erf(1.0) - 0.8427007929).abs() < 2e-7);
//!
//! // Bessel J₀
//! assert!((bessel_j0(0.0) - 1.0).abs() < 1e-14);
//! # Ok(())
//! # }
//! ```
//!
//! # References
//!
//! - Abramowitz & Stegun (A&S): Handbook of Mathematical Functions
//! - DLMF: Digital Library of Mathematical Functions (<https://dlmf.nist.gov>)

// Core special functions (CPU f64)
pub mod anderson_transport;
pub mod bessel;
pub mod chi_squared;
pub mod erf;
pub mod factorial;
pub mod gamma;
pub mod hermite;
pub mod laguerre;
pub mod legendre;
#[cfg(feature = "gpu")]
pub mod plasma_dispersion;
#[cfg(feature = "gpu")]
pub mod screened_coulomb;
pub mod stable_gpu;
pub mod tridiagonal_ql;

// Re-export CPU functions
pub use anderson_transport::{anderson_conductance, localization_length};
pub use bessel::{bessel_i0, bessel_j0, bessel_j1, bessel_k0};
#[cfg(feature = "gpu")]
pub use chi_squared::{ChiSquaredBatchGpu, ChiSquaredBatchResult};
pub use chi_squared::{
    chi_squared_cdf, chi_squared_f64, chi_squared_mean, chi_squared_mode, chi_squared_pdf,
    chi_squared_quantile, chi_squared_sf, chi_squared_statistic, chi_squared_test,
    chi_squared_variance,
};
pub use erf::{erf, erfc};
pub use factorial::factorial;
pub use gamma::{
    beta, digamma, gamma, ln_beta, ln_gamma, lower_incomplete_gamma, regularized_gamma_p,
    regularized_gamma_q, upper_incomplete_gamma,
};
pub use hermite::hermite;
pub use laguerre::{laguerre, laguerre_all, laguerre_simple};
pub use legendre::{assoc_legendre, legendre};
#[cfg(feature = "gpu")]
pub use plasma_dispersion::{plasma_dispersion_w, plasma_dispersion_w_stable, plasma_dispersion_z};
#[cfg(feature = "gpu")]
pub use screened_coulomb::screened_coulomb_eigenvalues;
pub use stable_gpu::{bessel_j0_minus1_f64, erfc_f64, expm1_f64, log1p_f64};
pub use tridiagonal_ql::{anderson_diagonalize, tridiagonal_ql};

// Re-export GPU ops for batch processing (requires GPU feature)
#[cfg(feature = "gpu")]
pub use crate::ops::bessel_i0_wgsl::BesselI0 as BesselI0Gpu;
#[cfg(feature = "gpu")]
pub use crate::ops::bessel_j0_wgsl::BesselJ0 as BesselJ0Gpu;
#[cfg(feature = "gpu")]
pub use crate::ops::bessel_j1_wgsl::BesselJ1 as BesselJ1Gpu;
#[cfg(feature = "gpu")]
pub use crate::ops::bessel_k0_wgsl::BesselK0 as BesselK0Gpu;
#[cfg(feature = "gpu")]
pub use crate::ops::erf_wgsl::Erf as ErfGpu;
#[cfg(feature = "gpu")]
pub use crate::ops::erfc_wgsl::Erfc as ErfcGpu;
#[cfg(feature = "gpu")]
pub use crate::ops::lgamma_wgsl::Lgamma as LgammaGpu;
#[cfg(feature = "gpu")]
pub use crate::ops::spherical_harmonics_wgsl::SphericalHarmonics as SphericalHarmonicsGpu;
