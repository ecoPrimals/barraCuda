// SPDX-License-Identifier: AGPL-3.0-only
//! Vector Mixing Operations for Self-Consistent Field Solvers
//!
//! **Physics-Agnostic Mixing Primitives**
//!
//! Generic vector mixing operations used in iterative solvers:
//! - Linear mixing: simple damped iteration
//! - Broyden mixing: quasi-Newton acceleration with history
//!
//! ## Applications
//!
//! - **Nuclear physics**: HFB density mixing (validated by hotSpring)
//! - **DFT**: Electron density mixing in Kohn-Sham iterations
//! - **Chemistry**: Coupled-cluster amplitude mixing
//! - **Electrostatics**: Poisson-Boltzmann SCF convergence
//! - **General**: Any fixed-point iteration F(x) = x
//!
//! ## Algorithm
//!
//! ### Linear Mixing
//! ```text
//! x_{n+1} = (1-α)·x_n + α·F(x_n)
//! ```
//! Safe but slow convergence. Use for warmup (first 3-5 iterations).
//!
//! ### Modified Broyden II
//! ```text
//! x_{n+1} = x_n + α·r_n - Σ_m γ_m·(Δx_m + α·Δr_m)
//! ```
//! Where `r_n` = `F(x_n)` - `x_n` is the residual.
//! Fast convergence (quadratic near solution) but requires history management.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::ops::mixing::{LinearMixer, BroydenMixer};
//!
//! // For warmup or simple problems
//! let linear = LinearMixer::new(device, vec_dim, 0.5)?;
//! let x_mixed = linear.mix(&x_old, &x_computed).await?;
//!
//! // For production SCF
//! let broyden = BroydenMixer::new(device, vec_dim, 8)?; // 8 history vectors
//! let x_mixed = broyden.update(&x, &residual).await?;
//! ```
//!
//! ## Deep Debt Compliance
//!
//! - Pure WGSL (f64 via `SHADER_F64`)
//! - Physics-agnostic (no domain-specific parameters)
//! - Validated by hotSpring nuclear EOS (169/169 acceptance checks)

mod broyden_f64;

pub use broyden_f64::{BroydenMixer, LinearMixer, MixingParams};

/// Recommended mixing parameters for different problem types
pub mod presets {
    use super::MixingParams;

    /// Conservative linear mixing for initial iterations
    #[must_use]
    pub fn warmup_linear() -> MixingParams {
        MixingParams {
            alpha: 0.3,
            clamp_min: None,
            clamp_max: None,
            n_warmup: 5,
        }
    }

    /// Standard Broyden for general SCF problems
    #[must_use]
    pub fn standard_broyden() -> MixingParams {
        MixingParams {
            alpha: 0.4,
            clamp_min: None,
            clamp_max: None,
            n_warmup: 3,
        }
    }

    /// For density mixing (must be non-negative)
    #[must_use]
    pub fn density_mixing() -> MixingParams {
        MixingParams {
            alpha: 0.5,
            clamp_min: Some(0.0),
            clamp_max: None,
            n_warmup: 3,
        }
    }

    /// Aggressive mixing for well-conditioned problems
    #[must_use]
    pub fn aggressive() -> MixingParams {
        MixingParams {
            alpha: 0.7,
            clamp_min: None,
            clamp_max: None,
            n_warmup: 2,
        }
    }
}
