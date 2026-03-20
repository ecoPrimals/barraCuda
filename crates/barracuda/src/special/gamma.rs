// SPDX-License-Identifier: AGPL-3.0-or-later
//! Gamma and Incomplete Gamma Functions — CPU reference implementations.
//!
//! **Shader-first architecture**: These functions have f64 WGSL equivalents in
//! `shaders/math/math_f64_special.wgsl` (`gamma_f64`, `lgamma_f64`, `digamma_f64`,
//! `regularized_gamma_f64`, `incomplete_gamma_f64`, `beta_f64`, `ln_beta_f64`).
//! GPU pipelines that need these operations should use the shader versions
//! via `compile_shader_f64()`. These CPU functions serve as:
//! - Reference implementations for shader validation
//! - Building blocks for CPU-side test/validation code
//! - Scalar fallbacks where GPU dispatch overhead exceeds compute cost
//!
//! # Functions
//!
//! - `gamma(x)` — Γ(x) → shader: `gamma_f64()`
//! - `ln_gamma(x)` — ln(Γ(x)) → shader: `lgamma_f64()`
//! - `lower_incomplete_gamma(a, x)` — γ(a,x)
//! - `upper_incomplete_gamma(a, x)` — Γ(a,x)
//! - `regularized_gamma_p(a, x)` — P(a,x) → shader: `regularized_gamma_f64()`
//! - `regularized_gamma_q(a, x)` — Q(a,x) = 1 - P(a,x)
//! - `digamma(x)` — ψ(x) → shader: `digamma_f64()`
//! - `beta(a, b)` — B(a,b) → shader: `beta_f64()`
//! - `ln_beta(a, b)` → shader: `ln_beta_f64()`
//!
//! # References
//!
//! - Numerical Recipes, 3rd Edition, Chapter 6
//! - Abramowitz & Stegun, Chapter 6
//! - NIST Digital Library of Mathematical Functions, Chapter 8

use crate::error::{BarracudaError, Result};

/// Lanczos coefficients for gamma function (g=7)
///
/// These are high-precision constants from Numerical Recipes - intentionally exact.
const LANCZOS_G: f64 = 7.0;
#[expect(clippy::excessive_precision, reason = "Numerical Recipes constants")]
const LANCZOS_COEFFS: [f64; 9] = [
    0.999_999_999_999_809_93,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_13,
    -176.615_029_162_140_59,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_571_6e-6,
    1.505_632_735_149_311_6e-7,
];

/// Natural logarithm of the gamma function: ln(Γ(x))
///
/// Uses the Lanczos approximation for high accuracy.
///
/// # Arguments
///
/// * `x` - Input value (x > 0)
///
/// # Returns
///
/// ln(Γ(x))
///
/// # Errors
///
/// Returns [`Err`] if `x <= 0` (invalid domain for ln(Γ)), or if the internal
/// series/continued-fraction convergence fails when using the reflection formula.
///
/// # Example
///
/// ```
/// use barracuda::special::ln_gamma;
///
/// // ln(Γ(1)) = 0
/// assert!((ln_gamma(1.0).unwrap() - 0.0).abs() < 1e-10);
///
/// // ln(Γ(5)) = ln(4!) = ln(24)
/// assert!((ln_gamma(5.0).unwrap() - 24.0_f64.ln()).abs() < 1e-10);
/// ```
pub fn ln_gamma(x: f64) -> Result<f64> {
    if x <= 0.0 {
        return Err(BarracudaError::InvalidInput {
            message: format!("ln_gamma requires x > 0, got {x}"),
        });
    }

    if x < 0.5 {
        // Use reflection formula: Γ(z)Γ(1-z) = π/sin(πz)
        let pi = std::f64::consts::PI;
        Ok(pi.ln() - (pi * x).sin().ln() - ln_gamma(1.0 - x)?)
    } else {
        let x = x - 1.0;
        let mut ag = LANCZOS_COEFFS[0];
        for (i, &c) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
            ag += c / (x + i as f64);
        }

        let tmp = x + LANCZOS_G + 0.5;
        Ok((2.0 * std::f64::consts::PI).sqrt().ln() + (x + 0.5) * tmp.ln() - tmp + ag.ln())
    }
}

/// Complete gamma function: Γ(x)
///
/// # Arguments
///
/// * `x` - Input value (x > 0 or negative non-integer)
///
/// # Errors
///
/// Returns [`Err`] if `x` is a non-positive integer (gamma is undefined), or if
/// the underlying [`ln_gamma`] call fails.
///
/// # Example
///
/// ```
/// use barracuda::special::gamma;
///
/// // Γ(5) = 4! = 24
/// assert!((gamma(5.0).unwrap() - 24.0).abs() < 1e-10);
///
/// // Γ(0.5) = √π
/// assert!((gamma(0.5).unwrap() - std::f64::consts::PI.sqrt()).abs() < 1e-10);
/// ```
pub fn gamma(x: f64) -> Result<f64> {
    if x <= 0.0 && x.fract() == 0.0 {
        return Err(BarracudaError::InvalidInput {
            message: format!("gamma is undefined for non-positive integers, got {x}"),
        });
    }

    Ok(ln_gamma(x)?.exp())
}

/// WGSL kernel for incomplete gamma function evaluation (f64).
pub const WGSL_INCOMPLETE_GAMMA_F64: &str =
    include_str!("../shaders/special/incomplete_gamma_f64.wgsl");

/// Lower incomplete gamma function: γ(a, x) = ∫₀ˣ t^(a-1) e^(-t) dt
///
/// Uses series expansion for x < a+1 and continued fraction for x >= a+1.
///
/// # Arguments
///
/// * `a` - Shape parameter (a > 0)
/// * `x` - Upper limit of integration (x >= 0)
///
/// # Errors
///
/// Returns [`Err`] if `a <= 0`, `x < 0`, or if the internal series/continued-fraction
/// convergence fails.
///
/// # Example
///
/// ```
/// use barracuda::special::lower_incomplete_gamma;
///
/// let (gamma_val, _) = lower_incomplete_gamma(2.0, 1.0)?;
/// // γ(2, 1) ≈ 0.2642
/// assert!((gamma_val - 0.2642).abs() < 0.01);
/// # Ok::<(), barracuda::error::BarracudaError>(())
/// ```
pub fn lower_incomplete_gamma(a: f64, x: f64) -> Result<(f64, f64)> {
    if a <= 0.0 {
        return Err(BarracudaError::InvalidInput {
            message: format!("lower_incomplete_gamma requires a > 0, got {a}"),
        });
    }
    if x < 0.0 {
        return Err(BarracudaError::InvalidInput {
            message: format!("lower_incomplete_gamma requires x >= 0, got {x}"),
        });
    }

    if x == 0.0 {
        return Ok((0.0, gamma(a)?));
    }

    let gln = ln_gamma(a)?;
    let gamma_complete = gln.exp();

    if x < a + 1.0 {
        // Series expansion
        let p = gamma_series(a, x, gln)?;
        Ok((p * gamma_complete, gamma_complete))
    } else {
        // Continued fraction
        let q = gamma_cf(a, x, gln)?;
        Ok(((1.0 - q) * gamma_complete, gamma_complete))
    }
}

/// Upper incomplete gamma function: Γ(a, x) = ∫ₓ^∞ t^(a-1) e^(-t) dt
///
/// Γ(a, x) = Γ(a) - γ(a, x)
///
/// # Arguments
///
/// * `a` - Shape parameter (a > 0)
/// * `x` - Lower limit of integration (x >= 0)
///
/// # Errors
///
/// Returns [`Err`] if [`lower_incomplete_gamma`] fails (invalid `a`/`x` or convergence failure).
pub fn upper_incomplete_gamma(a: f64, x: f64) -> Result<f64> {
    let (lower, complete) = lower_incomplete_gamma(a, x)?;
    Ok(complete - lower)
}

/// WGSL kernel for regularized gamma function P(a, x) = γ(a,x)/Γ(a) (f64).
pub const WGSL_REGULARIZED_GAMMA_F64: &str =
    include_str!("../shaders/special/regularized_gamma_f64.wgsl");

/// Regularized lower incomplete gamma function: P(a, x) = γ(a, x) / Γ(a)
///
/// This is the CDF of the gamma distribution.
///
/// # Arguments
///
/// * `a` - Shape parameter (a > 0)
/// * `x` - Upper limit of integration (x >= 0)
///
/// # Returns
///
/// P(a, x) in [0, 1]
///
/// # Errors
///
/// Returns [`Err`] if `a <= 0`, `x < 0`, or if the internal series/continued-fraction
/// convergence fails.
///
/// # Example
///
/// ```
/// use barracuda::special::regularized_gamma_p;
///
/// // P(1, 1) = 1 - e^(-1) ≈ 0.6321
/// let p = regularized_gamma_p(1.0, 1.0)?;
/// assert!((p - 0.6321).abs() < 0.01);
/// # Ok::<(), barracuda::error::BarracudaError>(())
/// ```
pub fn regularized_gamma_p(a: f64, x: f64) -> Result<f64> {
    if a <= 0.0 {
        return Err(BarracudaError::InvalidInput {
            message: format!("regularized_gamma_p requires a > 0, got {a}"),
        });
    }
    if x < 0.0 {
        return Err(BarracudaError::InvalidInput {
            message: format!("regularized_gamma_p requires x >= 0, got {x}"),
        });
    }

    if x == 0.0 {
        return Ok(0.0);
    }

    let gln = ln_gamma(a)?;

    if x < a + 1.0 {
        gamma_series(a, x, gln)
    } else {
        Ok(1.0 - gamma_cf(a, x, gln)?)
    }
}

/// Regularized upper incomplete gamma function: Q(a, x) = Γ(a, x) / Γ(a)
///
/// Q(a, x) = 1 - P(a, x)
///
/// # Arguments
///
/// * `a` - Shape parameter (a > 0)
/// * `x` - Lower limit of integration (x >= 0)
///
/// # Errors
///
/// Returns [`Err`] if [`regularized_gamma_p`] fails (invalid `a`/`x` or convergence failure).
pub fn regularized_gamma_q(a: f64, x: f64) -> Result<f64> {
    Ok(1.0 - regularized_gamma_p(a, x)?)
}

/// Series expansion for regularized incomplete gamma P(a, x)
fn gamma_series(a: f64, x: f64, gln: f64) -> Result<f64> {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 1e-14;

    let mut sum = 1.0 / a;
    let mut term = sum;

    for n in 1..MAX_ITER {
        term *= x / (a + n as f64);
        sum += term;

        if term.abs() < sum.abs() * EPS {
            return Ok(sum * (-x + a * x.ln() - gln).exp());
        }
    }

    Err(BarracudaError::ExecutionError {
        message: "gamma_series: convergence failed".to_string(),
    })
}

/// Continued fraction for regularized incomplete gamma Q(a, x)
fn gamma_cf(a: f64, x: f64, gln: f64) -> Result<f64> {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 1e-14;
    const FPMIN: f64 = 1e-30;

    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;

    for n in 1..MAX_ITER {
        let an = -(n as f64) * (n as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;

        if (delta - 1.0).abs() < EPS {
            return Ok((-x + a * x.ln() - gln).exp() * h);
        }
    }

    Err(BarracudaError::ExecutionError {
        message: "gamma_cf: convergence failed".to_string(),
    })
}

/// Digamma function: ψ(x) = d/dx ln(Γ(x)) = Γ'(x) / Γ(x)
///
/// The logarithmic derivative of the gamma function, commonly used in
/// statistics (e.g., expectation of log-gamma random variables) and
/// physics (e.g., harmonic oscillator energy levels).
///
/// # Arguments
///
/// * `x` - Input value (x > 0)
///
/// # Returns
///
/// ψ(x)
///
/// # Errors
///
/// Returns [`Err`] if `x <= 0` (invalid domain for digamma).
///
/// # Algorithm
///
/// Uses recurrence relation ψ(x+1) = ψ(x) + 1/x to reduce to x ≥ 7,
/// then applies asymptotic expansion:
///
/// ψ(x) ≈ ln(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴) - 1/(252x⁶) + ...
///
/// # Example
///
/// ```
/// use barracuda::special::digamma;
///
/// // ψ(1) = -γ (Euler-Mascheroni constant)
/// let psi_1 = digamma(1.0).unwrap();
/// assert!((psi_1 - (-0.5772156649015329)).abs() < 1e-9);
///
/// // Recurrence: ψ(2) = ψ(1) + 1
/// let psi_2 = digamma(2.0).unwrap();
/// assert!((psi_2 - (psi_1 + 1.0)).abs() < 1e-9);
/// ```
///
/// # Reference
///
/// - Abramowitz & Stegun, 6.3.18
/// - hotSpring Phase 5: inline implementation in `validate_special_functions.rs`
pub fn digamma(x: f64) -> Result<f64> {
    if x <= 0.0 {
        return Err(BarracudaError::InvalidInput {
            message: format!("digamma requires x > 0, got {x}"),
        });
    }

    // Recurrence to shift x >= 7
    let mut val = 0.0;
    let mut xx = x;
    while xx < 7.0 {
        val -= 1.0 / xx;
        xx += 1.0;
    }

    // Asymptotic expansion for large x
    let inv_x = 1.0 / xx;
    let inv_x2 = inv_x * inv_x;

    // ψ(x) ≈ ln(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴) - 1/(252x⁶)
    val += xx.ln() - 0.5 * inv_x - inv_x2 * (1.0 / 12.0 - inv_x2 * (1.0 / 120.0 - inv_x2 / 252.0));

    Ok(val)
}

/// Beta function: B(a, b) = Γ(a)Γ(b) / Γ(a+b)
///
/// The beta function is fundamental in probability (beta distribution) and
/// statistical inference (posterior distributions, Bayesian analysis).
///
/// # Arguments
///
/// * `a` - First parameter (a > 0)
/// * `b` - Second parameter (b > 0)
///
/// # Returns
///
/// B(a, b)
///
/// # Properties
///
/// - Symmetric: B(a, b) = B(b, a)
/// - B(1, 1) = 1
/// - B(a, 1) = 1/a
/// - B(1/2, 1/2) = π
///
/// # Errors
///
/// Returns [`Err`] if `a <= 0` or `b <= 0`, or if the underlying [`ln_gamma`] call fails.
///
/// # Example
///
/// ```
/// use barracuda::special::beta;
/// use std::f64::consts::PI;
///
/// // B(1, 1) = 1
/// let b_11 = beta(1.0, 1.0).unwrap();
/// assert!((b_11 - 1.0).abs() < 1e-10);
///
/// // B(1/2, 1/2) = π
/// let b_half = beta(0.5, 0.5).unwrap();
/// assert!((b_half - PI).abs() < 1e-10);
///
/// // B(a, 1) = 1/a
/// let b_a1 = beta(3.0, 1.0).unwrap();
/// assert!((b_a1 - 1.0/3.0).abs() < 1e-10);
/// ```
///
/// # Reference
///
/// - Abramowitz & Stegun, 6.2
/// - hotSpring Phase 5: inline implementation in `validate_special_functions.rs`
pub fn beta(a: f64, b: f64) -> Result<f64> {
    if a <= 0.0 || b <= 0.0 {
        return Err(BarracudaError::InvalidInput {
            message: format!("beta requires a > 0 and b > 0, got a={a}, b={b}"),
        });
    }

    // B(a, b) = exp(ln(Γ(a)) + ln(Γ(b)) - ln(Γ(a+b)))
    let ln_beta = ln_gamma(a)? + ln_gamma(b)? - ln_gamma(a + b)?;
    Ok(ln_beta.exp())
}

/// WGSL kernel for natural log of beta function ln(B(a,b)) (f64).
pub const WGSL_LN_BETA_F64: &str = include_str!("../shaders/special/ln_beta_f64.wgsl");

/// Natural logarithm of the beta function: ln(B(a, b))
///
/// Useful for avoiding overflow when a, b are large.
///
/// # Errors
///
/// Returns [`Err`] if `a <= 0` or `b <= 0`, or if the underlying [`ln_gamma`] call fails.
///
/// # Example
///
/// ```
/// use barracuda::special::ln_beta;
///
/// // ln(B(1, 1)) = ln(1) = 0
/// let lb_11 = ln_beta(1.0, 1.0).unwrap();
/// assert!(lb_11.abs() < 1e-10);
///
/// // For large values, avoids overflow
/// let lb_large = ln_beta(100.0, 100.0).unwrap();
/// assert!(lb_large.is_finite());
/// ```
pub fn ln_beta(a: f64, b: f64) -> Result<f64> {
    if a <= 0.0 || b <= 0.0 {
        return Err(BarracudaError::InvalidInput {
            message: format!("ln_beta requires a > 0 and b > 0, got a={a}, b={b}"),
        });
    }

    Ok(ln_gamma(a)? + ln_gamma(b)? - ln_gamma(a + b)?)
}

#[cfg(test)]
#[path = "gamma_tests.rs"]
mod tests;
