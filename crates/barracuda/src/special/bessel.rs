// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bessel functions of the first and second kind
//!
//! Implements J₀, J₁, I₀, K₀ using rational polynomial approximations
//! from Abramowitz & Stegun.
//!
//! # Precision
//!
//! |ε| < 5e-8 for most functions and input ranges.
//!
//! # References
//!
//! - Abramowitz & Stegun, §9.4 (J), §9.8 (I, K)
//! - DLMF 10: <https://dlmf.nist.gov/10>

/// Bessel function of the first kind, order 0: J₀(x).
///
/// # Properties
///
/// - J₀(0) = 1
/// - J₀(x) oscillates and decays as x → ∞
///
/// # Precision
///
/// |ε| < 5e-8 (A&S 9.4.1-9.4.3)
///
/// # Examples
///
/// ```
/// use barracuda::special::bessel_j0;
///
/// assert!((bessel_j0(0.0) - 1.0).abs() < 1e-14);
/// // First zero of J₀ is at x ≈ 2.4048
/// assert!(bessel_j0(2.4048).abs() < 0.001);
/// ```
#[must_use]
pub fn bessel_j0(x: f64) -> f64 {
    // Special case: J₀(0) = 1 exactly
    if x == 0.0 {
        return 1.0;
    }

    let ax = x.abs();

    if ax < 8.0 {
        // A&S 9.4.1: polynomial approximation for |x| < 3
        let y = x * x;
        let ans1 = y.mul_add(
            y.mul_add(
                y.mul_add(
                    y.mul_add(y.mul_add(-184.905_245_6, 77_392.330_17), -11_214_424.18),
                    651_619_640.7,
                ),
                -13_362_590_354.0,
            ),
            57_568_490_574.0,
        );
        let ans2 = y.mul_add(
            y.mul_add(
                y.mul_add(
                    y.mul_add(y.mul_add(1.0, 267.853_271_2), 59_272.648_53),
                    9_494_680.718,
                ),
                1_029_532_985.0,
            ),
            57_568_490_411.0,
        );
        ans1 / ans2
    } else {
        // A&S 9.4.3: asymptotic expansion for |x| >= 8
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 0.785398164;
        let ans1 = 1.0
            + y * (-0.1098628627e-2
                + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        let ans2 = -0.1562499995e-1
            + y * (0.1430488765e-3
                + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934945152e-7)));
        (std::f64::consts::FRAC_2_PI / ax).sqrt() * xx.cos().mul_add(ans1, -(z * xx.sin() * ans2))
    }
}

/// Bessel function of the first kind, order 1: J₁(x).
///
/// # Properties
///
/// - J₁(0) = 0
/// - J₁(x) oscillates and decays as x → ∞
///
/// # Precision
///
/// |ε| < 5e-8 (A&S 9.4.4-9.4.6)
///
/// # Examples
///
/// ```
/// use barracuda::special::bessel_j1;
///
/// assert!(bessel_j1(0.0).abs() < 1e-14);
/// // J₁(1) ≈ 0.4400505857
/// assert!((bessel_j1(1.0) - 0.4400505857).abs() < 1e-7);
/// ```
#[must_use]
pub fn bessel_j1(x: f64) -> f64 {
    let ax = x.abs();

    if ax < 8.0 {
        let y = x * x;
        let ans1 = x * y.mul_add(
            y.mul_add(
                y.mul_add(
                    y.mul_add(y.mul_add(-30.16036606, 15704.48260), -2972611.439),
                    242396853.1,
                ),
                -7895059235.0,
            ),
            72362614232.0,
        );
        let ans2 = y.mul_add(
            y.mul_add(
                y.mul_add(
                    y.mul_add(y.mul_add(1.0, 376.9991397), 99447.43394),
                    18583304.74,
                ),
                2300535178.0,
            ),
            144725228442.0,
        );
        ans1 / ans2
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356194491;
        let ans1 = 1.0
            + y * (0.183105e-2
                + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
        let ans2 = 0.04687499995
            + y * (-0.2002690873e-3
                + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
        let result = (std::f64::consts::FRAC_2_PI / ax).sqrt()
            * xx.cos().mul_add(ans1, -(z * xx.sin() * ans2));
        if x < 0.0 { -result } else { result }
    }
}

/// Modified Bessel function of the first kind, order 0: I₀(x).
///
/// # Properties
///
/// - I₀(0) = 1
/// - I₀(x) grows exponentially as x → ∞
///
/// # Precision
///
/// |ε| < 2e-7 (A&S 9.8.1-9.8.2)
///
/// # Examples
///
/// ```
/// use barracuda::special::bessel_i0;
///
/// assert!((bessel_i0(0.0) - 1.0).abs() < 1e-14);
/// // I₀(1) ≈ 1.2660658778
/// assert!((bessel_i0(1.0) - 1.2660658778).abs() < 1e-7);
/// ```
#[must_use]
pub fn bessel_i0(x: f64) -> f64 {
    // Special case: I₀(0) = 1 exactly
    if x == 0.0 {
        return 1.0;
    }

    let ax = x.abs();

    if ax < 3.75 {
        let y = (x / 3.75).powi(2);
        1.0 + y
            * (3.515_622_9
                + y * (3.089_942_4
                    + y * (1.206_749_2 + y * (0.265_973_2 + y * (0.360_768e-1 + y * 0.458_13e-2)))))
    } else {
        let y = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.398_942_28
                + y * (0.132_859_2e-1
                    + y * (0.225319e-2
                        + y * (-0.157565e-2
                            + y * (0.916281e-2
                                + y * (-0.2057706e-1
                                    + y * (0.2635537e-1
                                        + y * (-0.1647633e-1 + y * 0.392377e-2))))))))
    }
}

/// Modified Bessel function of the second kind, order 0: K₀(x).
///
/// # Properties
///
/// - K₀(0) = ∞ (singular at origin)
/// - K₀(x) decays exponentially as x → ∞
///
/// # Precision
///
/// |ε| < 2e-7 (A&S 9.8.5-9.8.6)
///
/// # Examples
///
/// ```
/// use barracuda::special::bessel_k0;
///
/// // K₀(1) ≈ 0.4210244382
/// assert!((bessel_k0(1.0) - 0.4210244382).abs() < 1e-7);
/// // K₀(x) decays exponentially
/// assert!(bessel_k0(5.0) < 0.01);
/// ```
#[must_use]
pub fn bessel_k0(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }

    if x <= 2.0 {
        let y = x * x / 4.0;
        // NR formula: K₀(x) = -ln(x/2)·I₀(x) + polynomial
        (-(x / 2.0).ln()).mul_add(
            bessel_i0(x),
            -0.577_215_66
                + y * (0.422_784_20
                    + y * (0.230_697_56
                        + y * (0.348_859_0e-1
                            + y * (0.262_698e-2 + y * (0.107_50e-3 + y * 0.74e-5))))),
        )
    } else {
        let y = 2.0 / x;
        ((-x).exp() / x.sqrt())
            * (1.253_314_14
                + y * (-0.783_235_8e-1
                    + y * (0.2189568e-1
                        + y * (-0.1062446e-1
                            + y * (0.587872e-2 + y * (-0.251540e-2 + y * 0.53208e-3))))))
    }
}

/// Compute J₀ for a batch of values.
///
/// With `cpu-shader`, dispatches through `bessel_j0_f64.wgsl` via naga-exec.
/// The native Rust fallback is deprecated and will be removed in 0.5.0.
#[must_use]
pub fn bessel_j0_batch(x: &[f64]) -> Vec<f64> {
    #[cfg(feature = "cpu-shader")]
    {
        let wgsl = include_str!("../shaders/special/bessel_j0_f64.wgsl");
        if let Ok(out) = crate::unified_hardware::shader_batch_unary_f64_with_size(wgsl, "main", x)
        {
            return out;
        }
    }
    x.iter().map(|&v| bessel_j0(v)).collect()
}

/// Compute J₁ for a batch of values.
///
/// With `cpu-shader`, dispatches through `bessel_j1_f64.wgsl` via naga-exec.
/// The native Rust fallback is deprecated and will be removed in 0.5.0.
#[must_use]
pub fn bessel_j1_batch(x: &[f64]) -> Vec<f64> {
    #[cfg(feature = "cpu-shader")]
    {
        let wgsl = include_str!("../shaders/special/bessel_j1_f64.wgsl");
        if let Ok(out) = crate::unified_hardware::shader_batch_unary_f64_with_size(wgsl, "main", x)
        {
            return out;
        }
    }
    x.iter().map(|&v| bessel_j1(v)).collect()
}

/// Compute I₀ for a batch of values.
///
/// With `cpu-shader`, dispatches through `bessel_i0_f64.wgsl` via naga-exec.
/// The native Rust fallback is deprecated and will be removed in 0.5.0.
#[must_use]
pub fn bessel_i0_batch(x: &[f64]) -> Vec<f64> {
    #[cfg(feature = "cpu-shader")]
    {
        let wgsl = include_str!("../shaders/special/bessel_i0_f64.wgsl");
        if let Ok(out) = crate::unified_hardware::shader_batch_unary_f64_with_size(wgsl, "main", x)
        {
            return out;
        }
    }
    x.iter().map(|&v| bessel_i0(v)).collect()
}

/// Compute K₀ for a batch of values.
///
/// With `cpu-shader`, dispatches through `bessel_k0_f64.wgsl` via naga-exec.
/// The native Rust fallback is deprecated and will be removed in 0.5.0.
#[must_use]
pub fn bessel_k0_batch(x: &[f64]) -> Vec<f64> {
    #[cfg(feature = "cpu-shader")]
    {
        let wgsl = include_str!("../shaders/special/bessel_k0_f64.wgsl");
        if let Ok(out) = crate::unified_hardware::shader_batch_unary_f64_with_size(wgsl, "main", x)
        {
            return out;
        }
    }
    x.iter().map(|&v| bessel_k0(v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // J₀ tests
    #[test]
    fn test_bessel_j0_zero() {
        assert!((bessel_j0(0.0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_bessel_j0_small() {
        // scipy.special.j0(1.0) = 0.7651976865579666
        assert!((bessel_j0(1.0) - 0.7651976865579666).abs() < 1e-7);
    }

    #[test]
    fn test_bessel_j0_first_zero() {
        // First zero of J₀ is at x ≈ 2.4048255577
        assert!(bessel_j0(2.4048255577).abs() < 1e-6);
    }

    #[test]
    fn test_bessel_j0_large() {
        // scipy.special.j0(10.0) = -0.2459357644513483
        assert!((bessel_j0(10.0) - (-0.2459357644513483)).abs() < 1e-6);
    }

    #[test]
    fn test_bessel_j0_even() {
        // J₀ is an even function
        assert!((bessel_j0(-2.0) - bessel_j0(2.0)).abs() < 1e-14);
    }

    // J₁ tests
    #[test]
    fn test_bessel_j1_zero() {
        assert!(bessel_j1(0.0).abs() < 1e-14);
    }

    #[test]
    fn test_bessel_j1_one() {
        // scipy.special.j1(1.0) = 0.44005058574493355
        assert!((bessel_j1(1.0) - 0.44005058574493355).abs() < 1e-7);
    }

    #[test]
    fn test_bessel_j1_odd() {
        // J₁ is an odd function
        assert!((bessel_j1(-2.0) + bessel_j1(2.0)).abs() < 1e-14);
    }

    // I₀ tests
    #[test]
    fn test_bessel_i0_zero() {
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_bessel_i0_one() {
        // scipy.special.i0(1.0) = 1.2660658777520082
        assert!((bessel_i0(1.0) - 1.2660658777520082).abs() < 1e-7);
    }

    #[test]
    fn test_bessel_i0_even() {
        // I₀ is an even function
        assert!((bessel_i0(-2.0) - bessel_i0(2.0)).abs() < 1e-14);
    }

    // K₀ tests
    #[test]
    fn test_bessel_k0_one() {
        // scipy.special.k0(1.0) = 0.42102443824070823
        // A&S 9.8.5-9.8.6 precision: |ε| < 2e-7
        assert!((bessel_k0(1.0) - 0.42102443824070823).abs() < 2e-6);
    }

    #[test]
    fn test_bessel_k0_singular() {
        assert!(bessel_k0(0.0).is_infinite());
        assert!(bessel_k0(-1.0).is_infinite());
    }

    #[test]
    fn test_bessel_k0_decay() {
        // K₀ decays exponentially
        assert!(bessel_k0(5.0) < 0.01);
        assert!(bessel_k0(10.0) < 1e-4);
    }

    // Batch tests
    #[test]
    fn test_batch_functions() {
        let x = vec![0.0, 1.0, 2.0];

        let j0 = bessel_j0_batch(&x);
        assert_eq!(j0.len(), 3);
        assert!((j0[0] - 1.0).abs() < 1e-14);

        let j1 = bessel_j1_batch(&x);
        assert!(j1[0].abs() < 1e-14);

        let i0 = bessel_i0_batch(&x);
        assert!((i0[0] - 1.0).abs() < 1e-14);
    }
}
