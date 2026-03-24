// SPDX-License-Identifier: AGPL-3.0-or-later
//! CPU Thomas algorithm (O(n) sequential) — test/validation only.
//!
//! Production path uses GPU serial solver. This reference implementation
//! can be used for unit test validation.

/// Thomas algorithm for tridiagonal systems (O(n) sequential).
#[expect(
    dead_code,
    reason = "CPU reference implementation for GPU parity validation"
)]
pub fn solve_cpu_thomas(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut c_prime = vec![0.0f64; n];
    let mut d_prime = vec![0.0f64; n];

    // Forward sweep
    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for i in 1..n {
        let m = a[i].mul_add(-c_prime[i - 1], b[i]);
        c_prime[i] = c[i] / m;
        d_prime[i] = a[i].mul_add(-d_prime[i - 1], d[i]) / m;
    }

    // Back substitution
    let mut x = vec![0.0f64; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = c_prime[i].mul_add(-x[i + 1], d_prime[i]);
    }

    x
}
