//! CPU Thomas algorithm (O(n) sequential) — test/validation only.
//!
//! Production path uses GPU serial solver. This reference implementation
//! can be used for unit test validation.

/// Thomas algorithm for tridiagonal systems (O(n) sequential).
#[cfg(test)]
#[allow(dead_code)]
pub fn solve_cpu_thomas(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut c_prime = vec![0.0f64; n];
    let mut d_prime = vec![0.0f64; n];

    // Forward sweep
    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for i in 1..n {
        let m = b[i] - a[i] * c_prime[i - 1];
        c_prime[i] = c[i] / m;
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / m;
    }

    // Back substitution
    let mut x = vec![0.0f64; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    x
}
