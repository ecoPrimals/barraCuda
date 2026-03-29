// SPDX-License-Identifier: AGPL-3.0-or-later
//! CPU Thomas algorithm (O(n) sequential) — test/validation only.
//!
//! Production path uses GPU serial solver. This reference implementation
//! can be used for unit test validation.

/// Thomas algorithm for tridiagonal systems (O(n) sequential).
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

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    #[test]
    fn diagonal_system() {
        // 3x + 0 = 6  =>  x = 2
        // 0 + 4y = 8  =>  y = 2
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        let c = [0.0, 0.0];
        let d = [6.0, 8.0];
        let x = solve_cpu_thomas(&a, &b, &c, &d);
        assert!((x[0] - 2.0).abs() < TOL);
        assert!((x[1] - 2.0).abs() < TOL);
    }

    #[test]
    fn tridiagonal_3x3() {
        // [2 -1  0] [x0]   [1]
        // [-1 2 -1] [x1] = [0]
        // [0 -1  2] [x2]   [1]
        let a = [0.0, -1.0, -1.0];
        let b = [2.0, 2.0, 2.0];
        let c = [-1.0, -1.0, 0.0];
        let d = [1.0, 0.0, 1.0];
        let x = solve_cpu_thomas(&a, &b, &c, &d);
        // Exact solution: x0 = 1, x1 = 1, x2 = 1
        for (i, &xi) in x.iter().enumerate() {
            assert!((xi - 1.0).abs() < TOL, "x[{i}] = {xi}, expected 1.0");
        }
    }

    #[test]
    fn single_equation() {
        let x = solve_cpu_thomas(&[0.0], &[5.0], &[0.0], &[10.0]);
        assert!((x[0] - 2.0).abs() < TOL);
    }

    #[test]
    fn heat_equation_pattern() {
        let n = 10;
        let a: Vec<f64> = (0..n).map(|i| if i == 0 { 0.0 } else { -1.0 }).collect();
        let b = vec![2.0; n];
        let c: Vec<f64> = (0..n)
            .map(|i| if i == n - 1 { 0.0 } else { -1.0 })
            .collect();
        let d = vec![0.0; n];
        let x = solve_cpu_thomas(&a, &b, &c, &d);
        for xi in &x {
            assert!(xi.abs() < TOL, "homogeneous system should give zero");
        }
    }

    #[test]
    fn known_4x4_system() {
        // A = [1 2 0 0; 3 4 5 0; 0 6 7 8; 0 0 9 10]
        let a = [0.0, 3.0, 6.0, 9.0];
        let b = [1.0, 4.0, 7.0, 10.0];
        let c = [2.0, 5.0, 8.0, 0.0];
        let d = [5.0, 24.0, 49.0, 47.0];
        let x = solve_cpu_thomas(&a, &b, &c, &d);
        // Verify Ax = d by back-substitution check
        let r0 = 1.0f64.mul_add(x[0], 2.0 * x[1]);
        let r1 = 3.0f64.mul_add(x[0], 4.0f64.mul_add(x[1], 5.0 * x[2]));
        let r2 = 6.0f64.mul_add(x[1], 7.0f64.mul_add(x[2], 8.0 * x[3]));
        let r3 = 9.0f64.mul_add(x[2], 10.0 * x[3]);
        assert!((r0 - 5.0).abs() < TOL, "row 0: {r0}");
        assert!((r1 - 24.0).abs() < TOL, "row 1: {r1}");
        assert!((r2 - 49.0).abs() < TOL, "row 2: {r2}");
        assert!((r3 - 47.0).abs() < TOL, "row 3: {r3}");
    }
}
