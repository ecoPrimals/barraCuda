// SPDX-License-Identifier: AGPL-3.0-or-later
//! Numerical Hessian computation via finite differences

/// Numerical Hessian via central finite differences.
///
/// H(i,j) = (f(x+ei+ej) - f(x+ei-ej) - f(x-ei+ej) + f(x-ei-ej)) / (4*epsilon^2)
#[must_use]
pub fn numerical_hessian(
    loss_fn: &dyn Fn(&[f64]) -> f64,
    params: &[f64],
    epsilon: f64,
) -> Vec<f64> {
    let n = params.len();
    let mut hessian = vec![0.0; n * n];
    let mut p = params.to_vec();
    for i in 0..n {
        for j in i..n {
            p.copy_from_slice(params);
            p[i] += epsilon;
            p[j] += epsilon;
            let fpp = loss_fn(&p);
            p.copy_from_slice(params);
            p[i] += epsilon;
            p[j] -= epsilon;
            let fpm = loss_fn(&p);
            p.copy_from_slice(params);
            p[i] -= epsilon;
            p[j] += epsilon;
            let fmp = loss_fn(&p);
            p.copy_from_slice(params);
            p[i] -= epsilon;
            p[j] -= epsilon;
            let fmm = loss_fn(&p);
            let hij = (fpp - fpm - fmp + fmm) / (4.0 * epsilon * epsilon);
            hessian[i * n + j] = hij;
            hessian[j * n + i] = hij;
        }
    }
    hessian
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hessian_quadratic() {
        // f(x) = sum(x_i^2) -> H = 2I
        let f = |x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>();
        let params = vec![1.0, 2.0, 3.0];
        let h = numerical_hessian(&f, &params, 1e-5);
        let n = 3;
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 2.0 } else { 0.0 };
                assert!(
                    (h[i * n + j] - expected).abs() < 1e-4,
                    "H[{},{}] = {}, expected {}",
                    i,
                    j,
                    h[i * n + j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_hessian_rosenbrock() {
        // Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        // At (1,1): minimum, Hessian positive definite
        let f = |x: &[f64]| {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        };
        let params = vec![1.0, 1.0];
        let h = numerical_hessian(&f, &params, 1e-5);
        // At (1,1): H = [[802, -400], [-400, 200]] (approx)
        // Both eigenvalues positive -> positive definite
        let det = h[0] * h[3] - h[1] * h[2];
        assert!(
            det > 0.0,
            "Hessian at minimum should have positive determinant"
        );
        assert!(h[0] > 0.0, "H[0,0] should be positive");
    }

    #[test]
    fn test_hessian_symmetric() {
        let f = |x: &[f64]| x[0] * x[0] * x[1] + x[1] * x[1]; // arbitrary smooth function
        let params = vec![1.0, 2.0];
        let h = numerical_hessian(&f, &params, 1e-5);
        let n = 2;
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (h[i * n + j] - h[j * n + i]).abs() < 1e-6,
                    "H[{},{}] = {} != H[{},{}] = {}",
                    i,
                    j,
                    h[i * n + j],
                    j,
                    i,
                    h[j * n + i]
                );
            }
        }
    }
}
