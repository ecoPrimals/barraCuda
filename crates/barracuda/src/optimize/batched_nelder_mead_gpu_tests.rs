// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;

fn rosenbrock(x: &[f64]) -> f64 {
    let (a, b) = (1.0, 100.0);
    (a - x[0]).mul_add(a - x[0], b * x[0].mul_add(-x[0], x[1]).powi(2))
}

fn init_simplex(x0: &[f64], delta: f64) -> Vec<f64> {
    let n = x0.len();
    let mut out = Vec::with_capacity((n + 1) * n);
    out.extend_from_slice(x0);
    for i in 0..n {
        let mut x = x0.to_vec();
        x[i] += delta;
        out.extend_from_slice(&x);
    }
    out
}

#[tokio::test]
async fn test_batched_nelder_mead_rosenbrock() {
    let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
    else {
        return;
    };

    let config = BatchNelderMeadConfig {
        dims: 2,
        max_iters: 500,
        tol: 1e-6,
        ..Default::default()
    };

    let n_problems = 4;
    let mut initial_simplices = Vec::new();
    for p in 0..n_problems {
        let x0 = [(p as f64).mul_add(0.5, -1.0), -1.0];
        initial_simplices.extend_from_slice(&init_simplex(&x0, 0.1));
    }

    let results =
        batched_nelder_mead_gpu(&device, &config, n_problems, &initial_simplices, |pts| {
            pts.chunks(2).map(rosenbrock).collect()
        })
        .await
        .unwrap();

    assert_eq!(results.len(), n_problems);
    for (i, r) in results.iter().enumerate() {
        assert!(
            (r.best_point[0] - 1.0).abs() < 1e-2,
            "Problem {}: x[0]={}",
            i,
            r.best_point[0]
        );
        assert!(
            (r.best_point[1] - 1.0).abs() < 1e-2,
            "Problem {}: x[1]={}",
            i,
            r.best_point[1]
        );
        assert!(r.best_value < 1e-2, "Problem {}: f={}", i, r.best_value);
    }
}
