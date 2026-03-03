use super::*;
use std::sync::Arc;

fn get_test_device() -> Option<Arc<crate::device::WgpuDevice>> {
    crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
}

#[test]
fn test_simple_system() {
    let Some(device) = get_test_device() else {
        return;
    };
    let solver = CyclicReductionF64::new(device).unwrap();

    // 4x₀ + x₁ = 5
    // x₀ + 4x₁ + x₂ = 6
    // x₁ + 4x₂ = 5
    // Solution: x = [1, 1, 1]
    let a = vec![0.0, 1.0, 1.0];
    let b = vec![4.0, 4.0, 4.0];
    let c = vec![1.0, 1.0, 0.0];
    let d = vec![5.0, 6.0, 5.0];

    let x = solver.solve(&a, &b, &c, &d).unwrap();

    for (i, xi) in x.iter().enumerate() {
        assert!((*xi - 1.0).abs() < 1e-10, "x[{}] = {}, expected 1.0", i, xi);
    }
}

#[test]
fn test_heat_equation_stencil() {
    // Discretized 1D heat equation: -u'' = f with u(0)=u(1)=0
    // Standard 3-point stencil: [-1, 2, -1]
    let Some(device) = get_test_device() else {
        return;
    };
    let solver = CyclicReductionF64::new(device).unwrap();

    let n = 100;
    let h = 1.0 / (n + 1) as f64;

    let mut a = vec![0.0f64; n];
    let b = vec![2.0f64; n];
    let mut c = vec![0.0f64; n];
    let mut d = vec![0.0f64; n];

    for i in 1..n {
        a[i] = -1.0;
    }
    for i in 0..n - 1 {
        c[i] = -1.0;
    }

    // f(x) = sin(πx), solution: u(x) = sin(πx)/π²
    for i in 0..n {
        let x = (i + 1) as f64 * h;
        d[i] = h * h * (std::f64::consts::PI * x).sin();
    }

    let x = solver.solve(&a, &b, &c, &d).unwrap();

    // Verify against analytical solution
    for (i, xi) in x.iter().enumerate() {
        let x_pos = (i + 1) as f64 * h;
        let expected =
            (std::f64::consts::PI * x_pos).sin() / (std::f64::consts::PI * std::f64::consts::PI);
        let error = (*xi - expected).abs();
        assert!(
            error < 0.01, // Allow for discretization error
            "x[{}] = {}, expected {}, error {}",
            i,
            xi,
            expected,
            error
        );
    }
}

#[test]
fn test_large_system() {
    let Some(device) = get_test_device() else {
        return;
    };
    let solver = CyclicReductionF64::new(device).unwrap();

    // Large system to exercise GPU path
    let n = 1000;
    let a: Vec<f64> = (0..n).map(|i| if i > 0 { -1.0 } else { 0.0 }).collect();
    let b: Vec<f64> = vec![2.5; n];
    let c: Vec<f64> = (0..n).map(|i| if i < n - 1 { -1.0 } else { 0.0 }).collect();
    let d: Vec<f64> = vec![1.0; n];

    let x = solver.solve(&a, &b, &c, &d).unwrap();

    // Verify Ax = d
    for i in 0..n {
        let mut ax_i = b[i] * x[i];
        if i > 0 {
            ax_i += a[i] * x[i - 1];
        }
        if i < n - 1 {
            ax_i += c[i] * x[i + 1];
        }
        let residual = (ax_i - d[i]).abs();
        assert!(
            residual < 1e-8,
            "Residual at i={} is {}, expected < 1e-8",
            i,
            residual
        );
    }
}
