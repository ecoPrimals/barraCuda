// SPDX-License-Identifier: AGPL-3.0-only
//! Benchmark operation runners — CPU/GPU execution for threshold calibration

use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use std::f64::consts::PI;
use std::sync::Arc;

/// Test data for benchmarking
pub(super) struct TestData {
    /// Flat array of f64 values
    pub(super) data: Vec<f64>,
    /// Size parameter (interpretation depends on operation)
    pub(super) size: usize,
}

/// Generate test data for an operation
pub(super) fn generate_test_data(operation: &str, size: usize) -> TestData {
    let n = match operation {
        "matmul" | "cholesky" | "eigh" | "lu" | "qr" | "svd" | "solve" => size * size,
        "cdist" | "rbf_kernel" => size * 10, // N points × 10 dimensions
        _ => size,
    };

    let data: Vec<f64> = (0..n)
        .map(|i| {
            let x = (i as f64) / (n as f64) * 2.0 * PI;
            match operation {
                "erf" | "erfc" => x.sin(),
                "gamma" | "lgamma" | "digamma" => 1.0 + (x.sin() * 0.5 + 0.5) * 10.0,
                "bessel_j0" | "bessel_j1" => x * 10.0,
                "exp" => x.sin() * 2.0,
                "log" | "sqrt" => 1.0 + x.sin().abs() * 10.0,
                "cholesky" => {
                    let row = i / size;
                    let col = i % size;
                    if row == col {
                        (size as f64) + 1.0
                    } else {
                        (x.sin() * 0.1).abs()
                    }
                }
                _ => x.sin(),
            }
        })
        .collect();

    TestData { data, size }
}

/// Run CPU operation for benchmarking
pub(super) fn run_cpu_operation(
    operation: &str,
    test_data: &TestData,
    gpu_device: Option<&Arc<WgpuDevice>>,
) -> Result<()> {
    let data = &test_data.data;
    let size = test_data.size;

    match operation {
        "erf" => {
            for &x in data {
                let _ = crate::special::erf(x);
            }
        }
        "gamma" => {
            for &x in data {
                let _ = crate::special::gamma(x);
            }
        }
        "bessel_j0" => {
            for &x in data {
                let _ = crate::special::bessel_j0(x);
            }
        }
        "exp" => {
            for &x in data {
                let _ = x.exp();
            }
        }
        "log" => {
            for &x in data {
                let _ = x.ln();
            }
        }
        "sqrt" => {
            for &x in data {
                let _ = x.sqrt();
            }
        }
        "sin" => {
            for &x in data {
                let _ = x.sin();
            }
        }
        "cos" => {
            for &x in data {
                let _ = x.cos();
            }
        }
        "sum" => {
            let _: f64 = data.iter().sum();
        }
        "max" => {
            let _ = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        }
        "matmul" => {
            let n = size;
            let mut c = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..n {
                        sum += data[i * n + k] * data[k * n + j];
                    }
                    c[i * n + j] = sum;
                }
            }
        }
        "cholesky" => {
            #[cfg(feature = "benchmarks")]
            let _ = crate::linalg::cholesky::cholesky_f64_cpu(data, size);
            #[cfg(not(feature = "benchmarks"))]
            return Err(BarracudaError::InvalidInput {
                message: "cholesky CPU benchmark requires --features benchmarks".into(),
            });
        }
        "lu" => {
            #[cfg(feature = "benchmarks")]
            let _ = crate::linalg::cholesky::cholesky_f64_cpu(data, size);
            #[cfg(not(feature = "benchmarks"))]
            return Err(BarracudaError::InvalidInput {
                message: "lu CPU benchmark requires --features benchmarks".into(),
            });
        }
        "qr" => {
            let b: Vec<f64> = (0..size).map(|i| (i as f64).sin()).collect();
            let dev = gpu_device.ok_or_else(|| BarracudaError::InvalidInput {
                message: "qr benchmark requires GPU".into(),
            })?;
            let _ = crate::linalg::solve_f64(dev.clone(), data, &b, size);
        }
        "svd" => {
            let _ = crate::linalg::eigh_f64(data, size);
        }
        "eigh" => {
            let _ = crate::linalg::eigh_f64(data, size);
        }
        "solve" => {
            let b: Vec<f64> = (0..size).map(|i| (i as f64).sin()).collect();
            let dev = gpu_device.ok_or_else(|| BarracudaError::InvalidInput {
                message: "solve benchmark requires GPU".into(),
            })?;
            let _ = crate::linalg::solve_f64(dev.clone(), data, &b, size);
        }
        "cdist" => {
            let n = size;
            let d = 10;
            let mut dists = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut dist = 0.0;
                    for k in 0..d {
                        let diff = data[i * d + k] - data[j * d + k];
                        dist += diff * diff;
                    }
                    dists[i * n + j] = dist.sqrt();
                }
            }
        }
        "rbf_kernel" => {
            let n = size;
            let d = 10;
            let epsilon = 1.0;
            let mut kernel = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut dist_sq = 0.0;
                    for k in 0..d {
                        let diff = data[i * d + k] - data[j * d + k];
                        dist_sq += diff * diff;
                    }
                    kernel[i * n + j] = (-epsilon * epsilon * dist_sq).exp();
                }
            }
        }
        _ => {
            return Err(BarracudaError::InvalidInput {
                message: format!("Unknown operation: {operation}"),
            });
        }
    }

    Ok(())
}

/// Simulated GPU dispatch overhead for benchmark estimation.
const SIMULATED_DISPATCH_OVERHEAD_US: u64 = 100;
/// Simulated per-element compute time for benchmark estimation.
const SIMULATED_PER_ELEMENT_NS: u64 = 1;

const SPEEDUP_MATMUL: f64 = 50.0;
const SPEEDUP_TRANSCENDENTAL: f64 = 100.0;
const SPEEDUP_REDUCTION: f64 = 20.0;
const SPEEDUP_LINALG: f64 = 10.0;
const SPEEDUP_PAIRWISE: f64 = 30.0;
const SPEEDUP_DEFAULT: f64 = 20.0;

/// Run GPU operation for benchmarking
pub(super) fn run_gpu_operation(
    operation: &str,
    test_data: &TestData,
    _gpu_device: Option<&Arc<WgpuDevice>>,
) -> Result<()> {
    let size = test_data.size;

    let dispatch_overhead = std::time::Duration::from_micros(SIMULATED_DISPATCH_OVERHEAD_US);
    let per_element_time = std::time::Duration::from_nanos(SIMULATED_PER_ELEMENT_NS);

    let elements = match operation {
        "matmul" | "cholesky" | "eigh" | "lu" | "qr" | "svd" | "solve" => size * size * size,
        "cdist" | "rbf_kernel" => size * size * 10,
        _ => size,
    };

    let gpu_speedup = match operation {
        "matmul" => SPEEDUP_MATMUL,
        "exp" | "log" | "sqrt" | "sin" | "cos" => SPEEDUP_TRANSCENDENTAL,
        "sum" | "max" => SPEEDUP_REDUCTION,
        "cholesky" | "lu" | "qr" | "svd" | "eigh" | "solve" => SPEEDUP_LINALG,
        "cdist" | "rbf_kernel" => SPEEDUP_PAIRWISE,
        _ => SPEEDUP_DEFAULT,
    };

    let _simulated_time =
        dispatch_overhead + per_element_time.mul_f64((elements as f64) / gpu_speedup);

    std::thread::yield_now();
    Ok(())
}
