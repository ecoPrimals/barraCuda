// SPDX-License-Identifier: AGPL-3.0-or-later
//! `SparsitySampler` tests.

use super::filter::compute_surrogate_rmse;
use super::sampler::sparsity_sampler;
use super::*;
use crate::device::test_pool::get_test_device_if_f64_gpu_available_sync;
use crate::surrogate::{RBFKernel, RBFSurrogate};

#[test]
fn test_sparsity_sampler_quadratic() {
    let Some(dev) = get_test_device_if_f64_gpu_available_sync() else {
        return;
    };
    let f = |x: &[f64]| (x[1] - 3.0).mul_add(x[1] - 3.0, (x[0] - 2.0).powi(2));
    let bounds = vec![(-10.0, 10.0), (-10.0, 10.0)];
    let config = SparsitySamplerConfig::new(2, 42)
        .with_initial_samples(20)
        .with_solvers(4)
        .with_eval_budget(30)
        .with_iterations(3);
    let result = sparsity_sampler(dev, f, &bounds, &config).unwrap();

    assert!((result.x_best[0] - 2.0).abs() < 2.0);
    assert!((result.x_best[1] - 3.0).abs() < 2.0);
    assert!(result.f_best < 5.0);
    assert!(result.cache.len() > 20);
    assert_eq!(result.iteration_results.len(), 3);
}

#[test]
fn test_sparsity_sampler_rosenbrock() {
    let Some(dev) = get_test_device_if_f64_gpu_available_sync() else {
        return;
    };
    let rosenbrock =
        |x: &[f64]| (1.0 - x[0]).mul_add(1.0 - x[0], 100.0 * x[0].mul_add(-x[0], x[1]).powi(2));
    let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

    let config = SparsitySamplerConfig::new(2, 42)
        .with_initial_samples(30)
        .with_solvers(8)
        .with_eval_budget(50)
        .with_iterations(5);

    let result = sparsity_sampler(dev, rosenbrock, &bounds, &config).unwrap();

    assert!(
        result.f_best < 50.0,
        "Should find reasonable Rosenbrock solution, got f={}",
        result.f_best
    );
    assert!(result.surrogate.is_some());
}

#[test]
fn test_sparsity_sampler_captures_all_evals() {
    let Some(dev) = get_test_device_if_f64_gpu_available_sync() else {
        return;
    };
    let f = |x: &[f64]| x[0].mul_add(x[0], x[1].powi(2));
    let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
    let config = SparsitySamplerConfig::new(2, 42)
        .with_initial_samples(10)
        .with_solvers(3)
        .with_iterations(2);
    let result = sparsity_sampler(dev, f, &bounds, &config).unwrap();

    assert!(
        result.cache.len() >= 10,
        "Should have at least initial samples, got {}",
        result.cache.len()
    );
    let (x_data, y_data) = result.cache.training_data();
    assert_eq!(x_data.len(), y_data.len());
}

#[test]
fn test_sparsity_sampler_iteration_diagnostics() {
    let Some(dev) = get_test_device_if_f64_gpu_available_sync() else {
        return;
    };
    let f = |x: &[f64]| x[0].powi(2);
    let bounds = vec![(-5.0, 5.0)];
    let config = SparsitySamplerConfig::new(1, 42)
        .with_initial_samples(10)
        .with_solvers(3)
        .with_eval_budget(20)
        .with_iterations(3);
    let result = sparsity_sampler(dev, f, &bounds, &config).unwrap();

    assert_eq!(result.iteration_results.len(), 3);

    for (i, ir) in result.iteration_results.iter().enumerate() {
        assert_eq!(ir.iteration, i);
        assert!(ir.n_new_evals > 0);
        assert!(ir.total_evals > 0);
    }

    for i in 1..result.iteration_results.len() {
        assert!(
            result.iteration_results[i].total_evals >= result.iteration_results[i - 1].total_evals
        );
    }
}

#[test]
fn test_sparsity_config_builder() {
    let config = SparsitySamplerConfig::new(3, 42)
        .with_initial_samples(50)
        .with_solvers(16)
        .with_eval_budget(100)
        .with_iterations(10)
        .with_kernel(RBFKernel::Gaussian { epsilon: 1.0 });

    assert_eq!(config.n_initial, 50);
    assert_eq!(config.n_solvers, 16);
    assert_eq!(config.max_eval_per_solver, 100);
    assert_eq!(config.n_iterations, 10);
    assert_eq!(config.seed, 42);
}

#[test]
fn test_sparsity_sampler_total_budget() {
    let config = SparsitySamplerConfig::new(2, 42)
        .with_initial_samples(20)
        .with_solvers(4)
        .with_eval_budget(50)
        .with_iterations(5);

    assert_eq!(config.total_budget(), 1020);
}

#[test]
fn test_sparsity_sampler_errors() {
    let Some(dev) = get_test_device_if_f64_gpu_available_sync() else {
        return;
    };
    let f = |x: &[f64]| x[0].powi(2);
    let config = SparsitySamplerConfig::new(1, 42);
    let empty_bounds: [(f64, f64); 0] = [];
    assert!(sparsity_sampler(dev.clone(), f, &empty_bounds, &config).is_err());

    let bounds = [(0.0, 1.0)];
    let config = SparsitySamplerConfig::new(1, 42).with_initial_samples(1);
    assert!(sparsity_sampler(dev, f, &bounds, &config).is_err());
}

#[test]
fn test_sparsity_sampler_1d() {
    let Some(dev) = get_test_device_if_f64_gpu_available_sync() else {
        return;
    };
    let f = |x: &[f64]| (x[0] - 3.0).mul_add(x[0] - 3.0, 1.0);
    let bounds = vec![(-10.0, 10.0)];
    let config = SparsitySamplerConfig::new(1, 42)
        .with_initial_samples(10)
        .with_solvers(4)
        .with_eval_budget(30)
        .with_iterations(3);
    let result = sparsity_sampler(dev, f, &bounds, &config).unwrap();

    assert!(
        (result.x_best[0] - 3.0).abs() < 2.0,
        "Should find x near 3.0, got {}",
        result.x_best[0]
    );
    assert!(result.f_best < 5.0);
}

#[test]
fn test_sparsity_sampler_with_gaussian_kernel() {
    let Some(dev) = get_test_device_if_f64_gpu_available_sync() else {
        return;
    };
    let f = |x: &[f64]| x[0].mul_add(x[0], x[1].powi(2));
    let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
    let config = SparsitySamplerConfig::new(2, 42)
        .with_initial_samples(15)
        .with_solvers(3)
        .with_iterations(2)
        .with_kernel(RBFKernel::Gaussian { epsilon: 0.5 });
    let result = sparsity_sampler(dev, f, &bounds, &config).unwrap();

    assert!(result.f_best < 10.0);
    assert!(result.surrogate.is_some());
}

#[test]
fn test_surrogate_rmse() {
    let Some(dev) = get_test_device_if_f64_gpu_available_sync() else {
        return;
    };
    let x_train = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
    let y_train = vec![0.0, 1.0, 4.0, 9.0];
    let surrogate =
        RBFSurrogate::train(dev, &x_train, &y_train, RBFKernel::ThinPlateSpline, 1e-12).unwrap();

    let rmse = compute_surrogate_rmse(&surrogate, &x_train, &y_train);
    assert!(
        rmse < 1e-6,
        "Surrogate should interpolate training data exactly, RMSE={rmse}"
    );
}
