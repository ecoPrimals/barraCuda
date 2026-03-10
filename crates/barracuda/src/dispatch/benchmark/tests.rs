// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use std::time::Duration;

#[test]
fn test_benchmark_config() {
    let config = BenchmarkConfig::default();
    assert!(!config.sizes.is_empty());
    assert!(config.timed_iterations > 0);
}

#[test]
fn test_quick_config() {
    let config = BenchmarkConfig::quick();
    assert!(config.timed_iterations < BenchmarkConfig::default().timed_iterations);
}

#[test]
fn test_operation_benchmark_crossover() {
    let bench = OperationBenchmark {
        operation: "test".to_string(),
        sizes: vec![16, 32, 64, 128, 256],
        cpu_times: vec![
            Duration::from_micros(10),
            Duration::from_micros(20),
            Duration::from_micros(40),
            Duration::from_micros(80),
            Duration::from_micros(160),
        ],
        gpu_times: vec![
            Duration::from_micros(100), // GPU slower at small size
            Duration::from_micros(100),
            Duration::from_micros(50), // Crossover around here
            Duration::from_micros(30),
            Duration::from_micros(20),
        ],
        speedups: vec![0.1, 0.2, 0.8, 2.67, 8.0],
        gpu_available: true,
    };

    // Crossover where speedup >= 1.2
    let crossover = bench.crossover_size(1.2);
    assert_eq!(crossover, Some(128));

    // Optimal threshold with safety margin
    let threshold = bench.optimal_threshold(1.2, 1.5);
    assert!(threshold < 128); // Below crossover due to safety margin
}

#[test]
fn test_benchmark_suite_creation() {
    let suite = BenchmarkSuite::new(BenchmarkConfig::quick());
    // Just check it creates without panic
    let _ = suite.has_gpu();
}

#[test]
fn test_generate_test_data() {
    let data = generate_test_data("matmul", 32);
    assert_eq!(data.data.len(), 32 * 32);

    let data = generate_test_data("erf", 100);
    assert_eq!(data.data.len(), 100);
}

#[test]
fn test_run_cpu_operation() {
    let data = generate_test_data("erf", 100);
    assert!(run_cpu_operation("erf", &data, None).is_ok());

    let data = generate_test_data("sum", 100);
    assert!(run_cpu_operation("sum", &data, None).is_ok());
}
