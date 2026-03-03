//! Benchmark Harness - Execute and manage benchmarks

use super::{BenchmarkConfig, BenchmarkResult};
use crate::error::Result;
use std::time::Instant;

/// Benchmark harness
pub struct Harness {
    config: BenchmarkConfig,
}

impl Harness {
    /// Create new harness
    pub fn new(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Run a benchmark function multiple times and collect statistics
    pub async fn run<F, Fut>(&self, name: &str, mut benchmark_fn: F) -> Result<BenchmarkResult>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<()>>,
    {
        println!("Running: {name}");

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            benchmark_fn().await?;
        }

        // Measurement
        let mut times = Vec::new();
        for _ in 0..self.config.measurement_iterations {
            let start = Instant::now();
            benchmark_fn().await?;
            times.push(start.elapsed());
        }

        // Compute statistics
        let n = times.len() as f64;
        let mean_ns = times.iter().map(|t| t.as_nanos() as f64).sum::<f64>() / n;
        let variance = times
            .iter()
            .map(|t| {
                let d = t.as_nanos() as f64 - mean_ns;
                d * d
            })
            .sum::<f64>()
            / n;
        let std_ns = variance.sqrt();

        let mut sorted = times.clone();
        sorted.sort();
        let median = sorted[sorted.len() / 2];
        let min_time = *sorted.first().unwrap_or(&std::time::Duration::ZERO);
        let max_time = *sorted.last().unwrap_or(&std::time::Duration::ZERO);
        let mean_time = std::time::Duration::from_nanos(mean_ns as u64);
        let std_dev = std::time::Duration::from_nanos(std_ns as u64);

        Ok(BenchmarkResult {
            operation: name.to_string(),
            hardware: String::new(),
            framework: super::Framework::BarraCuda,
            median_time: median,
            mean_time,
            std_dev,
            min_time,
            max_time,
            throughput: 1_000_000_000.0 / mean_ns,
            bandwidth_gbps: 0.0,
            tflops: 0.0,
        })
    }
}
