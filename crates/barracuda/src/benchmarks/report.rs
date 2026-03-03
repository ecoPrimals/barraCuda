// SPDX-License-Identifier: AGPL-3.0-or-later
//! Benchmark Report Generation
//!
//! Generate comprehensive reports from benchmark results.
//!
//! **Deep Debt Evolution (Feb 16, 2026)**: Refactored to use `write!` macro
//! instead of `push_str` + `format!` for more idiomatic Rust.

use super::ComparisonResult;
use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Report generator for benchmark results
pub struct ReportGenerator {
    results: Vec<ComparisonResult>,
}

impl ReportGenerator {
    /// Create new report generator
    pub fn new(results: Vec<ComparisonResult>) -> Self {
        Self { results }
    }

    /// Generate markdown report
    ///
    /// Uses `write!` macro for cleaner string formatting (idiomatic Rust).
    pub fn generate_markdown(&self) -> String {
        let mut report = String::new();

        // Header
        writeln!(report, "# BarraCuda vs CUDA Performance Report\n").unwrap_or_default();

        {
            use std::time::SystemTime;
            let timestamp = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| format!("{}s since epoch", d.as_secs()))
                .unwrap_or_else(|_| "[unknown]".to_string());
            writeln!(report, "**Generated:** {timestamp}\n").unwrap_or_default();
        }

        // Summary section
        self.write_summary(&mut report);

        // Detailed results
        self.write_detailed_results(&mut report);

        report
    }

    /// Write summary statistics section
    fn write_summary(&self, report: &mut String) {
        writeln!(report, "## Summary\n").unwrap_or_default();

        let total_ops = self.results.len();
        if total_ops == 0 {
            writeln!(report, "_No benchmark results available._\n").unwrap_or_default();
            return;
        }

        let ops_with_cuda: Vec<_> = self.results.iter().filter(|r| r.cuda.is_some()).collect();
        let cuda_count = ops_with_cuda.len();

        if cuda_count == 0 {
            writeln!(report, "_No CUDA comparison data available._\n").unwrap_or_default();
            return;
        }

        let ops_at_90 = ops_with_cuda
            .iter()
            .filter(|r| r.parity_percent >= 90.0)
            .count();
        let ops_at_95 = ops_with_cuda
            .iter()
            .filter(|r| r.parity_percent >= 95.0)
            .count();
        let ops_at_100 = ops_with_cuda
            .iter()
            .filter(|r| r.parity_percent >= 100.0)
            .count();
        let avg_parity =
            ops_with_cuda.iter().map(|r| r.parity_percent).sum::<f64>() / cuda_count as f64;

        writeln!(report, "| Metric | Value |").unwrap_or_default();
        writeln!(report, "|--------|-------|").unwrap_or_default();
        writeln!(report, "| Total Operations | {total_ops} |").unwrap_or_default();
        writeln!(report, "| Operations with CUDA comparison | {cuda_count} |").unwrap_or_default();
        writeln!(
            report,
            "| ≥90% parity | {} ({:.1}%) |",
            ops_at_90,
            ops_at_90 as f64 / cuda_count as f64 * 100.0
        )
        .unwrap_or_default();
        writeln!(
            report,
            "| ≥95% parity | {} ({:.1}%) |",
            ops_at_95,
            ops_at_95 as f64 / cuda_count as f64 * 100.0
        )
        .unwrap_or_default();
        writeln!(
            report,
            "| ≥100% parity (faster) | {} ({:.1}%) |",
            ops_at_100,
            ops_at_100 as f64 / cuda_count as f64 * 100.0
        )
        .unwrap_or_default();
        writeln!(report, "| Average parity | {avg_parity:.1}% |").unwrap_or_default();
        writeln!(report).unwrap_or_default();
    }

    /// Write detailed results for each operation
    fn write_detailed_results(&self, report: &mut String) {
        writeln!(report, "## Detailed Results\n").unwrap_or_default();

        for result in &self.results {
            writeln!(report, "### {}\n", result.operation).unwrap_or_default();
            writeln!(report, "- **Hardware:** {}", result.hardware).unwrap_or_default();
            writeln!(
                report,
                "- **BarraCuda:** {:.3}ms",
                result.barracuda.median_time.as_secs_f64() * 1000.0
            )
            .unwrap_or_default();

            if let Some(ref cuda) = result.cuda {
                writeln!(
                    report,
                    "- **CUDA:** {:.3}ms",
                    cuda.median_time.as_secs_f64() * 1000.0
                )
                .unwrap_or_default();
                writeln!(report, "- **Parity:** {:.1}%", result.parity_percent).unwrap_or_default();
            }
            writeln!(report).unwrap_or_default();
        }
    }

    /// Save report to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let report = self.generate_markdown();
        let mut file = File::create(path)?;
        file.write_all(report.as_bytes())?;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::benchmarks::{BenchmarkResult, ComparisonResult, Framework};
    use std::time::Duration;

    fn make_result(op: &str, hw: &str, framework: Framework, median_ms: f64) -> BenchmarkResult {
        let median = Duration::from_secs_f64(median_ms / 1000.0);
        BenchmarkResult {
            operation: op.to_string(),
            hardware: hw.to_string(),
            framework,
            median_time: median,
            mean_time: median,
            std_dev: Duration::from_millis(1),
            min_time: median,
            max_time: median,
            throughput: 1000.0 / median_ms,
            bandwidth_gbps: 0.0,
            tflops: 0.0,
        }
    }

    #[test]
    fn test_empty_report() {
        let gen = ReportGenerator::new(vec![]);
        let md = gen.generate_markdown();
        assert!(md.contains("No benchmark results"));
    }

    #[test]
    fn test_report_with_results() {
        let barracuda = make_result("MatMul", "CPU", Framework::BarraCuda, 10.0);
        let results = vec![ComparisonResult::new(barracuda, None)];
        let gen = ReportGenerator::new(results);
        let md = gen.generate_markdown();
        assert!(md.contains("MatMul"));
    }

    #[test]
    fn test_report_with_cuda_comparison() {
        let barracuda = make_result("MatMul", "CPU", Framework::BarraCuda, 10.0);
        let cuda = make_result("MatMul", "CPU", Framework::CUDA, 8.0);
        let results = vec![ComparisonResult::new(barracuda, Some(cuda))];
        let gen = ReportGenerator::new(results);
        let md = gen.generate_markdown();
        assert!(md.contains("MatMul"));
        assert!(md.contains("parity") || md.contains("Parity"));
        assert!(md.contains("≥90%") || md.contains("90%"));
    }

    #[test]
    fn test_save_to_file() {
        let barracuda = make_result("MatMul", "CPU", Framework::BarraCuda, 10.0);
        let results = vec![ComparisonResult::new(barracuda, None)];
        let gen = ReportGenerator::new(results);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("report.md");
        gen.save_to_file(&path).unwrap();
        assert!(path.exists());
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("MatMul"));
    }
}
