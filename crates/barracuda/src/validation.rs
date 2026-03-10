// SPDX-License-Identifier: AGPL-3.0-only

//! Validation harness for barracuda validation binaries.
//!
//! Provides [`ValidationHarness`] for structured, machine-readable pass/fail
//! checks with explicit tolerances. Every validation binary uses this to
//! accumulate checks and produce a deterministic exit code.
//!
//! Also provides [`exit_no_gpu`] for graceful GPU-optional handling.
//!
//! # Provenance
//! Absorbed from neuralSpring `src/validation.rs` (Feb 2026), itself adapted
//! from the hotSpring validation pattern.

use std::process;

/// How a tolerance threshold is applied.
#[derive(Debug, Clone, Copy)]
pub enum ToleranceMode {
    /// |observed - expected| < tolerance
    Absolute,
    /// |observed - expected| / |expected| < tolerance
    Relative,
    /// observed < threshold (upper bound only)
    UpperBound,
    /// observed > threshold (lower bound only)
    LowerBound,
}

impl std::fmt::Display for ToleranceMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Absolute => write!(f, "abs"),
            Self::Relative => write!(f, "rel"),
            Self::UpperBound => write!(f, "<"),
            Self::LowerBound => write!(f, ">"),
        }
    }
}

/// A single validation check with result tracking.
#[derive(Debug, Clone)]
pub struct Check {
    /// Human-readable label for the check
    pub label: String,
    /// Whether the check passed
    pub passed: bool,
    /// Observed value from the test
    pub observed: f64,
    /// Expected value for comparison
    pub expected: f64,
    /// Tolerance threshold (interpretation depends on `mode`)
    pub tolerance: f64,
    /// How the tolerance is applied
    pub mode: ToleranceMode,
}

/// Accumulates validation checks and produces a summary with exit code.
#[derive(Debug, Default)]
pub struct ValidationHarness {
    /// Name of the validation suite (e.g. "`matmul_validation`")
    pub name: String,
    /// Accumulated checks
    pub checks: Vec<Check>,
}

impl ValidationHarness {
    /// Create a new validation harness with the given suite name.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            checks: Vec::new(),
        }
    }

    /// Absolute tolerance check: |observed - expected| < tolerance.
    pub fn check_abs(&mut self, label: &str, observed: f64, expected: f64, tolerance: f64) {
        let passed = (observed - expected).abs() < tolerance;
        self.checks.push(Check {
            label: label.to_string(),
            passed,
            observed,
            expected,
            tolerance,
            mode: ToleranceMode::Absolute,
        });
    }

    /// Relative tolerance check: |observed - expected| / |expected| < tolerance.
    pub fn check_rel(&mut self, label: &str, observed: f64, expected: f64, tolerance: f64) {
        let passed = if expected.abs() > f64::EPSILON {
            ((observed - expected) / expected).abs() < tolerance
        } else {
            observed.abs() < tolerance
        };
        self.checks.push(Check {
            label: label.to_string(),
            passed,
            observed,
            expected,
            tolerance,
            mode: ToleranceMode::Relative,
        });
    }

    /// Upper-bound check: observed < threshold.
    pub fn check_upper(&mut self, label: &str, observed: f64, threshold: f64) {
        self.checks.push(Check {
            label: label.to_string(),
            passed: observed < threshold,
            observed,
            expected: threshold,
            tolerance: threshold,
            mode: ToleranceMode::UpperBound,
        });
    }

    /// Lower-bound check: observed > threshold.
    pub fn check_lower(&mut self, label: &str, observed: f64, threshold: f64) {
        self.checks.push(Check {
            label: label.to_string(),
            passed: observed > threshold,
            observed,
            expected: threshold,
            tolerance: threshold,
            mode: ToleranceMode::LowerBound,
        });
    }

    /// Boolean pass/fail check.
    pub fn check_bool(&mut self, label: &str, passed: bool) {
        self.checks.push(Check {
            label: label.to_string(),
            passed,
            observed: f64::from(u8::from(passed)),
            expected: 1.0,
            tolerance: 0.0,
            mode: ToleranceMode::Absolute,
        });
    }

    /// Try to unwrap a `Result`, recording a FAIL check on error.
    ///
    /// Returns `Some(value)` on success, `None` on failure.
    pub fn require<T, E: std::fmt::Display>(
        &mut self,
        label: &str,
        result: Result<T, E>,
    ) -> Option<T> {
        match result {
            Ok(v) => Some(v),
            Err(e) => {
                self.check_bool(&format!("{label}: {e}"), false);
                None
            }
        }
    }

    /// Number of checks that passed.
    #[must_use]
    pub fn passed_count(&self) -> usize {
        self.checks.iter().filter(|c| c.passed).count()
    }

    /// Total number of checks.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.checks.len()
    }

    /// Returns true if all checks passed.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.checks.iter().all(|c| c.passed)
    }

    /// Print summary and exit with appropriate code.
    pub fn finish(&self) -> ! {
        tracing::info!("");
        for check in &self.checks {
            let icon = if check.passed { "PASS" } else { "FAIL" };
            tracing::info!(
                "[{icon}] {}: observed={:.10e}, expected={:.10e}, tol={:.2e} ({})",
                check.label,
                check.observed,
                check.expected,
                check.tolerance,
                check.mode
            );
        }

        tracing::info!("");
        tracing::info!(
            "=== {}: {}/{} PASS, {} FAIL ===",
            self.name,
            self.passed_count(),
            self.total_count(),
            self.total_count() - self.passed_count(),
        );

        if self.all_passed() {
            process::exit(0);
        } else {
            let failed: Vec<&str> = self
                .checks
                .iter()
                .filter(|c| !c.passed)
                .map(|c| c.label.as_str())
                .collect();
            tracing::info!("FAILED: {}", failed.join(", "));
            process::exit(1);
        }
    }
}

/// Parse a `BARRACUDA_REQUIRE_GPU` value into a boolean.
///
/// Accepts `"1"` or `"true"` (case-insensitive) as truthy.
#[must_use]
fn parse_gpu_required(val: Option<&str>) -> bool {
    val.is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// Whether `BARRACUDA_REQUIRE_GPU=1` is set.
///
/// When `true`, validation binaries that cannot obtain a GPU adapter must exit 1
/// instead of silently skipping. For CI pipelines with a known-good GPU.
#[must_use]
pub fn gpu_required() -> bool {
    parse_gpu_required(std::env::var("BARRACUDA_REQUIRE_GPU").ok().as_deref())
}

/// Handle the absence of a GPU adapter in a validation binary.
///
/// If `BARRACUDA_REQUIRE_GPU=1`, prints an error and exits 1.
/// Otherwise, prints a skip message and exits 0.
pub fn exit_no_gpu() -> ! {
    if gpu_required() {
        tracing::error!("FAIL: no GPU adapter (BARRACUDA_REQUIRE_GPU=1)");
        process::exit(1);
    }
    tracing::info!("0/0 checks — skipping gracefully (no GPU adapter)");
    process::exit(0);
}

/// Unwrap a `Result` or record failure and early-return from the caller.
#[macro_export]
macro_rules! require {
    ($harness:expr, $result:expr, $label:expr) => {
        match $result {
            Ok(v) => v,
            Err(e) => {
                $harness.check_bool(&format!("{}: {}", $label, e), false);
                return;
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harness_tracks_pass_fail() {
        let mut h = ValidationHarness::new("test");
        h.check_abs("good", 1.0, 1.0, 0.01);
        h.check_abs("bad", 1.0, 2.0, 0.01);
        assert_eq!(h.passed_count(), 1);
        assert_eq!(h.total_count(), 2);
        assert!(!h.all_passed());
    }

    #[test]
    fn harness_all_pass() {
        let mut h = ValidationHarness::new("test");
        h.check_abs("a", 1.0, 1.0, 0.01);
        h.check_rel("b", 100.0, 100.0, 0.01);
        h.check_bool("c", true);
        assert!(h.all_passed());
    }

    #[test]
    fn harness_bounds() {
        let mut h = ValidationHarness::new("test");
        h.check_upper("upper ok", 0.5, 1.0);
        h.check_upper("upper fail", 1.5, 1.0);
        h.check_lower("lower ok", 1.5, 1.0);
        h.check_lower("lower fail", 0.5, 1.0);
        assert_eq!(h.passed_count(), 2);
    }

    #[test]
    fn harness_require_success() {
        let mut h = ValidationHarness::new("test");
        let val: Result<i32, String> = Ok(42);
        let result = h.require("op", val);
        assert_eq!(result, Some(42));
        assert!(h.all_passed());
    }

    #[test]
    fn harness_require_failure() {
        let mut h = ValidationHarness::new("test");
        let val: Result<i32, String> = Err("boom".into());
        let result = h.require("op", val);
        assert_eq!(result, None);
        assert!(!h.all_passed());
    }

    #[test]
    fn gpu_required_default_false() {
        assert!(!parse_gpu_required(None));
    }

    #[test]
    fn test_tolerance_mode_display() {
        assert_eq!(ToleranceMode::Absolute.to_string(), "abs");
        assert_eq!(ToleranceMode::Relative.to_string(), "rel");
        assert_eq!(ToleranceMode::UpperBound.to_string(), "<");
        assert_eq!(ToleranceMode::LowerBound.to_string(), ">");
    }

    #[test]
    fn test_check_rel_with_zero_expected() {
        let mut h = ValidationHarness::new("test");
        h.check_rel("zero_expected_pass", 0.0001, 0.0, 0.01);
        h.check_rel("zero_expected_fail", 1.0, 0.0, 0.01);
        assert!(h.checks[0].passed);
        assert!(!h.checks[1].passed);
    }

    #[test]
    fn test_empty_harness() {
        let h = ValidationHarness::new("empty");
        assert!(h.all_passed());
        assert_eq!(h.passed_count(), 0);
        assert_eq!(h.total_count(), 0);
    }

    #[test]
    fn test_check_bool_false() {
        let mut h = ValidationHarness::new("test");
        h.check_bool("x", false);
        assert_eq!(h.passed_count(), 0);
        assert_eq!(h.total_count(), 1);
        assert!(!h.all_passed());
    }

    #[test]
    fn test_gpu_required_when_set_to_1() {
        assert!(parse_gpu_required(Some("1")));
    }

    #[test]
    fn test_gpu_required_when_set_to_true() {
        assert!(parse_gpu_required(Some("true")));
    }

    #[test]
    fn test_gpu_required_case_insensitive() {
        assert!(parse_gpu_required(Some("TRUE")));
    }

    #[test]
    fn test_gpu_required_false_when_other_value() {
        assert!(!parse_gpu_required(Some("0")));
    }
}
