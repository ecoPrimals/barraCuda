//! Chaos tests — malformed input, boundary values, adversarial patterns.
//! Extracted from precision_tests.rs for maintainability.

use super::super::*;

#[test]
fn test_chaos_empty_source_downcast_f32() {
    let result = downcast_f64_to_f32("");
    assert_eq!(result, "");
}

#[test]
fn test_chaos_empty_source_downcast_df64() {
    let result = downcast_f64_to_df64("");
    assert_eq!(result, "");
}

#[test]
fn test_chaos_empty_source_downcast_f16() {
    let result = downcast_f64_to_f16("");
    assert_eq!(result, "");
}

#[test]
fn test_chaos_no_f64_in_source() {
    let source = "fn main() { let x: f32 = 1.0; }";
    let f32_result = downcast_f64_to_f32(source);
    let df64_result = downcast_f64_to_df64(source);
    let f16_result = downcast_f64_to_f16(source);
    assert_eq!(f32_result, source, "no-f64 source unchanged for f32");
    assert_eq!(df64_result, source, "no-f64 source unchanged for df64");
    assert_eq!(f16_result, source, "no-f64 source unchanged for f16");
}

#[test]
fn test_chaos_nested_f64_patterns() {
    let source = "let x: f64 = f64(f64(1.0));";
    let result = downcast_f64_to_f32(source);
    assert!(
        result.contains("f32(f32(1.0))"),
        "nested constructors downcast: {result}"
    );
    assert!(!result.contains("f64"), "no f64 in result");
}

#[test]
fn test_chaos_f64_in_variable_name() {
    // Text-based downcast replaces ALL occurrences of f64 patterns,
    // including in variable names. This is a known limitation of text replacement.
    // The fix would be naga-IR-based transformation. Document behavior here.
    let source = "let my_f64_value: f64 = f64(1.0);";
    let result = downcast_f64_to_f32(source);
    // `: f64` in the type position is correctly downcasted
    assert!(result.contains(": f32"), "type annotation downcasted");
    // `f64(` constructor is correctly downcasted
    assert!(result.contains("f32(1.0)"), "constructor downcasted");
}

#[test]
fn test_chaos_multiple_sentinel_f64_calls() {
    let source = "let a = exp_f64(sin_f64(cos_f64(x)));";
    let result = downcast_f64_to_f32(source);
    // f64( should not affect _f64( function suffixes
    assert!(result.contains("exp_f64("), "nested polyfills preserved");
    assert!(result.contains("sin_f64("), "nested polyfills preserved");
    assert!(result.contains("cos_f64("), "nested polyfills preserved");
}

#[test]
fn test_chaos_mixed_precision_source() {
    let source = "let a: f32 = 1.0;\nlet b: f64 = f64(2.0);\nlet c: u32 = 3u;";
    let result = downcast_f64_to_f32(source);
    assert!(result.contains("a: f32"), "f32 preserved");
    assert!(result.contains("b: f32"), "f64 → f32");
    assert!(result.contains("c: u32"), "u32 preserved");
    assert!(result.contains("f32(2.0)"), "constructor downcasted");
}

#[test]
fn test_chaos_consecutive_sentinels() {
    let source = "-1e308 + 1e308 + -1e300 + 1e300";
    let result = downcast_f64_to_f32(source);
    assert!(!result.contains("e308"), "all e308 clamped");
    assert!(!result.contains("e300"), "all e300 clamped");
    assert_eq!(
        result.matches("3.4028235e+38").count(),
        4,
        "four clamped values"
    );
}
