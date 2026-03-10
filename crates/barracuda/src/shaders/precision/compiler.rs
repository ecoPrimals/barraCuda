// SPDX-License-Identifier: AGPL-3.0-only
//! Shader compilation pipeline: type substitution and downcast transforms.
//!
//! Handles f64 → f32 and f64 → df64 text-based transforms. The "math is
//! universal, precision is silicon" philosophy: one f64-canonical shader
//! produces variants for all precisions.

/// Detect whether WGSL source is f64-canonical (contains f64 type declarations).
///
/// Checks for `array<f64>`, struct field / param types (`: f64,`, `: f64;`,
/// `: f64\n`), and return types (`-> f64`).
#[must_use]
pub fn source_is_f64(source: &str) -> bool {
    source.contains("array<f64>")
        || source.contains(": f64,")
        || source.contains(": f64;")
        || source.contains(": f64\n")
        || source.contains("-> f64")
}

/// Downcast an f64 shader source to f32 via text substitution.
///
/// This is the core of "math is universal, precision is silicon": the shader
/// is written once in f64 (the conceptually true math), and this function
/// produces the f32 variant by replacing type declarations. Only safe for
/// shaders that use basic arithmetic (`+`, `-`, `*`, `/`, `fma`). Shaders
/// with f64 polyfill calls (`exp_f64`, `sin_f64`, etc.) need
/// `downcast_f64_to_f32_with_transcendentals` instead.
#[must_use]
pub fn downcast_f64_to_f32(f64_source: &str) -> String {
    let result = f64_source
        .replace("enable f64;\n", "")
        .replace("enable f64;", "")
        .replace("_f64(", "\x00_F64_CALL\x00")
        .replace("array<f64>", "array<f32>")
        .replace("array<f64,", "array<f32,")
        .replace(": f64", ": f32")
        .replace("-> f64", "-> f32")
        .replace("f64(", "f32(")
        .replace("<f64>", "<f32>")
        .replace("\x00_F64_CALL\x00", "_f64(");

    clamp_f64_range_literals(&result)
}

/// Replace f64-range sentinel literals with f32-safe equivalents.
///
/// f64 canonical shaders use values like `-1e308` or `1.7976931348623157e+308`
/// as min/max initialization sentinels. These exceed f32 range (~3.4e38) and
/// cause WGSL parse errors when downcasted. We replace them with the
/// corresponding f32 extremes.
#[must_use]
pub fn clamp_f64_range_literals(source: &str) -> String {
    source
        .replace("-1.7976931348623157e+308", "-3.4028235e+38")
        .replace("1.7976931348623157e+308", "3.4028235e+38")
        .replace("-1.0e308", "-3.4028235e+38")
        .replace("1.0e308", "3.4028235e+38")
        .replace("-1e308", "-3.4028235e+38")
        .replace("1e308", "3.4028235e+38")
        .replace("-1.0e300", "-3.4028235e+38")
        .replace("1.0e300", "3.4028235e+38")
        .replace("-1e300", "-3.4028235e+38")
        .replace("1e300", "3.4028235e+38")
}

/// Downcast an f64 shader source to f32, also replacing polyfill
/// transcendental calls with native WGSL builtins.
///
/// `exp_f64(x)` → `exp(x)`, `sin_f64(x)` → `sin(x)`, etc.
/// Use for shaders that call `math_f64` polyfill functions.
#[must_use]
pub fn downcast_f64_to_f32_with_transcendentals(f64_source: &str) -> String {
    let base = downcast_f64_to_f32(f64_source);
    base.replace("exp_f64(", "exp(")
        .replace("log_f64(", "log(")
        .replace("pow_f64(", "pow(")
        .replace("sin_f64(", "sin(")
        .replace("cos_f64(", "cos(")
        .replace("tan_f64(", "tan(")
        .replace("asin_f64(", "asin(")
        .replace("acos_f64(", "acos(")
        .replace("atan_f64(", "atan(")
        .replace("atan2_f64(", "atan2(")
        .replace("sinh_f64(", "sinh(")
        .replace("cosh_f64(", "cosh(")
        .replace("tanh_f64(", "tanh(")
        .replace("sqrt_f64(", "sqrt(")
        .replace("abs_f64(", "abs(")
        .replace("erf_f64(", "erf(")
}

/// Transform an f64 shader source to DF64 (f32-pair) representation.
///
/// Handles:
/// - Storage types: `array<f64>` → `array<vec2<f32>>`
/// - Type declarations: `: f64` → `: Df64`, `-> f64` → `-> Df64`
/// - Constructors: `f64(X)` → `df64_from_f32(X)` for literal casts
/// - Transcendentals: `exp_f64(` → `exp_df64(`, `sin_f64(` → `sin_df64(`, etc.
/// - Polyfill builtins: `abs_f64(` → `df64_abs(`, `sqrt_f64(` → `sqrt_df64(`
/// - Sentinels: f64-range literals clamped to f32-range (same as f32 downcast)
///
/// Does NOT handle infix arithmetic operators (`+`, `-`, `*`, `/`) between
/// f64 values — these require naga-IR-based rewriting (see `df64_rewrite` module).
/// Shaders that only use `_f64()` function calls work fully with this transform.
///
/// The caller must compile through `compile_shader_df64()` which prepends the
/// DF64 core library (`df64_core.wgsl` + `df64_transcendentals.wgsl`).
#[must_use]
pub fn downcast_f64_to_df64(f64_source: &str) -> String {
    let result = f64_source
        // Protect function-name _f64( from constructor replacement
        .replace("_f64(", "\x00_F64_CALL\x00")
        // Storage: array<f64> → array<vec2<f32>> (DF64 wire format)
        .replace("array<f64>", "array<vec2<f32>>")
        .replace("array<f64,", "array<vec2<f32>,")
        // Type declarations: f64 → Df64
        .replace(": f64", ": Df64")
        .replace("-> f64", "-> Df64")
        // Generic angle brackets
        .replace("<f64>", "<Df64>")
        // Constructor: f64(X) → df64_from_f32(X)
        // df64_from_f32 takes an f32 arg; WGSL auto-converts abstract literals
        .replace("f64(", "df64_from_f32(")
        // Restore function-name suffixes
        .replace("\x00_F64_CALL\x00", "_f64(");

    let with_transcendentals = result
        // Transcendentals with DF64 implementations in df64_transcendentals.wgsl:
        .replace("exp_f64(", "exp_df64(")
        .replace("log_f64(", "log_df64(")
        .replace("pow_f64(", "pow_df64(")
        .replace("sin_f64(", "sin_df64(")
        .replace("cos_f64(", "cos_df64(")
        .replace("tanh_f64(", "tanh_df64(")
        .replace("sqrt_f64(", "sqrt_df64(")
        .replace("abs_f64(", "df64_abs(");

    // Clamp f64-range sentinels to f32-range (DF64 uses f32 components)
    clamp_f64_range_literals(&with_transcendentals)
}
