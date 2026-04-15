// SPDX-License-Identifier: AGPL-3.0-or-later
//! F64 polyfill injection and driver-specific patching.
//!
//! Handles:
//! - Injecting missing `_f64` polyfill functions (exp, log, sin, etc.)
//! - Transcendental workaround: native f64 → polyfill for drivers without
//!   double-precision transcendentals (e.g. NVIDIA)
//! - Fossil function substitution: legacy `abs_f64(` → native `abs(`

use super::math_f64::{
    F64_FOSSIL_FUNCTIONS, F64_FUNCTION_DEPS, F64_FUNCTION_ORDER, extract_wgsl_function,
};

/// Taylor-series sin/cos for drivers with broken f64 implementations (NVK).
/// 7-term Taylor for |x| ≤ π, with range reduction. cos derived from sin.
pub(crate) const SIN_COS_F64_SAFE_PREAMBLE: &str = r"
// sin_f64_safe: 7-term Taylor for |x| ≤ π, with range reduction (NVK workaround)
fn sin_f64_safe(x: f64) -> f64 {
    let pi = 3.14159265358979323846;
    let two_pi = 6.28318530717958647692;
    var t = x - floor(x / two_pi) * two_pi;
    if (t < 0.0) { t = t + two_pi; }
    if (t > pi) { t = t - two_pi; }
    let t2 = t * t;
    let t3 = t2 * t;
    return t - t3 / 6.0 + t2 * t3 / 120.0 - t2 * t2 * t3 / 5040.0
           + t2 * t2 * t2 * t3 / 362880.0;
}

// cos_f64_safe: derived from sin
fn cos_f64_safe(x: f64) -> f64 {
    return sin_f64_safe(x + 1.5707963267948966);
}
";

/// Full `math_f64` library preamble (fossils + core + special).
///
/// Fossils are included for backward compatibility with legacy shaders
/// that still reference `abs_f64(`, `sqrt_f64(` etc. New shaders should
/// use native WGSL builtins directly; `substitute_fossil_f64()` rewrites
/// legacy calls.
#[must_use]
pub fn math_f64_preamble() -> String {
    let fossils = include_str!("../math/math_f64_fossils.wgsl");
    let core = include_str!("../math/math_f64.wgsl");
    let special = include_str!("../math/math_f64_special.wgsl");
    format!("{fossils}\n{core}\n{special}")
}

/// Replace legacy fossil f64 function calls with native WGSL equivalents.
///
/// Probe-confirmed native on all `SHADER_F64` hardware via Vulkan (Feb 2026):
/// `abs`, `sign`, `floor`, `ceil`, `round`, `fract`, `min`, `max`, `clamp`, `sqrt`.
///
/// Rewrites `abs_f64(` → `abs(`, `sqrt_f64(` → `sqrt(` etc. in legacy shaders.
/// New shaders must use native WGSL builtins directly — this method is the
/// migration path for older code still using the `_f64` names.
#[must_use]
pub fn substitute_fossil_f64(shader_body: &str) -> String {
    let mut result = shader_body.to_string();
    for (fossil_name, native_name) in F64_FOSSIL_FUNCTIONS {
        let from = format!("{fossil_name}(");
        let to = format!("{native_name}(");
        result = result.replace(&from, &to);
    }
    result
}

/// Replace native f64 transcendentals with their polyfill equivalents while
/// preserving WGSL comments so generated shader source stays readable.
///
/// Covers: `exp`, `log`, `pow`, `sin`/`cos`/`tan` (plus inverse variants
/// `asin`/`acos`/`atan` via suffix matching), `sinh`/`cosh`/`tanh`, `atan2`.
///
/// When `use_sin_cos_taylor` is true (e.g. NVK), uses `sin_f64_safe`/`cos_f64_safe`
/// Taylor series instead of `sin_f64`/`cos_f64`, and protects `asin`/`acos` from
/// being mangled (they stay as `asin_f64`/`acos_f64` from the polyfill).
///
/// Processes the shader line-by-line:
/// - Pure comment lines (`//…`) are passed through unchanged.
/// - Lines with inline comments have only the code portion patched.
/// - Block comments `/* … */` are not yet handled (rare in WGSL compute
///   shaders; revisit when encountered).
#[must_use]
pub fn apply_transcendental_workaround_with_sin_cos(
    shader: &str,
    needs_exp_log: bool,
    use_sin_cos_taylor: bool,
) -> String {
    shader
        .lines()
        .map(|line| {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") {
                return line.to_string();
            }
            if let Some(comment_start) = line.find("//") {
                let code = &line[..comment_start];
                let comment = &line[comment_start..];
                let patched =
                    patch_transcendentals_in_code(code, needs_exp_log, use_sin_cos_taylor);
                format!("{patched}{comment}")
            } else {
                patch_transcendentals_in_code(line, needs_exp_log, use_sin_cos_taylor)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Legacy entry point: applies full transcendental workaround (exp, log, sin, cos, etc.).
#[must_use]
pub fn apply_transcendental_workaround(shader: &str) -> String {
    apply_transcendental_workaround_with_sin_cos(shader, true, false)
}

/// Replace native f64 transcendentals with polyfill calls in a code fragment.
///
/// NVVM's PTXAS does not implement double-precision transcendentals —
/// they require libdevice which SPIR-V cannot link. All known NVIDIA
/// proprietary drivers (Ampere, Ada Lovelace, Hopper) and NVK/RADV
/// exhibit this. The `_f64` polyfills (Cody-Waite + minimax polynomial)
/// are defined in `math_f64.wgsl` and auto-injected by
/// `inject_f64_polyfills`.
///
/// When `use_sin_cos_taylor` is true (NVK), replaces `sin(`/`cos(` with
/// `sin_f64_safe`/`cos_f64_safe` and protects `asin`/`acos` from being
/// mangled (they become `asin_f64`/`acos_f64` from the polyfill).
#[inline]
#[must_use]
pub fn patch_transcendentals_in_code(
    code: &str,
    needs_exp_log: bool,
    use_sin_cos_taylor: bool,
) -> String {
    // Protect WGSL builtins and DF64 functions whose names contain
    // transcendental substrings (e.g. ldexp contains "exp", exp_df64
    // contains "exp") from being mangled by the substring replacer.
    let mut s = code
        .replace("ldexp(", "\x00LDEXP\x00")
        .replace("exp_df64(", "\x00EXP_DF64\x00")
        .replace("exp_f64(", "\x00EXP_F64\x00")
        .replace("log_df64(", "\x00LOG_DF64\x00")
        .replace("log_f64(", "\x00LOG_F64\x00")
        .replace("sin_f64_safe(", "\x00SIN_F64_SAFE\x00")
        .replace("cos_f64_safe(", "\x00COS_F64_SAFE\x00");

    // Protect asin/acos so sin/cos replacement does not mangle them into asin_f64_safe.
    s = s
        .replace("asin(", "\x00ASIN\x00")
        .replace("acos(", "\x00ACOS\x00");

    if needs_exp_log {
        s = s
            .replace("exp(", "exp_f64(")
            .replace("log(", "log_f64(")
            .replace("pow(", "pow_f64(")
            .replace("sinh(", "sinh_f64(")
            .replace("cosh(", "cosh_f64(")
            .replace("tanh(", "tanh_f64(")
            .replace("tan(", "tan_f64(")
            .replace("atan2(", "atan2_f64(");

        if use_sin_cos_taylor {
            s = s
                .replace("sin(", "sin_f64_safe(")
                .replace("cos(", "cos_f64_safe(");
        } else {
            s = s.replace("sin(", "sin_f64(").replace("cos(", "cos_f64(");
        }
    }

    // Restore protected tokens. asin/acos become asin_f64/acos_f64 when workaround active.
    let asin_restore = if needs_exp_log { "asin_f64(" } else { "asin(" };
    let acos_restore = if needs_exp_log { "acos_f64(" } else { "acos(" };
    s.replace("\x00LDEXP\x00", "ldexp(")
        .replace("\x00EXP_DF64\x00", "exp_df64(")
        .replace("\x00EXP_F64\x00", "exp_f64(")
        .replace("\x00LOG_DF64\x00", "log_df64(")
        .replace("\x00LOG_F64\x00", "log_f64(")
        .replace("\x00SIN_F64_SAFE\x00", "sin_f64_safe(")
        .replace("\x00COS_F64_SAFE\x00", "cos_f64_safe(")
        .replace("\x00ASIN\x00", asin_restore)
        .replace("\x00ACOS\x00", acos_restore)
}

/// Inject only the f64 polyfill functions that are called but not defined.
///
/// Skips fossil functions (use native WGSL builtins). Resolves dependencies
/// and hoists `enable` directives above injected code.
///
/// When `extra_preamble` is `Some`, it is prepended to the injected preamble
/// (e.g. `sin_f64_safe/cos_f64_safe` Taylor series for NVK).
#[must_use]
pub fn inject_f64_polyfills(shader_body: &str, extra_preamble: Option<&str>) -> String {
    let mut preamble = String::new();
    if let Some(extra) = extra_preamble {
        preamble.push_str(extra);
        preamble.push('\n');
    }

    let missing_functions: Vec<&str> = F64_FUNCTION_ORDER
        .iter()
        .filter(|func_name| {
            // Fossil functions are universally-native WGSL builtins on all
            // SHADER_F64 hardware — never inject them; use native calls directly.
            !F64_FOSSIL_FUNCTIONS.iter().any(|(f, _)| *f == **func_name)
        })
        .filter(|func_name| {
            let call_pattern = format!("{func_name}(");
            shader_body.contains(&call_pattern) && !shader_defines_function(shader_body, func_name)
        })
        .copied()
        .collect();
    if missing_functions.is_empty() && extra_preamble.is_none() {
        return shader_body.to_string();
    }
    let full_lib = math_f64_preamble();
    preamble.push_str("// math_f64 driver workaround - auto-injected\n");
    if !missing_functions.is_empty() && !shader_defines_function(shader_body, "f64_const") {
        preamble
            .push_str("fn f64_const(x: f64, c: f32) -> f64 {\n    return x - x + f64(c);\n}\n\n");
    }
    let mut all_needed = std::collections::HashSet::new();
    for func in &missing_functions {
        collect_deps(func, &mut all_needed);
    }
    for func_name in F64_FUNCTION_ORDER {
        if all_needed.contains(*func_name) && !shader_defines_function(shader_body, func_name) {
            if let Some(func_code) = extract_wgsl_function(&full_lib, func_name) {
                // Substitute fossil calls (sqrt_f64, etc.) with native builtins in extracted code
                let substituted = substitute_fossil_f64(&func_code);
                preamble.push_str(&substituted);
                preamble.push_str("\n\n");
            }
        }
    }
    let (enables, rest) = split_enable_directives(shader_body);
    if enables.is_empty() {
        format!("{preamble}\n{shader_body}")
    } else {
        format!("{enables}\n{preamble}\n{rest}")
    }
}

/// Split `enable ...;` directives (must precede all declarations in WGSL)
/// from the rest of the shader body so they can be hoisted above injected code.
#[must_use]
pub fn split_enable_directives(source: &str) -> (String, String) {
    let mut enables = String::new();
    let mut rest = String::new();
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("enable ") || trimmed.starts_with("//") || trimmed.is_empty() {
            if trimmed.starts_with("enable ") {
                enables.push_str(line);
                enables.push('\n');
            } else {
                rest.push_str(line);
                rest.push('\n');
            }
        } else {
            rest.push_str(line);
            rest.push('\n');
        }
    }
    (enables, rest)
}

fn collect_deps<'a>(name: &'a str, needed: &mut std::collections::HashSet<&'a str>) {
    if needed.contains(name) {
        return;
    }
    needed.insert(name);
    for (func, func_deps) in F64_FUNCTION_DEPS {
        if *func == name {
            for dep in func_deps.iter().copied() {
                collect_deps(dep, needed);
            }
            break;
        }
    }
}

/// Build a subset of `math_f64` containing only the requested functions and deps.
#[must_use]
pub fn math_f64_subset(functions: &[&str]) -> String {
    use std::collections::HashSet;
    let deps = F64_FUNCTION_DEPS;
    let mut needed: HashSet<&str> = HashSet::new();

    fn add_with_deps<'a>(
        name: &'a str,
        needed: &mut HashSet<&'a str>,
        deps: &'a [(&'a str, &'a [&'a str])],
    ) {
        if needed.contains(name) {
            return;
        }
        needed.insert(name);
        for (func, func_deps) in deps {
            if *func == name {
                for dep in func_deps.iter().copied() {
                    add_with_deps(dep, needed, deps);
                }
                break;
            }
        }
    }
    for func in functions {
        add_with_deps(func, &mut needed, deps);
    }
    if needed.is_empty() {
        return String::new();
    }
    let full_lib = math_f64_preamble();
    let mut output = String::new();
    output.push_str(
        "// math_f64 subset - auto-generated\n\
         fn f64_const(x: f64, c: f32) -> f64 {\n    return x - x + f64(c);\n}\n\n",
    );
    for func_name in F64_FUNCTION_ORDER {
        if needed.contains(*func_name) {
            if let Some(func_code) = extract_wgsl_function(&full_lib, func_name) {
                output.push_str(&func_code);
                output.push_str("\n\n");
            }
        }
    }
    output
}

/// Check if the shader body defines a function with the given name.
#[must_use]
pub fn shader_defines_function(shader_body: &str, func_name: &str) -> bool {
    let pattern1 = format!("fn {func_name}(");
    let pattern2 = format!("fn {func_name} (");
    shader_body.contains(&pattern1) || shader_body.contains(&pattern2)
}

/// Check if the shader defines a module-level variable with the given name.
#[must_use]
pub fn shader_defines_module_var(shader_body: &str, var_name: &str) -> bool {
    for line in shader_body.lines() {
        let trimmed = line.trim();
        let is_decl =
            line.starts_with("let ") || line.starts_with("var ") || line.starts_with("const ");
        let has_var = trimmed.contains(&format!("{var_name} "))
            || trimmed.contains(&format!("{var_name}="))
            || trimmed.contains(&format!("{var_name}:"));
        if is_decl && has_var {
            return true;
        }
    }
    false
}
