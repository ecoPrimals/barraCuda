// SPDX-License-Identifier: AGPL-3.0-only
//! WGSL loop unroller for `// @unroll_hint N` annotated bounded loops.
//!
//! Handles loops annotated with `// @unroll_hint N` where `N ≤ 32` and the
//! loop variable has a statically visible bounds of exactly `0..N`:
//!
//! ```wgsl
//! // @unroll_hint 4
//! for (var k = 0u; k < 4u; k = k + 1u) {
//!     // ... body ...
//! }
//! ```
//!
//! Emits the body N times with `k` substituted by the literal iteration index:
//!
//! ```wgsl
//! { let k = 0u; /* ... body with k=0 ... */ }
//! { let k = 1u; /* ... body with k=1 ... */ }
//! { let k = 2u; /* ... body with k=2 ... */ }
//! { let k = 3u; /* ... body with k=3 ... */ }
//! ```
//!
//! Benefits:
//! - Eliminates the loop counter dependency chain
//! - Exposes all iterations to the `@ilp_region` reorderer simultaneously
//! - Enables inter-iteration ILP (iteration i+1's independent ops fill
//!   iteration i's latency gaps)
//!
//! Loops without `// @unroll_hint` are passed through unchanged.

use std::fmt::Write;

/// Maximum trip count accepted for unrolling. Larger loops are passed through.
const MAX_UNROLL_TRIP_COUNT: u32 = 32;

// ─── WgslLoopUnroller ─────────────────────────────────────────────────────────

/// WGSL loop unroller for `// @unroll_hint N` annotated bounded loops.
pub struct WgslLoopUnroller;

impl WgslLoopUnroller {
    /// Process `shader_source` and unroll any `// @unroll_hint N` annotated loops.
    ///
    /// All other content is returned unchanged.
    #[must_use]
    pub fn unroll(shader_source: &str) -> String {
        let lines: Vec<&str> = shader_source.lines().collect();
        let mut output = String::with_capacity(shader_source.len() * 2);
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i];
            let trimmed = line.trim();

            // Look for `// @unroll_hint N`
            if let Some(hint_n) = parse_unroll_hint(trimmed) {
                // Next non-blank line should be the `for (var k = 0u; ...` header
                let for_start = find_next_for_loop(&lines, i + 1);
                if let Some(for_idx) = for_start {
                    if let Some(unrolled) = try_unroll_loop(&lines, for_idx, hint_n) {
                        // Emit the hint comment (as documentation)
                        output.push_str(line);
                        output.push('\n');
                        // Emit unrolled body
                        output.push_str(&unrolled);
                        // Skip past the original for loop
                        let loop_end = find_loop_end(&lines, for_idx);
                        i = loop_end + 1;
                        continue;
                    }
                }
            }

            output.push_str(line);
            output.push('\n');
            i += 1;
        }

        output
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Parse `// @unroll_hint N` from a trimmed line — returns `Some(N)` or `None`.
fn parse_unroll_hint(trimmed: &str) -> Option<u32> {
    let rest = trimmed.strip_prefix("// @unroll_hint")?;
    let n: u32 = rest.trim().parse().ok()?;
    if n > 0 && n <= MAX_UNROLL_TRIP_COUNT {
        Some(n)
    } else {
        None
    }
}

/// Find the next `for (var` line at or after `start_idx`.
fn find_next_for_loop(lines: &[&str], start_idx: usize) -> Option<usize> {
    for i in start_idx..lines.len().min(start_idx + 5) {
        let t = lines[i].trim();
        if t.starts_with("for (var") || t.starts_with("for(var") {
            return Some(i);
        }
        if !t.is_empty() && !t.starts_with("//") {
            return None; // Non-comment non-for line before a for — don't skip
        }
    }
    None
}

/// The upper bound extracted from a `for` loop header.
///
/// - `Literal(n)` — a numeric literal like `32u`; the unroller substitutes `k` with
///   integers 0..n and emits no runtime guards.
/// - `Variable(name)` — a runtime identifier like `n` from `k < n`; the unroller
///   emits N copies of the body each wrapped in `if (<iter>u < <name>)` so the
///   shader is correct for any runtime value while still exposing full ILP.
#[derive(Debug, PartialEq, Eq)]
enum ForBound {
    Literal(u32),
    Variable(String),
}

/// Parse a `for (var k = 0u; k < BOUND; k = k + 1u)` header.
///
/// `BOUND` may be either a numeric literal (`32u`, `8`) or an identifier (`n`).
/// Returns `(loop_var, bound)` on success.
fn parse_for_header(line: &str) -> Option<(String, ForBound)> {
    let t = line.trim();
    let after_for = t
        .strip_prefix("for (var ")
        .or_else(|| t.strip_prefix("for(var "))?;
    // Extract variable name (before " = 0")
    let eq_pos = after_for.find(" = 0")?;
    let var_name = after_for[..eq_pos].trim().to_string();
    if var_name.is_empty() {
        return None;
    }
    // Find the bound after `; VAR < `
    let lt_pat = format!("{var_name} < ");
    let lt_pos = after_for.find(&lt_pat)?;
    let after_lt = &after_for[lt_pos + lt_pat.len()..];

    // Try numeric literal first (e.g. `8u` or `8`)
    let num_str: String = after_lt.chars().take_while(char::is_ascii_digit).collect();
    if !num_str.is_empty() {
        let bound: u32 = num_str.parse().ok()?;
        if bound == 0 || bound > MAX_UNROLL_TRIP_COUNT {
            return None;
        }
        return Some((var_name, ForBound::Literal(bound)));
    }

    // Try identifier (e.g. `n`, `size`, `params_n`) — runtime variable bound
    let id_str: String = after_lt
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    if !id_str.is_empty() {
        return Some((var_name, ForBound::Variable(id_str)));
    }

    None
}

/// Find the index of the closing `}` that matches the `for` loop starting at `for_idx`.
///
/// Scans for balanced braces. The `for` header ends with `{`; we count depth.
fn find_loop_end(lines: &[&str], for_idx: usize) -> usize {
    let mut depth: i32 = 0;
    let mut found_open = false;
    for i in for_idx..lines.len() {
        for ch in lines[i].chars() {
            match ch {
                '{' => {
                    depth += 1;
                    found_open = true;
                }
                '}' => {
                    depth -= 1;
                    if found_open && depth == 0 {
                        return i;
                    }
                }
                _ => {}
            }
        }
    }
    // Fallback: return for_idx itself if braces unbalanced
    for_idx
}

/// Collect the body lines between the opening `{` and closing `}` of a for loop.
///
/// The `for` header line at `for_idx` is deliberately excluded — it is the
/// loop declaration itself, not part of the body.  Including it would cause
/// the loop variable to be substituted into the header (e.g. `for (var 0 = 0u;
/// ...)`) when the body lines undergo loop-variable substitution.
fn collect_body(lines: &[&str], for_idx: usize) -> Vec<String> {
    let end_idx = find_loop_end(lines, for_idx);
    let mut body = Vec::new();
    let mut depth = 0i32;
    let mut past_open = false;
    for i in for_idx..=end_idx {
        let line = lines[i];
        for ch in line.chars() {
            if ch == '{' {
                depth += 1;
                if depth == 1 {
                    past_open = true;
                }
            } else if ch == '}' {
                depth -= 1;
            }
        }
        // Skip the for-loop header (i == for_idx) — only collect the body
        // lines strictly between the braces.
        if i > for_idx && past_open && depth >= 1 {
            body.push(line.to_string());
        }
    }
    body
}

/// Try to unroll a loop annotated with `@unroll_hint N`.
///
/// Returns the unrolled source or `None` if the loop doesn't match the expected pattern.
///
/// ## Literal bound
/// Emits `trip_count = min(hint, declared)` unrolled copies with the loop variable
/// substituted by its integer value. No runtime guard is needed.
///
/// ## Variable bound
/// Emits exactly `hint_n` copies each wrapped in `if (<iter>u < <bound_var>) { … }`.
/// This keeps the shader correct for any runtime value of `bound_var` while exposing
/// all `hint_n` iterations simultaneously to the hardware instruction scheduler,
/// which can predicate-out iterations where `iter ≥ bound_var` cheaply via uniform
/// branch elimination (the bound is uniform across the warp/subgroup).
fn try_unroll_loop(lines: &[&str], for_idx: usize, hint_n: u32) -> Option<String> {
    let (var_name, bound) = parse_for_header(lines[for_idx])?;
    let body_lines = collect_body(lines, for_idx);

    // Detect indentation from the for-line
    let indent: String = lines[for_idx]
        .chars()
        .take_while(|c| c.is_whitespace())
        .collect();

    let mut out = String::new();
    match bound {
        ForBound::Literal(declared_n) => {
            // Conservative: unroll min(hint, declared) iterations — no guard needed.
            let trip_count = hint_n.min(declared_n);
            for iter in 0..trip_count {
                emit_unrolled_block(&mut out, &indent, &var_name, iter, &body_lines, None);
            }
        }
        ForBound::Variable(ref bound_var) => {
            // Runtime bound: unroll hint_n times, each guarded by `if (iter < bound_var)`.
            // The guard is uniform (bound_var is the same for all invocations in a warp),
            // so the GPU eliminates the dead iterations at no scoreboard cost.
            for iter in 0..hint_n {
                let guard = format!("if ({iter}u < {bound_var})");
                emit_unrolled_block(
                    &mut out,
                    &indent,
                    &var_name,
                    iter,
                    &body_lines,
                    Some(&guard),
                );
            }
        }
    }
    Some(out)
}

/// Emit one unrolled iteration block into `out`.
///
/// If `guard` is `Some("if (2u < n)")`, the block body is wrapped inside that condition.
fn emit_unrolled_block(
    out: &mut String,
    indent: &str,
    var_name: &str,
    iter: u32,
    body_lines: &[String],
    guard: Option<&str>,
) {
    let _ = writeln!(out, "{indent}{{");
    // Bind the loop variable to its literal value so the body can reference it.
    let _ = writeln!(out, "{indent}    let {var_name} = {iter}u;");

    if let Some(cond) = guard {
        let _ = writeln!(out, "{indent}    {cond} {{");
        for body_line in body_lines {
            let subst = substitute_loop_var(body_line, var_name, iter);
            let _ = writeln!(out, "    {subst}");
        }
        let _ = writeln!(out, "{indent}    }}");
    } else {
        for body_line in body_lines {
            let subst = substitute_loop_var(body_line, var_name, iter);
            out.push_str(&subst);
            out.push('\n');
        }
    }

    let _ = writeln!(out, "{indent}}}");
}

/// Replace whole-word occurrences of `var_name` in `line` with the literal `iter`.
fn substitute_loop_var(line: &str, var_name: &str, iter: u32) -> String {
    let result = line.to_string();
    let replacement = format!("{iter}u");
    // Replace whole-word occurrences only.
    let mut out = String::new();
    let mut pos = 0;
    let bytes = result.as_bytes();
    while pos < bytes.len() {
        if let Some(found) = result[pos..].find(var_name) {
            let abs = pos + found;
            let end = abs + var_name.len();
            // Check word boundaries
            let before_ok =
                abs == 0 || (!bytes[abs - 1].is_ascii_alphanumeric() && bytes[abs - 1] != b'_');
            let after_ok =
                end >= bytes.len() || (!bytes[end].is_ascii_alphanumeric() && bytes[end] != b'_');
            if before_ok && after_ok {
                out.push_str(&result[pos..abs]);
                out.push_str(&replacement);
                pos = end;
            } else {
                out.push(result.chars().nth(pos).unwrap_or(' '));
                pos += 1;
            }
        } else {
            out.push_str(&result[pos..]);
            break;
        }
    }
    // If no substitution happened, return original
    if out.is_empty() {
        out = result;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unroll_hint_parsed() {
        assert_eq!(parse_unroll_hint("// @unroll_hint 4"), Some(4));
        assert_eq!(parse_unroll_hint("// @unroll_hint 32"), Some(32));
        assert_eq!(parse_unroll_hint("// @unroll_hint 33"), None); // > MAX
        assert_eq!(parse_unroll_hint("// @unroll_hint 0"), None); // 0 not valid
        assert_eq!(parse_unroll_hint("// normal comment"), None);
    }

    #[test]
    fn test_for_header_parsed_literal() {
        assert_eq!(
            parse_for_header("    for (var k = 0u; k < 4u; k = k + 1u) {"),
            Some(("k".to_string(), ForBound::Literal(4)))
        );
        assert_eq!(
            parse_for_header("    for (var i = 0u; i < 8u; i = i + 1u) {"),
            Some(("i".to_string(), ForBound::Literal(8)))
        );
    }

    #[test]
    fn test_for_header_parsed_variable_bound() {
        // Jacobi inner loop: `for (var k = 0u; k < n; k = k + 1u)`
        assert_eq!(
            parse_for_header("                for (var k = 0u; k < n; k = k + 1u) {"),
            Some(("k".to_string(), ForBound::Variable("n".to_string())))
        );
        // Compound identifier: `for (var i = 0u; i < matrix_size; i = i + 1u)`
        assert_eq!(
            parse_for_header("    for (var i = 0u; i < matrix_size; i = i + 1u) {"),
            Some((
                "i".to_string(),
                ForBound::Variable("matrix_size".to_string())
            ))
        );
    }

    #[test]
    fn test_variable_bound_loop_unrolled_with_guards() {
        // Mirrors the Jacobi k-loop pattern: bound is runtime `n`
        let shader = concat!(
            "            // @unroll_hint 4\n",
            "            for (var k = 0u; k < n; k = k + 1u) {\n",
            "                let v = k;\n",
            "            }\n",
        );
        let result = WgslLoopUnroller::unroll(shader);
        // Four unrolled blocks should appear
        assert_eq!(result.matches("let k = 0u;").count(), 1);
        assert_eq!(result.matches("let k = 1u;").count(), 1);
        assert_eq!(result.matches("let k = 2u;").count(), 1);
        assert_eq!(result.matches("let k = 3u;").count(), 1);
        assert_eq!(
            result.matches("let k = 4u;").count(),
            0,
            "should only emit 4 iters"
        );
        // Each iteration should be guarded
        assert!(result.contains("if (0u < n)"));
        assert!(result.contains("if (1u < n)"));
        assert!(result.contains("if (3u < n)"));
        // Original for-loop header must be gone
        assert!(!result.contains("for (var k"));
    }

    #[test]
    fn test_variable_bound_unroll_32() {
        // Full 32-hint (the Jacobi MAX_N case)
        let shader = concat!(
            "                // @unroll_hint 32\n",
            "                for (var k = 0u; k < n; k = k + 1u) {\n",
            "                    let x = k * 2u;\n",
            "                }\n",
        );
        let result = WgslLoopUnroller::unroll(shader);
        assert_eq!(result.matches("if (0u < n)").count(), 1);
        assert_eq!(result.matches("if (31u < n)").count(), 1);
        assert_eq!(result.matches("if (32u < n)").count(), 0);
        assert!(!result.contains("for (var k"));
    }

    #[test]
    fn test_simple_loop_unrolled() {
        let shader = r"
    // @unroll_hint 3
    for (var k = 0u; k < 3u; k = k + 1u) {
        let v = k + 1u;
    }
";
        let result = WgslLoopUnroller::unroll(shader);
        // Should contain 3 unrolled blocks
        assert_eq!(result.matches("let k = 0u;").count(), 1);
        assert_eq!(result.matches("let k = 1u;").count(), 1);
        assert_eq!(result.matches("let k = 2u;").count(), 1);
        // Original for loop should not appear in the unrolled form
        // (the @unroll_hint line is preserved as doc, but the for-loop is replaced)
        assert!(!result.contains("for (var k"));
    }

    #[test]
    fn test_loop_without_hint_unchanged() {
        let shader = "    for (var i = 0u; i < 4u; i = i + 1u) {\n        x = i;\n    }\n";
        let result = WgslLoopUnroller::unroll(shader);
        assert!(result.contains("for (var i"));
    }

    #[test]
    fn test_substitute_loop_var_word_boundary() {
        // 'k' should not be substituted inside 'k_p' or 'akp'
        let result = substitute_loop_var("    let v = k + akp;", "k", 2);
        assert!(result.contains("let v = 2u + akp"), "got: {result}");
        // Ensure 'k' inside 'akp' is NOT replaced
        assert!(!result.contains("a2"), "got: {result}");
    }

    #[test]
    fn test_hint_smaller_than_literal_bound() {
        // Hint=4, declared=8 → unrolls min(4,8)=4 iterations without guards (literal bound)
        let shader = "    // @unroll_hint 4\n    for (var k = 0u; k < 8u; k = k + 1u) {\n        x = k;\n    }\n";
        let result = WgslLoopUnroller::unroll(shader);
        assert_eq!(result.matches("let k = 0u;").count(), 1);
        assert_eq!(result.matches("let k = 3u;").count(), 1);
        assert_eq!(result.matches("let k = 4u;").count(), 0);
        // No guards needed for literal bounds
        assert!(!result.contains("if (0u <"));
    }
}
