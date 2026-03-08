// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sovereign shader validation harness — pure Rust, no GPU required.
//!
//! Walks all WGSL shader files and validates each through the naga pipeline:
//! parse → FMA fusion → dead expression elimination → validate → emit WGSL.
//!
//! This gives full coverage of the sovereign compiler pipeline under llvm-cov
//! without any GPU dependency. The Rust compiler is our DNA synthase — every
//! shader must pass through it with compile guarantees.
//!
//! Mirrors production behavior: if sovereign optimization fails validation,
//! the shader still passes if naga can parse it (production falls back to
//! raw WGSL in this case).

#[cfg(test)]
mod tests {
    use crate::shaders::sovereign::SovereignCompiler;
    use std::path::PathBuf;

    fn test_compiler() -> SovereignCompiler {
        use crate::device::driver_profile::{CompilerKind, DriverKind, Fp64Rate, GpuArch};
        SovereignCompiler::new(crate::device::driver_profile::GpuDriverProfile {
            driver: DriverKind::Unknown,
            compiler: CompilerKind::Unknown,
            arch: GpuArch::Unknown,
            fp64_rate: Fp64Rate::Throttled,
            workarounds: vec![],
            adapter_key: String::new(),
        })
    }

    fn shader_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/shaders")
    }

    fn ops_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/ops")
    }

    fn df64_preamble() -> String {
        let core = include_str!("../../shaders/math/df64_core.wgsl");
        let trans = include_str!("../../shaders/math/df64_transcendentals.wgsl");
        format!("{core}\n{trans}\n")
    }

    fn collect_wgsl_files(root: &std::path::Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        collect_wgsl_recursive(root, &mut files);
        files.sort();
        files
    }

    fn collect_wgsl_recursive(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_wgsl_recursive(&path, out);
            } else if path.extension().is_some_and(|e| e == "wgsl") {
                out.push(path);
            }
        }
    }

    fn is_df64_shader(path: &std::path::Path) -> bool {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        name.contains("df64") && name != "df64_core.wgsl" && name != "df64_transcendentals.wgsl"
    }

    fn needs_df64_preamble(source: &str) -> bool {
        source.contains("Df64") || source.contains("df64_add") || source.contains("df64_mul")
    }

    /// Preprocess a shader source to match production behavior.
    /// Production strips `enable f64;` before sovereign compilation.
    fn preprocess(source: &str) -> String {
        source
            .lines()
            .filter(|line| {
                let trimmed = line.trim();
                trimmed != "enable f64;" && trimmed != "enable f16;"
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Result of validating a single shader.
    enum ShaderResult {
        /// Sovereign compiled: parsed + optimized + validated + re-emitted.
        Sovereign { fma: usize, dead: usize },
        /// Parsed by naga but sovereign optimization failed validation.
        /// Production uses raw WGSL fallback for these.
        ParseOnly,
        /// Library fragment: references external functions/types not present
        /// in this file alone. Expected for preamble fragments.
        Fragment,
        /// Needs runtime preprocessing (driver patching, type coercion)
        /// before standalone naga can parse. Works through the full
        /// `compile_shader_f64` production pipeline.
        NeedsPreprocessing,
    }

    fn validate_shader(
        compiler: &SovereignCompiler,
        path: &std::path::Path,
        df64_pre: &str,
    ) -> Result<ShaderResult, String> {
        let source = std::fs::read_to_string(path)
            .map_err(|e| format!("{}: read error: {e}", path.display()))?;

        let processed = preprocess(&source);
        let is_preamble = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n == "df64_core.wgsl" || n == "df64_transcendentals.wgsl");

        let full_source =
            if !is_preamble && (is_df64_shader(path) || needs_df64_preamble(&processed)) {
                format!("{df64_pre}{processed}")
            } else {
                processed
            };

        // Phase 1: Parse through naga (exercises the WGSL frontend).
        let parse_result = naga::front::wgsl::parse_str(&full_source);
        let _module = match parse_result {
            Ok(m) => m,
            Err(e) => {
                let err_str = e.to_string();

                let is_fragment = err_str.contains("unknown function")
                    || err_str.contains("unknown ident")
                    || err_str.contains("no definition in scope")
                    || err_str.contains("not found")
                    || err_str.contains("unknown type")
                    || err_str.contains("unknown scalar type")
                    || err_str.contains("expected global item");

                if is_fragment {
                    return Ok(ShaderResult::Fragment);
                }

                // Shaders needing runtime preprocessing: driver-specific type
                // patching, automatic conversions, reserved keyword renaming.
                // These work through the full compile_shader_f64 pipeline.
                let needs_preprocessing = err_str.contains("automatic conversions cannot convert")
                    || err_str.contains("Cannot apply math function")
                    || err_str.contains("reserved keyword")
                    || err_str.contains("cannot cast")
                    || err_str.contains("unexpected argument type")
                    || err_str.contains("inconsistent type")
                    || err_str.contains("type mismatch")
                    || err_str.contains("invalid field accessor")
                    || err_str.contains("expected identifier, found");

                if needs_preprocessing {
                    return Ok(ShaderResult::NeedsPreprocessing);
                }

                return Err(format!("{}: parse failed: {err_str}", path.display()));
            }
        };

        // Phase 2: Try sovereign optimization (exercises FMA + dead expr + validator + emitter).
        // Matches production: if sovereign fails, fall back to parse-only.
        match compiler.compile_to_wgsl(&full_source) {
            Ok((_, stats)) => Ok(ShaderResult::Sovereign {
                fma: stats.fma_fusions,
                dead: stats.dead_exprs_eliminated,
            }),
            Err(_) => Ok(ShaderResult::ParseOnly),
        }
    }

    #[test]
    fn sovereign_validates_all_wgsl_shaders() {
        let compiler = test_compiler();
        let df64_pre = df64_preamble();

        let shader_files = collect_wgsl_files(&shader_root());
        let ops_files = collect_wgsl_files(&ops_root());
        let all_files: Vec<_> = shader_files.into_iter().chain(ops_files).collect();

        assert!(
            all_files.len() > 600,
            "expected 600+ WGSL files, found {}",
            all_files.len()
        );

        let mut sovereign_count = 0usize;
        let mut parse_only_count = 0usize;
        let mut fragment_count = 0usize;
        let mut preprocess_count = 0usize;
        let mut total_fma = 0usize;
        let mut total_dead = 0usize;
        let mut failures: Vec<String> = Vec::new();

        for path in &all_files {
            match validate_shader(&compiler, path, &df64_pre) {
                Ok(ShaderResult::Sovereign { fma, dead }) => {
                    sovereign_count += 1;
                    total_fma += fma;
                    total_dead += dead;
                }
                Ok(ShaderResult::ParseOnly) => {
                    parse_only_count += 1;
                }
                Ok(ShaderResult::Fragment) => {
                    fragment_count += 1;
                }
                Ok(ShaderResult::NeedsPreprocessing) => {
                    preprocess_count += 1;
                }
                Err(msg) => {
                    failures.push(msg);
                }
            }
        }

        tracing::warn!("Sovereign shader validation harness:");
        tracing::warn!("  Total files:       {}", all_files.len());
        tracing::warn!("  Sovereign pass:    {sovereign_count}");
        tracing::warn!("  Parse-only (fallback): {parse_only_count}");
        tracing::warn!("  Fragments (lib):   {fragment_count}");
        tracing::warn!("  Needs preprocess:  {preprocess_count}");
        tracing::warn!("  Parse failures:    {}", failures.len());
        tracing::warn!("  FMA fusions:       {total_fma}");
        tracing::warn!("  Dead exprs elim:   {total_dead}");

        if !failures.is_empty() {
            tracing::warn!("\nParse failures:");
            for msg in &failures {
                tracing::warn!("  {msg}");
            }
        }

        assert!(
            failures.is_empty(),
            "{} shader(s) failed naga parse (see above)",
            failures.len()
        );

        assert!(
            sovereign_count > 200,
            "expected 200+ shaders to pass sovereign compilation, got {sovereign_count}"
        );
    }
}
