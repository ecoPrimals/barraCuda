// SPDX-License-Identifier: AGPL-3.0-or-later
//! WGSL Shader Infrastructure
//!
//! This module provides:
//! - **15-tier precision continuum**: Binary→DF128 with per-tier `op_preamble` WGSL and coralReef strategy mapping
//! - **Driver-aware shader preparation**: polyfill injection, ILP optimization
//! - **CPU implementations**: Same algorithms via local `CpuFloat` trait for CPU fallback
//! - **Quantized inference shaders**: INT4/INT8 dequantization and GEMV
//!
//! # Design Philosophy
//!
//! Math is written in f64-canonical WGSL — pure math, conceptually infinite
//! precision. The compilation pipeline targets one of three hardware tiers:
//! - **f32** — consumer default, lossy downcast (coralReef: `Fp64Strategy::F32Only`)
//! - **f64** — scientific computing, native hardware (coralReef: `Fp64Strategy::Native`)
//! - **df64** — fp48 sweet spot, f32-pair emulation (coralReef: `Fp64Strategy::DoubleFloat`)
//!
//! # Usage
//!
//! ```rust,ignore
//! use barracuda::shaders::precision::ShaderTemplate;
//!
//! // Prepare f64-canonical shader for driver-aware dispatch
//! let prepared = ShaderTemplate::for_driver_auto(shader_source, needs_workaround);
//!
//! // CPU equivalent (same algorithm)
//! use barracuda::shaders::precision::cpu;
//! let mut out = vec![0.0f64; 3];
//! cpu::elementwise_add(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &mut out);
//! ```

pub mod optimizer; // WgslDependencyGraph + IlpReorderer + WgslLoopUnroller (SOVEREIGN Phase 3, live)
pub mod precision;
pub mod provenance; // Cross-spring shader evolution tracking (Write → Absorb → Lean)
pub mod quantized;
#[cfg(feature = "gpu")]
pub mod sovereign; // SovereignCompiler — naga IR optimizer + SPIR-V emission (SOVEREIGN Phase 4)

pub use optimizer::WgslOptimizer;
pub use precision::{Precision, ShaderTemplate};
pub use provenance::{ShaderCategory, ShaderRecord, SpringDomain};

/// DF64 core arithmetic library (Dekker f32-pair).
pub const DF64_CORE: &str = include_str!("math/df64_core.wgsl");
/// DF64 transcendental functions (exp, log, sin, cos, etc.).
pub const DF64_TRANSCENDENTALS: &str = include_str!("math/df64_transcendentals.wgsl");

/// Combine the DF64 core library with a domain-specific shader source.
///
/// Returns the concatenated source suitable for `LazyLock` caching.
#[must_use]
pub fn df64_source(domain_shader: &str) -> String {
    format!("{DF64_CORE}\n{domain_shader}")
}

/// Combine the DF64 core library with a domain shader, prefixed with `enable f64;`.
///
/// For Hybrid devices that need both the DF64 core and f64 enable directive.
#[must_use]
pub fn df64_f64_source(domain_shader: &str) -> String {
    format!("enable f64;\n{DF64_CORE}\n{domain_shader}")
}
