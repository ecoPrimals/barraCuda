// SPDX-License-Identifier: AGPL-3.0-or-later
//! WGSL Shader Infrastructure
//!
//! This module provides:
//! - **3-tier precision model**: f32 / f64 / df64 (fp48) — aligned with coralReef's `Fp64Strategy`
//! - **Driver-aware shader preparation**: polyfill injection, ILP optimization
//! - **CPU implementations**: Same algorithms via `num-traits` for CPU fallback
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
