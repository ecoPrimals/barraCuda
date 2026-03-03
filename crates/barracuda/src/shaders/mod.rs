// SPDX-License-Identifier: AGPL-3.0-or-later
//! WGSL Shader Infrastructure
//!
//! This module provides:
//! - **Precision-generic shader templates**: ONE source generates f16/f32/f64 shaders
//! - **CPU implementations**: Same algorithms via `num-traits` for CPU fallback
//! - **Quantized inference shaders**: INT4/INT8 dequantization and GEMV
//!
//! # Design Philosophy
//!
//! Same math runs on CPU and GPU:
//! - GPU: WGSL shaders (generated from templates)
//! - CPU: Rust implementations (via num-traits)
//!
//! # Quantized Inference
//!
//! For LLM inference with quantized weights (GGUF/llama.cpp):
//! - `dequant_q4.wgsl`: Q4_0 dequantization (4-bit weights)
//! - `dequant_q8.wgsl`: Q8_0 dequantization (8-bit weights)
//! - `gemv_q4.wgsl`: On-the-fly Q4 matrix-vector multiply
//! - `gemv_q8.wgsl`: On-the-fly Q8 matrix-vector multiply
//!
//! # Usage
//!
//! ```rust,ignore
//! use barracuda::shaders::precision::{Precision, ShaderTemplate};
//!
//! // Generate f64 shader at runtime
//! let f64_add = ShaderTemplate::elementwise_add(Precision::F64);
//!
//! // CPU equivalent (same algorithm)
//! use barracuda::shaders::precision::cpu;
//! let mut out = vec![0.0f64; 3];
//! cpu::elementwise_add(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &mut out);
//! ```

pub mod optimizer; // WgslDependencyGraph + IlpReorderer + WgslLoopUnroller (SOVEREIGN Phase 3, live)
pub mod precision;
pub mod quantized;
#[cfg(feature = "gpu")]
pub mod sovereign; // SovereignCompiler — naga IR optimizer + SPIR-V emission (SOVEREIGN Phase 4)

pub use optimizer::WgslOptimizer;
pub use precision::{Precision, ShaderTemplate};
