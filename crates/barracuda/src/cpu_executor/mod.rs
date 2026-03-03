//! CPU Executor - Native Rust Implementation with SIMD
//!
//! **Philosophy**: Always available, well-optimized fallback
//!
//! This module provides CPU execution for all operations:
//! - Native Rust implementations
//! - SIMD optimizations (AVX2, NEON)
//! - Rayon parallel execution
//! - Zero unsafe (leverages std library)
//!
//! **Deep Debt Principles**:
//! - ✅ Always available (no hardware requirements)
//! - ✅ Well-optimized (SIMD + parallel)
//! - ✅ Safe Rust (zero unsafe)
//! - ✅ Clear implementations (readable code)

mod executor;
mod ops;
mod storage;
#[cfg(test)]
mod tests;

pub use executor::CpuExecutor;
