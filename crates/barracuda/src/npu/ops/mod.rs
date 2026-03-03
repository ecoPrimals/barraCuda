// SPDX-License-Identifier: AGPL-3.0-or-later
//! NPU Operations Module - Extended NPU Coverage
//!
//! **Mission**: Provide NPU-accelerated implementations of common operations
//!
//! **Philosophy**: Transparent acceleration - users call standard ops,
//! backend auto-selects NPU when beneficial based on:
//! - Workload type (ML, sparse ops preferred)
//! - Data sparsity (>50% benefits NPU)
//! - Performance priority (energy → NPU)
//!
//! **Status**: Phase 5a - Core ML operations
//!
//! **Deep Debt**:
//! - Pure Rust via akida-driver
//! - Runtime selection (no hardcoding)
//! - Validated performance (benchmark-driven)

pub mod gelu;
pub mod layer_norm;
pub mod matmul;
pub mod relu;
pub mod softmax; // NEW!
