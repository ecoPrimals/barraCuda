//! Interpolation Operations
//!
//! **Deep Debt Compliant Interpolation Module**
//!
//! Scientific interpolation methods for:
//! - RBF surrogate learning (hotSpring physics integration)
//! - Surrogate-based optimization
//! - Data-driven modeling
//!
//! ## Operations
//!
//! - `rbf_kernel` - Radial basis function kernel evaluation
//! - `rbf` - Complete RBF interpolator (fit + predict)
//!
//! ## Design Principles
//!
//! - ✅ Pure WGSL (hardware-agnostic)
//! - ✅ Safe Rust (no unsafe blocks)
//! - ✅ Runtime-configured parameters
//! - ✅ Composable operations
//! - ✅ scipy.interpolate compatible

pub mod rbf;
pub mod rbf_kernel;

pub use rbf::RbfInterpolator;
pub use rbf_kernel::{RbfKernel, RbfKernelType};
