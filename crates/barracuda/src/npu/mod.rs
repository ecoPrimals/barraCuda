//! NPU Backend Module for BarraCuda v2.0
//!
//! Event-driven ML execution on Akida neuromorphic processors.
//!
//! **Deep Debt Principles**:
//! - Pure Rust (using akida-driver)
//! - Runtime discovery (no hardcoded devices)
//! - Capability-based configuration
//! - Zero unsafe code
//! - Measured performance (not simulated)

pub mod constants;
pub mod event_codec;
#[cfg(feature = "npu-akida")]
pub mod ml_backend;
#[cfg(feature = "npu-akida")]
pub mod ops;

pub use constants as npu_constants;
pub use event_codec::EventCodec;
#[cfg(feature = "npu-akida")]
pub use ml_backend::NpuMlBackend;
