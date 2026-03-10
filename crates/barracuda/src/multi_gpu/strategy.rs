// SPDX-License-Identifier: AGPL-3.0-only
//! Pool implementations and device selection strategy.
//!
//! Split into focused modules:
//! - [`gpu_pool`]: Basic round-robin pool with semaphore concurrency
//! - [`multi_device_pool`]: Advanced pool with quotas and requirements-based selection

pub use super::gpu_pool::GpuPool;
pub use super::multi_device_pool::{DeviceLease, MultiDevicePool};
