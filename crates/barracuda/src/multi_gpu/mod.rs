// SPDX-License-Identifier: AGPL-3.0-or-later
//! Multi-GPU Workload Distribution
//!
//! Provides load balancing and parallel execution across multiple GPUs.
//! Works with NVIDIA and AMD via wgpu's vendor-agnostic API.
//!
//! # Features
//!
//! - **`GpuPool`**: Basic round-robin load balancing
//! - **`MultiDevicePool`**: Advanced device selection with quotas and requirements
//! - **`DeviceRequirements`**: Specify minimum VRAM, preferred vendor, etc.
//! - **`ResourceQuota` integration**: Per-task VRAM budget enforcement
//!
//! # Example
//!
//! ```ignore
//! use barracuda::multi_gpu::{MultiDevicePool, DeviceRequirements};
//! use barracuda::resource_quota::ResourceQuota;
//!
//! // Create a pool with all available GPUs
//! let pool = MultiDevicePool::new().await?;
//! println!("{}", pool.summary());
//!
//! // Acquire a device with specific requirements
//! let reqs = DeviceRequirements::new()
//!     .with_min_vram_gb(8)
//!     .prefer_nvidia();
//!
//! let lease = pool.acquire(&reqs).await?;
//! // Use lease.device() for operations
//! // Device automatically released when lease is dropped
//! ```
//!
//! # Deep Debt Compliance
//!
//! - Modern idiomatic Rust (builder patterns, no global state mutation)
//! - Zero unsafe code
//! - Capability-based device discovery
//! - Proper error handling (Result types, no panics)

mod gpu_pool;
pub mod interconnect;
mod multi_device_pool;
pub mod pipeline_dispatch;
mod strategy;
mod topology;
mod types;

#[cfg(test)]
mod tests;

// Topology (vendor, driver, workload types)
pub use topology::{GpuDriver, GpuInfo, GpuVendor, WorkloadType};

// Interconnect topology (PCIe bus links, bandwidth tiers, P2P routing)
pub use interconnect::{BandwidthTier, InterconnectTopology, Link};

// Multi-stage pipeline dispatch across substrates
pub use pipeline_dispatch::{
    FallbackPolicy, PipelineStage, ResolvedPipeline, ResolvedStage, StageResolution, StageWorkload,
    SubstratePipeline, TransferStrategy,
};

// Types and configuration
pub use types::{DeviceInfo, DeviceRequirements, WorkloadConfig};

// Pool implementations and leases
pub use strategy::{DeviceLease, GpuPool, MultiDevicePool};
