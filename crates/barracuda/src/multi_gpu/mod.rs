// SPDX-License-Identifier: AGPL-3.0-only
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

/// Bytes per gibibyte for VRAM estimates.
const BYTES_PER_GIB: u64 = 1024 * 1024 * 1024;

/// Conservative GFLOPS estimate when runtime probing is unavailable.
///
/// Values are lower bounds by vendor and device class. The scheduler always
/// prefers runtime-probed metrics when available; these exist only so device
/// scoring never divides by zero.
pub(crate) fn estimate_gflops(vendor: GpuVendor, device_type: wgpu::DeviceType) -> f64 {
    match (vendor, device_type) {
        (GpuVendor::Software, _) => 10.0,
        (GpuVendor::Nvidia, wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::Other) => 5_000.0,
        (GpuVendor::Amd, wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::Other) => 4_000.0,
        (_, wgpu::DeviceType::DiscreteGpu) => 1_000.0,
        (_, wgpu::DeviceType::IntegratedGpu) => 200.0,
        (_, wgpu::DeviceType::Cpu) => 50.0,
        _ => 100.0,
    }
}

/// Conservative VRAM estimate when runtime probing is unavailable.
pub(crate) fn estimate_vram_bytes(vendor: GpuVendor, device_type: wgpu::DeviceType) -> u64 {
    match (vendor, device_type) {
        (GpuVendor::Software, _) | (_, wgpu::DeviceType::Cpu) => 0,
        (GpuVendor::Nvidia, wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::Other) => {
            12 * BYTES_PER_GIB
        }
        (GpuVendor::Amd, wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::Other) => {
            16 * BYTES_PER_GIB
        }
        (_, wgpu::DeviceType::DiscreteGpu) => 8 * BYTES_PER_GIB,
        (_, wgpu::DeviceType::IntegratedGpu) => 2 * BYTES_PER_GIB,
        _ => 4 * BYTES_PER_GIB,
    }
}
