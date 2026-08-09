// SPDX-License-Identifier: AGPL-3.0-or-later
//! WGSL device limits for scientific computing

// ============================================================================
// Capability-Based Constants
// ============================================================================

/// Science-grade max storage buffer binding size (1 GiB).
/// Raised from 512 MiB to support 32⁴+ SU(3) lattices where a single link
/// buffer is ~604 MB. Both NVIDIA (maxStorageBufferRange=4 GB) and AMD
/// (maxStorageBufferRange=4 GB) support this on strandGate hardware.
pub const SCIENCE_MAX_STORAGE_BUFFER_BINDING_SIZE: u32 = 1024 * 1024 * 1024;

/// Science-grade max buffer size (2 GiB).
/// Raised from 1 GiB. Covers 32⁴ SU(3) (largest single buffer ~604 MB)
/// with headroom for 48⁴ in the future.
pub const SCIENCE_MAX_BUFFER_SIZE: u64 = 2 * 1024 * 1024 * 1024;

/// High-capacity max storage buffer binding size (1 GiB).
pub const HIGH_CAPACITY_MAX_STORAGE_BUFFER_BINDING_SIZE: u32 = 1 << 30;

/// High-capacity max buffer size (2 GiB).
pub const HIGH_CAPACITY_MAX_BUFFER_SIZE: u64 = 1 << 31;

/// Science-grade limits — 512 MiB binding, 1 GiB buffer, 12 storage buffers.
/// Validated by hotSpring nuclear EOS study (169/169 acceptance checks).
#[must_use]
pub fn science_limits() -> wgpu::Limits {
    wgpu::Limits {
        max_storage_buffer_binding_size: SCIENCE_MAX_STORAGE_BUFFER_BINDING_SIZE,
        max_buffer_size: SCIENCE_MAX_BUFFER_SIZE,
        max_storage_buffers_per_shader_stage: 12,
        ..wgpu::Limits::default()
    }
}

/// High-capacity limits — 1GB binding, 2GB buffer.
#[must_use]
pub fn high_capacity_limits() -> wgpu::Limits {
    wgpu::Limits {
        max_storage_buffer_binding_size: HIGH_CAPACITY_MAX_STORAGE_BUFFER_BINDING_SIZE,
        max_buffer_size: HIGH_CAPACITY_MAX_BUFFER_SIZE,
        ..wgpu::Limits::default()
    }
}
