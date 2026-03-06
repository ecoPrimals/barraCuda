// SPDX-License-Identifier: AGPL-3.0-or-later
//! WGSL device limits for scientific computing

// ============================================================================
// Capability-Based Constants
// ============================================================================

/// Science-grade max storage buffer binding size (512 MiB).
pub const SCIENCE_MAX_STORAGE_BUFFER_BINDING_SIZE: u32 = 512 * 1024 * 1024;

/// Science-grade max buffer size (1 GiB).
pub const SCIENCE_MAX_BUFFER_SIZE: u64 = 1024 * 1024 * 1024;

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
