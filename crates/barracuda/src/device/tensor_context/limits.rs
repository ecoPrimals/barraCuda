// SPDX-License-Identifier: AGPL-3.0-or-later
//! WGSL device limits for scientific computing

// ============================================================================
// Desired capability targets (negotiated down to hardware at device creation)
// ============================================================================

/// Science-grade max storage buffer binding size (1 GiB).
/// Raised from 512 MiB to support 32⁴+ SU(3) lattices where a single link
/// buffer is ~604 MB. Both NVIDIA (maxStorageBufferRange=4 GB) and AMD
/// (maxStorageBufferRange=4 GB) support this on strandGate hardware.
pub const DESIRED_SCIENCE_MAX_STORAGE_BUFFER_BINDING_SIZE: u32 = 1024 * 1024 * 1024;

/// Science-grade max buffer size (2 GiB - 1).
/// Set to i32::MAX to satisfy AMD RADV which reports exactly 2^31-1 as its
/// hardware maximum. Covers 32⁴ SU(3) (largest single buffer ~604 MB)
/// with headroom for 48⁴ in the future.
pub const DESIRED_SCIENCE_MAX_BUFFER_SIZE: u64 = (2 * 1024 * 1024 * 1024) - 1;

/// Science-grade max storage buffers per shader stage.
pub const DESIRED_SCIENCE_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE: u32 = 12;

/// High-capacity max storage buffer binding size (1 GiB).
pub const DESIRED_HIGH_CAPACITY_MAX_STORAGE_BUFFER_BINDING_SIZE: u32 = 1 << 30;

/// High-capacity max buffer size (2 GiB - 1).
pub const DESIRED_HIGH_CAPACITY_MAX_BUFFER_SIZE: u64 = (1u64 << 31) - 1;

/// High-capacity max storage buffers per shader stage (wgpu default).
pub const DESIRED_HIGH_CAPACITY_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE: u32 = 8;

// ============================================================================
// Negotiated limits
// ============================================================================

/// Buffer-related limits after hardware negotiation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NegotiatedLimits {
    /// Negotiated `max_buffer_size`.
    pub max_buffer: u64,
    /// Negotiated `max_storage_buffer_binding_size`.
    pub max_binding: u32,
    /// Negotiated `max_storage_buffers_per_shader_stage`.
    pub max_storage_buffers: u32,
}

/// Negotiate desired buffer limits against an adapter's hardware caps.
///
/// Each field is `min(hardware_limit, desired_limit)`, guaranteeing the
/// result fits within what the adapter can provide.
#[must_use]
pub fn negotiate_buffer_limits(
    adapter: &wgpu::Adapter,
    desired_max_buffer: u64,
    desired_max_binding: u32,
    desired_max_storage_buffers: u32,
) -> NegotiatedLimits {
    negotiate_buffer_limits_from_hardware(
        &adapter.limits(),
        desired_max_buffer,
        desired_max_binding,
        desired_max_storage_buffers,
    )
}

#[must_use]
fn negotiate_buffer_limits_from_hardware(
    hardware: &wgpu::Limits,
    desired_max_buffer: u64,
    desired_max_binding: u32,
    desired_max_storage_buffers: u32,
) -> NegotiatedLimits {
    NegotiatedLimits {
        max_buffer: hardware.max_buffer_size.min(desired_max_buffer),
        max_binding: hardware
            .max_storage_buffer_binding_size
            .min(desired_max_binding),
        max_storage_buffers: hardware
            .max_storage_buffers_per_shader_stage
            .min(desired_max_storage_buffers),
    }
}

fn limits_from_negotiated(negotiated: NegotiatedLimits) -> wgpu::Limits {
    wgpu::Limits {
        max_storage_buffer_binding_size: negotiated.max_binding,
        max_buffer_size: negotiated.max_buffer,
        max_storage_buffers_per_shader_stage: negotiated.max_storage_buffers,
        ..wgpu::Limits::default()
    }
}

// ============================================================================
// Static limit profiles (backward compat for tests without adapters)
// ============================================================================

/// Science-grade limits — 1 GiB binding, 2 GiB - 1 buffer, 12 storage buffers.
/// Validated by hotSpring nuclear EOS study (169/169 acceptance checks).
#[must_use]
pub fn science_limits() -> wgpu::Limits {
    wgpu::Limits {
        max_storage_buffer_binding_size: DESIRED_SCIENCE_MAX_STORAGE_BUFFER_BINDING_SIZE,
        max_buffer_size: DESIRED_SCIENCE_MAX_BUFFER_SIZE,
        max_storage_buffers_per_shader_stage: DESIRED_SCIENCE_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
        ..wgpu::Limits::default()
    }
}

/// Science-grade limits negotiated against adapter hardware caps.
#[must_use]
pub fn science_limits_from_adapter(adapter: &wgpu::Adapter) -> wgpu::Limits {
    let negotiated = negotiate_buffer_limits(
        adapter,
        DESIRED_SCIENCE_MAX_BUFFER_SIZE,
        DESIRED_SCIENCE_MAX_STORAGE_BUFFER_BINDING_SIZE,
        DESIRED_SCIENCE_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
    );
    limits_from_negotiated(negotiated)
}

/// High-capacity limits — 1 GiB binding, 2 GiB - 1 buffer.
#[must_use]
pub fn high_capacity_limits() -> wgpu::Limits {
    wgpu::Limits {
        max_storage_buffer_binding_size: DESIRED_HIGH_CAPACITY_MAX_STORAGE_BUFFER_BINDING_SIZE,
        max_buffer_size: DESIRED_HIGH_CAPACITY_MAX_BUFFER_SIZE,
        max_storage_buffers_per_shader_stage: DESIRED_HIGH_CAPACITY_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
        ..wgpu::Limits::default()
    }
}

/// High-capacity limits negotiated against adapter hardware caps.
#[must_use]
pub fn high_capacity_limits_from_adapter(adapter: &wgpu::Adapter) -> wgpu::Limits {
    let negotiated = negotiate_buffer_limits(
        adapter,
        DESIRED_HIGH_CAPACITY_MAX_BUFFER_SIZE,
        DESIRED_HIGH_CAPACITY_MAX_STORAGE_BUFFER_BINDING_SIZE,
        DESIRED_HIGH_CAPACITY_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
    );
    limits_from_negotiated(negotiated)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hardware_limits(max_buffer: u64, max_binding: u32, max_storage_buffers: u32) -> wgpu::Limits {
        wgpu::Limits {
            max_buffer_size: max_buffer,
            max_storage_buffer_binding_size: max_binding,
            max_storage_buffers_per_shader_stage: max_storage_buffers,
            ..wgpu::Limits::default()
        }
    }

    #[test]
    fn negotiate_clamps_to_smaller_hardware_limits() {
        let hardware = hardware_limits(128 * 1024 * 1024, 64 * 1024 * 1024, 4);
        let negotiated = negotiate_buffer_limits_from_hardware(
            &hardware,
            DESIRED_SCIENCE_MAX_BUFFER_SIZE,
            DESIRED_SCIENCE_MAX_STORAGE_BUFFER_BINDING_SIZE,
            DESIRED_SCIENCE_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
        );
        assert_eq!(negotiated.max_buffer, 128 * 1024 * 1024);
        assert_eq!(negotiated.max_binding, 64 * 1024 * 1024);
        assert_eq!(negotiated.max_storage_buffers, 4);
    }

    #[test]
    fn negotiate_preserves_desired_when_hardware_is_larger() {
        let hardware = hardware_limits(u64::MAX, u32::MAX, u32::MAX);
        let negotiated = negotiate_buffer_limits_from_hardware(
            &hardware,
            DESIRED_SCIENCE_MAX_BUFFER_SIZE,
            DESIRED_SCIENCE_MAX_STORAGE_BUFFER_BINDING_SIZE,
            DESIRED_SCIENCE_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
        );
        assert_eq!(negotiated.max_buffer, DESIRED_SCIENCE_MAX_BUFFER_SIZE);
        assert_eq!(
            negotiated.max_binding,
            DESIRED_SCIENCE_MAX_STORAGE_BUFFER_BINDING_SIZE
        );
        assert_eq!(
            negotiated.max_storage_buffers,
            DESIRED_SCIENCE_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE
        );
    }

    #[test]
    fn negotiate_zero_hardware_limits() {
        let hardware = hardware_limits(0, 0, 0);
        let negotiated = negotiate_buffer_limits_from_hardware(
            &hardware,
            DESIRED_HIGH_CAPACITY_MAX_BUFFER_SIZE,
            DESIRED_HIGH_CAPACITY_MAX_STORAGE_BUFFER_BINDING_SIZE,
            DESIRED_HIGH_CAPACITY_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
        );
        assert_eq!(negotiated.max_buffer, 0);
        assert_eq!(negotiated.max_binding, 0);
        assert_eq!(negotiated.max_storage_buffers, 0);
    }

    #[test]
    fn science_limits_from_adapter_matches_static_when_hardware_exceeds_desired() {
        let hardware = hardware_limits(u64::MAX, u32::MAX, u32::MAX);
        let negotiated = negotiate_buffer_limits_from_hardware(
            &hardware,
            DESIRED_SCIENCE_MAX_BUFFER_SIZE,
            DESIRED_SCIENCE_MAX_STORAGE_BUFFER_BINDING_SIZE,
            DESIRED_SCIENCE_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
        );
        let static_limits = science_limits();
        assert_eq!(
            negotiated.max_binding,
            static_limits.max_storage_buffer_binding_size
        );
        assert_eq!(negotiated.max_buffer, static_limits.max_buffer_size);
        assert_eq!(
            negotiated.max_storage_buffers,
            static_limits.max_storage_buffers_per_shader_stage
        );
    }

    #[test]
    fn high_capacity_static_limits_match_desired_constants() {
        let limits = high_capacity_limits();
        assert_eq!(
            limits.max_storage_buffer_binding_size,
            DESIRED_HIGH_CAPACITY_MAX_STORAGE_BUFFER_BINDING_SIZE
        );
        assert_eq!(limits.max_buffer_size, DESIRED_HIGH_CAPACITY_MAX_BUFFER_SIZE);
    }
}
