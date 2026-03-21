// SPDX-License-Identifier: AGPL-3.0-or-later
//! `TensorSession` — internal types: op enum, params structs, matmul tier

use crate::device::capabilities::DeviceCapabilities;
use bytemuck::{Pod, Zeroable};
use wgpu::DeviceType;

// ─── MatMul tier selection ────────────────────────────────────────────────────

const MATMUL_SMALL_THRESHOLD: usize = 32;
const MATMUL_GPU_EVOLVED_THRESHOLD: usize = 256;

static MATMUL_SMALL_THRESHOLD_RESOLVED: std::sync::LazyLock<usize> =
    std::sync::LazyLock::new(|| {
        std::env::var("BARRACUDA_MATMUL_SMALL_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(MATMUL_SMALL_THRESHOLD)
    });

static MATMUL_GPU_THRESHOLD_RESOLVED: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
    std::env::var("BARRACUDA_MATMUL_GPU_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MATMUL_GPU_EVOLVED_THRESHOLD)
});

/// Tiered matmul shader selection — same logic as `ops::MatMul::select_tier`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MatMulTier {
    Naive,
    Tiled16,
    CpuTiled32,
    GpuEvolved32,
}

impl MatMulTier {
    pub(super) fn select(caps: &DeviceCapabilities, m: usize, n: usize) -> Self {
        let max_dim = m.max(n);
        if max_dim < *MATMUL_SMALL_THRESHOLD_RESOLVED {
            return Self::Naive;
        }
        if caps.device_type == DeviceType::Cpu {
            return Self::CpuTiled32;
        }
        if max_dim >= *MATMUL_GPU_THRESHOLD_RESOLVED {
            Self::GpuEvolved32
        } else {
            Self::Tiled16
        }
    }

    pub(super) fn dispatch(self, m: u32, n: u32) -> (u32, u32) {
        match self {
            Self::Naive => (m.div_ceil(8), n.div_ceil(8)),
            Self::Tiled16 => (m.div_ceil(16), n.div_ceil(16)),
            Self::CpuTiled32 => (m.div_ceil(32), n.div_ceil(32)),
            Self::GpuEvolved32 => (m.div_ceil(32), n.div_ceil(32)),
        }
    }
}

// ─── Params structs (GPU-side uniforms) ──────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(super) struct MatMulParams {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub _padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(super) struct LayerNormParams {
    pub size: u32,
    pub feature_size: u32,
    pub epsilon: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(super) struct AttentionParams {
    pub batch_size: u32,
    pub num_heads: u32,
    pub q_seq_len: u32,
    pub kv_seq_len: u32,
    pub head_dim: u32,
    pub _padding: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(super) struct HeadReshapeParams {
    pub batch_size: u32,
    pub seq_len: u32,
    pub num_heads: u32,
    pub head_dim: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(super) struct ScaleParams {
    pub scalar: f32,
}

// ─── Operation enum ───────────────────────────────────────────────────────────

#[derive(Debug)]
pub(super) enum SessionOp {
    // ── Elementwise ──────────────────────────────────────────────────────────
    Add {
        input_a: usize,
        input_b: usize,
        output: usize,
    },
    Mul {
        input_a: usize,
        input_b: usize,
        output: usize,
    },
    Fma {
        input_a: usize,
        input_b: usize,
        input_c: usize,
        output: usize,
    },
    Scale {
        input: usize,
        scalar: f32,
        output: usize,
    },

    // ── Linear algebra ───────────────────────────────────────────────────────
    MatMul {
        input_a: usize,
        input_b: usize,
        output: usize,
        m: u32,
        k: u32,
        n: u32,
        tier: MatMulTier,
    },

    // ── Activations ──────────────────────────────────────────────────────────
    ReLU {
        input: usize,
        output: usize,
    },
    Gelu {
        input: usize,
        output: usize,
    },
    Softmax {
        input: usize,
        output: usize,
    },

    // ── Normalisation ─────────────────────────────────────────────────────────
    LayerNorm {
        input: usize,
        output: usize,
        feature_size: u32,
    },

    // ── Attention ─────────────────────────────────────────────────────────────
    Attention {
        q: usize,
        k: usize,
        v: usize,
        output: usize,
        batch_size: u32,
        num_heads: u32,
        seq_len: u32,
        head_dim: u32,
    },

    // ── Head reshape ──────────────────────────────────────────────────────────
    HeadSplit {
        input: usize,
        output: usize,
        batch_size: u32,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
    },
    HeadConcat {
        input: usize,
        output: usize,
        batch_size: u32,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
    },
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]

    use super::*;
    use crate::device::capabilities::DeviceCapabilities;
    use crate::device::vendor::VENDOR_NVIDIA;
    use wgpu::Backend;

    fn gpu_caps() -> DeviceCapabilities {
        DeviceCapabilities {
            device_name: "Test GPU".to_string(),
            device_type: wgpu::DeviceType::DiscreteGpu,
            max_buffer_size: 1024 * 1024 * 1024,
            max_workgroup_size: (256, 256, 64),
            max_compute_workgroups: (65_535, 65_535, 65_535),
            max_compute_invocations_per_workgroup: 256,
            max_storage_buffers_per_shader_stage: 8,
            max_uniform_buffers_per_shader_stage: 12,
            max_bind_groups: 4,
            backend: Backend::Vulkan,
            vendor: VENDOR_NVIDIA,
            gpu_dispatch_threshold_override: None,
            subgroup_min_size: 32,
            subgroup_max_size: 32,
            f64_shaders: true,
            f64_shared_memory: false,
            f64_capabilities: None,
        }
    }

    fn cpu_caps() -> DeviceCapabilities {
        DeviceCapabilities {
            device_name: "Test CPU".to_string(),
            device_type: wgpu::DeviceType::Cpu,
            max_buffer_size: 1024 * 1024 * 1024,
            max_workgroup_size: (256, 256, 64),
            max_compute_workgroups: (65_535, 65_535, 65_535),
            max_compute_invocations_per_workgroup: 256,
            max_storage_buffers_per_shader_stage: 8,
            max_uniform_buffers_per_shader_stage: 12,
            max_bind_groups: 4,
            backend: Backend::Vulkan,
            vendor: VENDOR_NVIDIA,
            gpu_dispatch_threshold_override: None,
            subgroup_min_size: 32,
            subgroup_max_size: 32,
            f64_shaders: true,
            f64_shared_memory: false,
            f64_capabilities: None,
        }
    }

    #[test]
    fn matmul_tier_small_dims_naive() {
        let caps = gpu_caps();
        let tier = MatMulTier::select(&caps, 16, 16);
        assert_eq!(tier, MatMulTier::Naive);
    }

    #[test]
    fn matmul_tier_boundary_31_naive() {
        let caps = gpu_caps();
        let tier = MatMulTier::select(&caps, 31, 31);
        assert_eq!(tier, MatMulTier::Naive);
    }

    #[test]
    fn matmul_tier_boundary_32_tiled16() {
        let caps = gpu_caps();
        let tier = MatMulTier::select(&caps, 32, 32);
        assert_eq!(tier, MatMulTier::Tiled16);
    }

    #[test]
    fn matmul_tier_large_gpu_evolved() {
        let caps = gpu_caps();
        let tier = MatMulTier::select(&caps, 256, 256);
        assert_eq!(tier, MatMulTier::GpuEvolved32);
    }

    #[test]
    fn matmul_tier_cpu_uses_cpu_tiled32() {
        let caps = cpu_caps();
        let tier = MatMulTier::select(&caps, 64, 64);
        assert_eq!(tier, MatMulTier::CpuTiled32);
    }

    #[test]
    fn matmul_tier_dispatch_naive() {
        let (x, y) = MatMulTier::Naive.dispatch(16, 16);
        assert_eq!(x, 2);
        assert_eq!(y, 2);
    }

    #[test]
    fn matmul_tier_dispatch_tiled16() {
        let (x, y) = MatMulTier::Tiled16.dispatch(32, 48);
        assert_eq!(x, 2);
        assert_eq!(y, 3);
    }

    #[test]
    fn matmul_tier_dispatch_zero_dims() {
        let (x, y) = MatMulTier::Naive.dispatch(0, 0);
        assert_eq!(x, 0);
        assert_eq!(y, 0);
    }
}
