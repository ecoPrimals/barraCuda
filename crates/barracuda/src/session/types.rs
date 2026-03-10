// SPDX-License-Identifier: AGPL-3.0-only
//! `TensorSession` — internal types: op enum, params structs, matmul tier

use crate::device::capabilities::DeviceCapabilities;
use bytemuck::{Pod, Zeroable};
use wgpu::DeviceType;

// ─── MatMul tier selection ────────────────────────────────────────────────────

const MATMUL_SMALL_THRESHOLD: usize = 32;
const MATMUL_GPU_EVOLVED_THRESHOLD: usize = 256;

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
        if max_dim < MATMUL_SMALL_THRESHOLD {
            return Self::Naive;
        }
        if caps.device_type == DeviceType::Cpu {
            return Self::CpuTiled32;
        }
        if max_dim >= MATMUL_GPU_EVOLVED_THRESHOLD {
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
