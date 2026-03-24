// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shader Warmup System - "Mise en Place" for GPU Computing
//!
//! Pre-compiles commonly used shader patterns before task execution.
//! barraCuda learns which operations are used and warms the cache proactively.
//!
//! ## Why This Matters
//!
//! First shader compilation: ~5000 μs (cold)
//! Subsequent calls: ~300 μs (warm cache)
//!
//! By warming up before a workload starts, we eliminate cold-start latency
//! from the critical path.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  barraCuda Shader Warmup                                         │
//! │                                                                  │
//! │  1. Receive workload description                                 │
//! │  2. Analyze required operations (add, mul, matmul, reduce, etc.) │
//! │  3. Warm up shaders for ALL detected ops on ALL available GPUs   │
//! │  4. THEN start actual computation                                │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use crate::device::WgpuDevice;
use crate::device::pipeline_cache::{BindGroupLayoutSignature, GLOBAL_CACHE};
use crate::error::Result;
use std::borrow::Cow;
use std::sync::Arc;
use std::time::Instant;

/// Operations that can be pre-warmed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WarmupOp {
    /// Element-wise addition (2 inputs, 1 output)
    Add,
    /// Element-wise multiplication
    Mul,
    /// Fused multiply-add (a*b + c)
    Fma,
    /// Scalar multiplication
    Scale,
    /// Matrix multiplication
    Matmul,
    /// Reduction (sum, mean, max, min)
    Reduce,
    /// Softmax
    Softmax,
    /// `ReLU` activation
    ReLU,
    /// Generic binary op
    BinaryOp,
    /// Generic unary op
    UnaryOp,
    /// Fused mean+variance (Welford, f64)
    MeanVarianceF64,
    /// Fused Pearson correlation (5-accumulator, f64)
    CorrelationF64,
    /// Sum reduction (f64)
    SumReduceF64,
}

impl WarmupOp {
    /// Get shader source for this operation
    fn shader_source(&self, workgroup_size: u32) -> Cow<'static, str> {
        match self {
            Self::Add => Cow::Owned(format!(
                r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= arrayLength(&out)) {{ return; }}
    out[idx] = a[idx] + b[idx];
}}
"
            )),

            Self::Mul => Cow::Owned(format!(
                r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= arrayLength(&out)) {{ return; }}
    out[idx] = a[idx] * b[idx];
}}
"
            )),

            Self::Fma => Cow::Owned(format!(
                r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> c: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= arrayLength(&out)) {{ return; }}
    out[idx] = fma(a[idx], b[idx], c[idx]);
}}
"
            )),

            Self::Scale => Cow::Owned(format!(
                r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> scale: f32;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= arrayLength(&out)) {{ return; }}
    out[idx] = a[idx] * scale;
}}
"
            )),

            Self::ReLU => Cow::Owned(format!(
                r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= arrayLength(&out)) {{ return; }}
    out[idx] = max(a[idx], 0.0);
}}
"
            )),

            Self::Matmul | Self::Reduce | Self::Softmax => Cow::Owned(format!(
                r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= arrayLength(&out)) {{ return; }}
    out[idx] = a[idx] + b[idx];
}}
"
            )),

            Self::BinaryOp => Cow::Owned(format!(
                r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= arrayLength(&out)) {{ return; }}
    out[idx] = a[idx] + b[idx];
}}
"
            )),

            Self::UnaryOp => Cow::Owned(format!(
                r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= arrayLength(&out)) {{ return; }}
    out[idx] = a[idx];
}}
"
            )),

            Self::MeanVarianceF64 => {
                Cow::Borrowed(include_str!("../shaders/reduce/mean_variance_f64.wgsl"))
            }
            Self::CorrelationF64 => {
                Cow::Borrowed(include_str!("../shaders/stats/correlation_full_f64.wgsl"))
            }
            Self::SumReduceF64 => {
                Cow::Borrowed(include_str!("../shaders/reduce/sum_reduce_f64.wgsl"))
            }
        }
    }

    /// Get bind group layout signature for this operation
    fn layout_signature(&self) -> BindGroupLayoutSignature {
        match self {
            Self::Add | Self::Mul | Self::BinaryOp => {
                BindGroupLayoutSignature::elementwise_binary()
            }
            Self::ReLU | Self::UnaryOp => BindGroupLayoutSignature::elementwise_unary(),
            Self::Scale => BindGroupLayoutSignature {
                read_only_buffers: 1,
                read_write_buffers: 1,
                uniform_buffers: 1,
            },
            Self::Fma => BindGroupLayoutSignature {
                read_only_buffers: 3,
                read_write_buffers: 1,
                uniform_buffers: 0,
            },
            Self::Matmul | Self::Reduce | Self::Softmax => BindGroupLayoutSignature::matmul(),
            Self::MeanVarianceF64 | Self::SumReduceF64 => BindGroupLayoutSignature::reduction(),
            Self::CorrelationF64 => BindGroupLayoutSignature {
                read_only_buffers: 2,
                read_write_buffers: 1,
                uniform_buffers: 1,
            },
        }
    }

    /// All standard operations for full warmup
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[
            Self::Add,
            Self::Mul,
            Self::Fma,
            Self::Scale,
            Self::ReLU,
            Self::BinaryOp,
            Self::UnaryOp,
        ]
    }

    /// ML inference operations
    #[must_use]
    pub fn ml_inference() -> &'static [Self] {
        &[
            Self::Add,
            Self::Mul,
            Self::Matmul,
            Self::ReLU,
            Self::Softmax,
        ]
    }

    /// Scientific computing operations (includes f64 fused reductions)
    #[must_use]
    pub fn scientific() -> &'static [Self] {
        &[
            Self::Add,
            Self::Mul,
            Self::Fma,
            Self::Scale,
            Self::Reduce,
            Self::MeanVarianceF64,
            Self::CorrelationF64,
            Self::SumReduceF64,
        ]
    }
}

/// Warmup configuration
#[derive(Debug, Clone)]
pub struct WarmupConfig {
    /// Operations to warm up
    pub ops: Vec<WarmupOp>,
    /// Workgroup sizes to compile (multiple for auto-tuning)
    pub workgroup_sizes: Vec<u32>,
    /// Whether to report timing
    pub verbose: bool,
}

impl Default for WarmupConfig {
    fn default() -> Self {
        Self {
            ops: WarmupOp::all().to_vec(),
            workgroup_sizes: vec![64, 128, 256], // Cover common optimal sizes
            verbose: false,
        }
    }
}

impl WarmupConfig {
    /// Minimal warmup (just add/mul with default workgroup)
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            ops: vec![WarmupOp::Add, WarmupOp::Mul],
            workgroup_sizes: vec![256],
            verbose: false,
        }
    }

    /// Full warmup (all ops, all workgroup sizes)
    #[must_use]
    pub fn full() -> Self {
        Self {
            ops: WarmupOp::all().to_vec(),
            workgroup_sizes: vec![32, 64, 128, 256],
            verbose: true,
        }
    }

    /// ML inference workload
    #[must_use]
    pub fn ml() -> Self {
        Self {
            ops: WarmupOp::ml_inference().to_vec(),
            workgroup_sizes: vec![64, 128, 256],
            verbose: false,
        }
    }

    /// Scientific computing workload
    #[must_use]
    pub fn scientific() -> Self {
        Self {
            ops: WarmupOp::scientific().to_vec(),
            workgroup_sizes: vec![64, 128, 256],
            verbose: false,
        }
    }
}

/// Warmup result for a single device
#[derive(Debug, Clone)]
pub struct WarmupResult {
    /// Device name
    pub device_name: String,
    /// Number of shaders compiled
    pub shaders_compiled: usize,
    /// Number of pipelines created
    pub pipelines_created: usize,
    /// Total warmup time
    pub warmup_time_ms: f64,
}

/// Warm up shader cache for a single device
///
/// Call this before starting a compute-intensive workload.
/// Compiles all specified shaders so they're ready when needed.
///
/// # Errors
///
/// Returns [`Err`] if shader compilation or pipeline creation fails (e.g.
/// device lost, invalid WGSL, or out of memory).
pub fn warmup_device(device: &WgpuDevice, config: &WarmupConfig) -> Result<WarmupResult> {
    let start = Instant::now();
    let adapter_info = device.adapter_info();
    let wgpu_device = device.device();

    let mut shaders = 0;
    let mut pipelines = 0;

    if config.verbose {
        tracing::info!("Warming up: {}", adapter_info.name);
    }

    for op in &config.ops {
        for &wg_size in &config.workgroup_sizes {
            let shader_source = op.shader_source(wg_size);
            let layout_sig = op.layout_signature();

            // This will compile shader and create pipeline if not cached
            let _pipeline = GLOBAL_CACHE.get_or_create_pipeline(
                wgpu_device,
                adapter_info,
                &shader_source,
                layout_sig,
                "main",
                Some(&format!("{op:?}_wg{wg_size}")),
            );

            shaders += 1;
            pipelines += 1;
        }
    }

    let elapsed = start.elapsed();

    if config.verbose {
        tracing::info!(
            ops = config.ops.len(),
            workgroup_sizes = config.workgroup_sizes.len(),
            pipelines,
            elapsed_ms = elapsed.as_secs_f64() * 1000.0,
            "warmup: compiled pipelines"
        );
    }

    Ok(WarmupResult {
        device_name: adapter_info.name.clone(),
        shaders_compiled: shaders,
        pipelines_created: pipelines,
        warmup_time_ms: elapsed.as_secs_f64() * 1000.0,
    })
}

/// Warm up all devices in a pool
///
/// Call this at application startup or before a batch job.
///
/// # Errors
///
/// Returns [`Err`] if shader compilation or pipeline creation fails for any
/// device (e.g. device lost, invalid WGSL, or out of memory).
pub fn warmup_pool(
    devices: &[Arc<WgpuDevice>],
    config: &WarmupConfig,
) -> Result<Vec<WarmupResult>> {
    let total_start = Instant::now();

    if config.verbose {
        tracing::info!("barraCuda Mise en Place — Shader Warmup starting");
    }

    let mut results = Vec::with_capacity(devices.len());

    for device in devices {
        let result = warmup_device(device, config)?;
        results.push(result);
    }

    if config.verbose {
        let total_time = total_start.elapsed();
        let total_pipelines: usize = results.iter().map(|r| r.pipelines_created).sum();
        tracing::info!(
            total_pipelines,
            gpus = devices.len(),
            elapsed_ms = total_time.as_secs_f64() * 1000.0,
            "warmup complete"
        );

        let stats = GLOBAL_CACHE.stats();
        tracing::info!(
            shaders = stats.shaders,
            layouts = stats.layouts,
            pipelines = stats.pipelines,
            "warmup cache stats"
        );
    }

    Ok(results)
}

/// Warmup workload hint for intelligent pre-compilation
#[derive(Debug, Clone)]
pub enum WarmupWorkloadHint {
    /// General purpose (warm all common ops)
    General,
    /// ML inference (matmul, activations, softmax)
    MlInference,
    /// ML training (inference + backward ops)
    MlTraining,
    /// Scientific simulation (FMA, reduction, etc.)
    Scientific,
    /// Custom list of operations
    Custom(Vec<WarmupOp>),
}

impl WarmupWorkloadHint {
    /// Convert to warmup config
    #[must_use]
    pub fn to_config(&self) -> WarmupConfig {
        match self {
            Self::General => WarmupConfig::default(),
            Self::MlInference => WarmupConfig::ml(),
            Self::MlTraining => WarmupConfig {
                ops: vec![
                    WarmupOp::Add,
                    WarmupOp::Mul,
                    WarmupOp::Fma,
                    WarmupOp::Matmul,
                    WarmupOp::ReLU,
                    WarmupOp::Softmax,
                    WarmupOp::Scale,
                    WarmupOp::Reduce,
                ],
                workgroup_sizes: vec![64, 128, 256],
                verbose: false,
            },
            Self::Scientific => WarmupConfig::scientific(),
            Self::Custom(ops) => WarmupConfig {
                ops: ops.clone(),
                workgroup_sizes: vec![64, 128, 256],
                verbose: false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_generation() {
        let shader = WarmupOp::Add.shader_source(64);
        assert!(shader.contains("@workgroup_size(64)"));
        assert!(shader.contains("a[idx] + b[idx]"));
    }

    #[test]
    fn test_config_presets() {
        let minimal = WarmupConfig::minimal();
        assert_eq!(minimal.ops.len(), 2);

        let full = WarmupConfig::full();
        assert!(full.ops.len() >= 5);
        assert!(full.workgroup_sizes.len() >= 3);
    }
}
