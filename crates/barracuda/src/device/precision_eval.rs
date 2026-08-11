// SPDX-License-Identifier: AGPL-3.0-or-later

//! Per-shader precision/throughput profiler across precision tiers.
//!
//! Compiles and benchmarks a WGSL shader at F32, F64, DF64, and F64Precise
//! on a single `WgpuDevice`, measuring compile time, dispatch throughput,
//! readback latency, and numerical accuracy (ULP error vs F64 reference).
//!
//! Upstreamed from hotSpring v0.6.32 per the deduplication handoff.

use super::WgpuDevice;
use super::precision_tier::PrecisionTier;
use std::time::Instant;
use wgpu::util::DeviceExt;

const WARMUP_REPS: usize = 3;
const MEASURE_REPS: usize = 10;

/// Result of evaluating a single precision tier.
#[derive(Debug, Clone)]
pub struct TierResult {
    pub tier: PrecisionTier,
    pub compiled: bool,
    pub compile_us: f64,
    pub dispatch_us: f64,
    pub readback_us: f64,
    pub output: Vec<f64>,
    pub max_ulp_error: f64,
}

/// Result of evaluating a shader across all applicable tiers.
#[derive(Debug, Clone)]
pub struct ShaderEvalResult {
    pub shader_name: String,
    pub tiers: Vec<TierResult>,
}

impl ShaderEvalResult {
    /// Find the best tier (lowest ULP error among those that compiled).
    #[must_use]
    pub fn best_tier(&self) -> Option<&TierResult> {
        self.tiers
            .iter()
            .filter(|t| t.compiled && t.max_ulp_error.is_finite())
            .min_by(|a, b| a.max_ulp_error.partial_cmp(&b.max_ulp_error).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Find the fastest tier among those that compiled.
    #[must_use]
    pub fn fastest_tier(&self) -> Option<&TierResult> {
        self.tiers
            .iter()
            .filter(|t| t.compiled && t.dispatch_us > 0.0)
            .min_by(|a, b| a.dispatch_us.partial_cmp(&b.dispatch_us).unwrap_or(std::cmp::Ordering::Equal))
    }
}

/// Per-shader precision/throughput profiler.
///
/// Evaluates shader compilation, dispatch, and numerical accuracy across
/// precision tiers on a given GPU device.
pub struct PrecisionEval<'a> {
    device: &'a WgpuDevice,
}

impl<'a> PrecisionEval<'a> {
    #[must_use]
    pub fn new(device: &'a WgpuDevice) -> Self {
        Self { device }
    }

    /// Evaluate a shader at F64, F64Precise, DF64, and F32 tiers.
    ///
    /// The shader must have bindings: `@group(0) @binding(0)` input (storage, read),
    /// `@group(0) @binding(1)` output (storage, read_write).
    pub fn eval_shader(
        &self,
        name: &str,
        f64_source: &str,
        input: &[f64],
        n_elements: usize,
        workgroups: u32,
    ) -> ShaderEvalResult {
        let tiers = [
            PrecisionTier::F64,
            PrecisionTier::F64Precise,
            PrecisionTier::DF64,
            PrecisionTier::F32,
        ];

        let mut results = Vec::new();
        let mut reference: Option<Vec<f64>> = None;

        for &tier in &tiers {
            let ref_slice = reference.as_deref();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.eval_tier(name, f64_source, input, n_elements, workgroups, tier, ref_slice)
            }))
            .unwrap_or(TierResult {
                tier,
                compiled: false,
                compile_us: 0.0,
                dispatch_us: 0.0,
                readback_us: 0.0,
                output: vec![],
                max_ulp_error: f64::NAN,
            });

            if tier == PrecisionTier::F64 && result.compiled {
                reference = Some(result.output.clone());
            }
            results.push(result);
        }

        ShaderEvalResult {
            shader_name: name.to_string(),
            tiers: results,
        }
    }

    fn eval_tier(
        &self,
        name: &str,
        f64_source: &str,
        input: &[f64],
        n_elements: usize,
        workgroups: u32,
        tier: PrecisionTier,
        reference: Option<&[f64]>,
    ) -> TierResult {
        let label = format!("{name}_{tier:?}");

        let t_compile = Instant::now();
        let shader_module = match tier {
            PrecisionTier::F64 | PrecisionTier::F64Precise => {
                self.device.compile_shader_f64(f64_source, Some(&label))
            }
            PrecisionTier::DF64 => {
                self.device.compile_shader_df64(f64_source, Some(&label))
            }
            _ => {
                self.device.compile_shader_f64(f64_source, Some(&label))
            }
        };
        let compile_us = t_compile.elapsed().as_secs_f64() * 1e6;

        let pipeline = self.device.device().create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(&label),
                layout: None,
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        );

        let input_bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_buf = self.device.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{label}_in")),
                contents: &input_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            },
        );

        let output_buf = self.device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}_out")),
            size: (n_elements * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = self.device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}_staging")),
            size: (n_elements * 8) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label}_bg")),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        // Warmup dispatches
        for _ in 0..WARMUP_REPS {
            let mut enc = self.device.device().create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("warmup") },
            );
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            self.device.submit_commands(std::iter::once(enc.finish()));
            let _ = self.device.poll_safe();
        }

        // Measured dispatches
        let t_dispatch = Instant::now();
        for _ in 0..MEASURE_REPS {
            let mut enc = self.device.device().create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("measure") },
            );
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            self.device.submit_commands(std::iter::once(enc.finish()));
            let _ = self.device.poll_safe();
        }
        let dispatch_us = t_dispatch.elapsed().as_secs_f64() * 1e6 / MEASURE_REPS as f64;

        // Readback
        let t_readback = Instant::now();
        {
            let mut enc = self.device.device().create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("readback") },
            );
            enc.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, (n_elements * 8) as u64);
            self.device.submit_commands(std::iter::once(enc.finish()));
        }

        let (tx, rx) = std::sync::mpsc::channel();
        staging_buf.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = self.device.poll_safe();
        let readback_us = t_readback.elapsed().as_secs_f64() * 1e6;

        let output = if rx.recv().is_ok() {
            let data = staging_buf.slice(..).get_mapped_range();
            let values: Vec<f64> = data
                .chunks_exact(8)
                .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            drop(data);
            staging_buf.unmap();
            values
        } else {
            vec![]
        };

        let max_ulp_error = if let Some(ref_vals) = reference {
            compute_max_ulp(&output, ref_vals)
        } else {
            0.0
        };

        TierResult {
            tier,
            compiled: true,
            compile_us,
            dispatch_us,
            readback_us,
            output,
            max_ulp_error,
        }
    }
}

fn compute_max_ulp(actual: &[f64], reference: &[f64]) -> f64 {
    actual
        .iter()
        .zip(reference.iter())
        .map(|(&a, &r)| {
            if a == r {
                return 0.0;
            }
            if !a.is_finite() || !r.is_finite() {
                return f64::INFINITY;
            }
            let ulp = (a.to_bits() as i64).wrapping_sub(r.to_bits() as i64).unsigned_abs() as f64;
            ulp
        })
        .fold(0.0f64, f64::max)
}
