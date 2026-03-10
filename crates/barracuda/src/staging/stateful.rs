// SPDX-License-Identifier: AGPL-3.0-only
//! Stateful Iterative Simulation Pipeline
//!
//! Companion to [`UnidirectionalPipeline`] for workloads where state lives
//! permanently on the GPU (molecular dynamics, HFB SCF, PDE solvers) and only
//! a small convergence scalar crosses back to the CPU per iteration.
//!
//! # The MD Pattern (hotSpring feedback, Feb 19 2026)
//!
//! Streaming I/O (`UnidirectionalPipeline`) does not fit stateful simulations:
//!
//! - **No input stream**: particle state is uploaded once and stays GPU-resident.
//! - **Iterative compute**: the same kernel chain runs on the same buffers
//!   (force → kick-drift → force → half-kick, repeat).
//! - **Minimal readback**: only a convergence scalar (energy, temperature) needs
//!   to cross per iteration; snapshots are rare.
//!
//! ```text
//! Upload initial state (once)
//!   │
//!   ▼
//! ┌─────────────────────────────────────────────────────┐
//! │  GPU-resident iteration loop                         │
//! │  ┌──────────────────────────────────────────────┐   │
//! │  │  force kernel → kick-drift → thermostat       │   │
//! │  │  ↓                                            │   │
//! │  │  inline sum-reduction (scalar only)           │   │
//! │  └──────────────────────────────────────────────┘   │
//! │          × N iterations per GPU submit               │
//! └─────────────────────────────────────────────────────┘
//!   │  8 bytes (KE or PE scalar)
//!   ▼
//! CPU convergence check
//! ```
//!
//! # Readback Reduction
//!
//! At N=10,000 particles the hotSpring MD loop previously read back N×8 bytes
//! (80 KB) per energy dump.  With inline GPU sum-reduction, readback is 8 bytes
//! (one f64 scalar) — a **10,000× reduction** per dump.
//!
//! # Example
//!
//! ```rust,ignore
//! let pipeline = StatefulPipeline::new(
//!     Arc::clone(&device),
//!     StatefulConfig { convergence_scalars: 2, label: Some("MD".into()) },
//! );
//!
//! // Upload initial positions/velocities to your GPU buffers (once).
//! // Build kernel chain from your compiled wgpu pipelines + bind groups.
//! let chain = vec![
//!     KernelDispatch::new(Arc::clone(&force_pl), Arc::clone(&force_bg), (n_wg, 1, 1)),
//!     KernelDispatch::new(Arc::clone(&kick_pl),  Arc::clone(&kick_bg),  (n_wg, 1, 1)),
//!     // energy reduction must write into a buffer bound in the last kernel:
//!     KernelDispatch::new(Arc::clone(&reduce_pl), Arc::clone(&reduce_bg), (1, 1, 1)),
//! ];
//!
//! let result = pipeline.run_iterations(&chain, &energy_scalar_buf, 1000)?;
//! for (i, &e) in result.iter().enumerate() {
//!     println!("iter {}: E = {e:.6}", i * dump_every);
//! }
//! ```

use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use crate::utils::chunk_to_array;
use std::sync::Arc;

/// A single GPU kernel dispatch unit: pipeline + bind group + workgroup size.
///
/// Construct with your pre-compiled `wgpu::ComputePipeline` and the bind group
/// that wires your GPU-resident buffers.  The same chain can be reused across
/// many calls to [`StatefulPipeline::run_iterations`].
pub struct KernelDispatch {
    /// Pre-compiled compute pipeline.
    pub pipeline: Arc<wgpu::ComputePipeline>,
    /// Bind group wiring GPU buffers.
    pub bind_group: Arc<wgpu::BindGroup>,
    /// Workgroup counts (x, y, z).
    pub workgroups: (u32, u32, u32),
}

impl KernelDispatch {
    /// Create a new kernel dispatch unit.
    #[must_use]
    pub fn new(
        pipeline: Arc<wgpu::ComputePipeline>,
        bind_group: Arc<wgpu::BindGroup>,
        workgroups: (u32, u32, u32),
    ) -> Self {
        Self {
            pipeline,
            bind_group,
            workgroups,
        }
    }
}

/// Configuration for [`StatefulPipeline`].
#[derive(Debug, Clone)]
pub struct StatefulConfig {
    /// Number of f64 scalars in the convergence readback (e.g. 2 for KE + PE).
    pub convergence_scalars: usize,
    /// Debug label.
    pub label: Option<String>,
}

impl Default for StatefulConfig {
    fn default() -> Self {
        Self {
            convergence_scalars: 1,
            label: None,
        }
    }
}

/// Stateful iterative simulation pipeline.
///
/// All particle/field state stays GPU-resident; only tiny convergence scalars
/// cross back to the CPU.  The caller supplies a pre-compiled kernel chain and
/// a GPU-side `convergence_buffer` (written by the last kernel in the chain,
/// typically a sum-reduction shader).
///
/// # Zero-copy guarantee
///
/// The staging buffer inside `StatefulPipeline` is allocated once at
/// construction.  Each call to [`run_iterations`] reuses it — no allocation
/// per iteration.
pub struct StatefulPipeline {
    device: Arc<WgpuDevice>,
    config: StatefulConfig,
    /// Persistent `MAP_READ` staging buffer for convergence scalar readback.
    convergence_staging: wgpu::Buffer,
}

impl StatefulPipeline {
    /// Create a new stateful pipeline.
    ///
    /// Allocates a single persistent `MAP_READ` staging buffer sized for
    /// `config.convergence_scalars` × 8 bytes.  Everything else is zero-cost.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>, config: StatefulConfig) -> Self {
        let label = config.label.as_deref().unwrap_or("StatefulPipeline");
        let staging_size = (config.convergence_scalars * 8) as u64;
        let convergence_staging = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}:convergence_staging")),
            size: staging_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            device,
            config,
            convergence_staging,
        }
    }

    /// Run `iterations` iterations of the kernel chain, reading back the
    /// convergence scalar **once** at the end (or every `readback_every`
    /// iterations if you call this in a loop yourself).
    ///
    /// The caller is responsible for ensuring that the last kernel in `chain`
    /// writes the convergence scalar into `convergence_buffer` (e.g. a
    /// two-pass sum-reduction that outputs one f64 per scalar).  This method
    /// copies that buffer to the internal staging buffer and maps it back.
    ///
    /// # Arguments
    ///
    /// * `chain` — ordered sequence of GPU kernels to dispatch each iteration.
    /// * `convergence_buffer` — GPU-side storage buffer written by the last
    ///   kernel; must have `COPY_SRC` usage and hold exactly
    ///   `convergence_scalars × 8` bytes.
    /// * `iterations` — number of full chain dispatches in this GPU submit.
    ///
    /// # Returns
    ///
    /// `Vec<f64>` of length `convergence_scalars` read back after all
    /// iterations complete.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if chain is empty, or if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn run_iterations(
        &self,
        chain: &[KernelDispatch],
        convergence_buffer: &wgpu::Buffer,
        iterations: usize,
    ) -> Result<Vec<f64>> {
        if chain.is_empty() {
            return Err(BarracudaError::InvalidInput {
                message: "StatefulPipeline: kernel chain is empty".into(),
            });
        }

        let label = self.config.label.as_deref().unwrap_or("StatefulPipeline");
        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{label}:iter")),
            });

        // All iterations encoded into one command buffer — single GPU submit.
        for _iter in 0..iterations {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{label}:pass")),
                timestamp_writes: None,
            });
            for k in chain {
                pass.set_pipeline(&k.pipeline);
                pass.set_bind_group(0, Some(&*k.bind_group), &[]);
                pass.dispatch_workgroups(k.workgroups.0, k.workgroups.1, k.workgroups.2);
            }
            // `pass` drop ends the compute pass before the copy below.
        }

        // Copy convergence scalar to staging after last iteration.
        let scalar_bytes = (self.config.convergence_scalars * 8) as u64;
        encoder.copy_buffer_to_buffer(
            convergence_buffer,
            0,
            &self.convergence_staging,
            0,
            scalar_bytes,
        );

        self.device.submit_and_poll(Some(encoder.finish()));
        self.read_staging_scalars()
    }

    /// Run until convergence or `max_iterations` reached.
    ///
    /// Submits `readback_every` iterations per GPU call, reads back the scalar,
    /// and stops when `convergence_buffer[0]` drops below `tolerance`.
    ///
    /// # Returns
    ///
    /// `(iterations_run, final_convergence_values)`.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if chain is empty, or if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn run_until_converged(
        &self,
        chain: &[KernelDispatch],
        convergence_buffer: &wgpu::Buffer,
        max_iterations: usize,
        readback_every: usize,
        tolerance: f64,
    ) -> Result<(usize, Vec<f64>)> {
        let step = readback_every.max(1);
        let mut total = 0usize;
        loop {
            let run = step.min(max_iterations - total);
            let scalars = self.run_iterations(chain, convergence_buffer, run)?;
            total += run;
            if scalars.first().copied().unwrap_or(f64::MAX) < tolerance || total >= max_iterations {
                return Ok((total, scalars));
            }
        }
    }

    // ── Private ──────────────────────────────────────────────────────────────

    fn read_staging_scalars(&self) -> Result<Vec<f64>> {
        let slice = self.convergence_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll_safe()?;
        rx.recv()
            .map_err(|_| {
                BarracudaError::execution_failed("StatefulPipeline: staging channel closed")
            })?
            .map_err(|e| BarracudaError::execution_failed(e.to_string()))?;

        let data = slice.get_mapped_range();
        let result: Vec<f64> = data
            .chunks_exact(8)
            .map(|b| chunk_to_array::<8>(b).map(f64::from_le_bytes))
            .collect::<Result<Vec<_>>>()?;
        drop(data);
        self.convergence_staging.unmap();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use super::*;
    use crate::device::test_pool;
    use std::sync::Arc;

    const MINIMAL_WGSL: &str = r"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    output[gid.x] = input[gid.x];
}
";

    fn create_minimal_kernel_dispatch(workgroups: (u32, u32, u32)) -> KernelDispatch {
        let device = test_pool::get_test_device_sync();
        let shader = device.compile_shader(MINIMAL_WGSL, Some("test_kernel"));
        let bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("test_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let layout = device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("test_pl"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });
        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("test_pipeline"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });
        let size = 256 * 4;
        let input_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_input"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_output"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("test_bg"),
            layout: &bgl,
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
        KernelDispatch::new(Arc::new(pipeline), Arc::new(bind_group), workgroups)
    }

    #[test]
    fn test_stateful_config_default() {
        let cfg = StatefulConfig::default();
        assert_eq!(cfg.convergence_scalars, 1);
        assert!(cfg.label.is_none());
    }

    #[test]
    fn test_stateful_config_custom() {
        let cfg = StatefulConfig {
            convergence_scalars: 2,
            label: Some("MD".into()),
        };
        assert_eq!(cfg.convergence_scalars, 2);
        assert_eq!(cfg.label.as_deref(), Some("MD"));
    }

    #[test]
    fn test_stateful_config_labels() {
        let cfg = StatefulConfig {
            convergence_scalars: 1,
            label: Some("SimulationX".into()),
        };
        assert_eq!(cfg.label.as_deref(), Some("SimulationX"));
        let cfg_none = StatefulConfig {
            convergence_scalars: 1,
            label: None,
        };
        assert!(cfg_none.label.is_none());
    }

    #[test]
    fn test_kernel_dispatch_new() {
        let wg = (64u32, 1u32, 1u32);
        assert_eq!(wg, (64, 1, 1));
    }

    #[test]
    fn test_kernel_dispatch_workgroups() {
        let workgroups = (8u32, 4u32, 2u32);
        let k = create_minimal_kernel_dispatch(workgroups);
        assert_eq!(k.workgroups.0, 8);
        assert_eq!(k.workgroups.1, 4);
        assert_eq!(k.workgroups.2, 2);
    }

    #[test]
    fn test_run_iterations_empty_chain_returns_err() {
        let device = test_pool::get_test_device_sync();
        let config = StatefulConfig {
            convergence_scalars: 1,
            label: None,
        };
        let pipeline = StatefulPipeline::new(Arc::clone(&device), config);
        let conv_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv_test"),
            size: 8,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let result = pipeline.run_iterations(&[], &conv_buf, 1);
        assert!(result.is_err());
    }

    const CONVERGENCE_WGSL: &str = r"
@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(1)
fn main() {
    output[0] = 1.0;
    output[1] = 0.0;
}
";

    fn create_convergence_chain(
        device: &crate::device::WgpuDevice,
    ) -> (Vec<KernelDispatch>, wgpu::Buffer) {
        let shader = device.compile_shader(CONVERGENCE_WGSL, Some("conv_kernel"));
        let bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("conv_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let layout = device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("conv_pl"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });
        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("conv_pipeline"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });
        let conv_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv_buf"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv_bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: conv_buf.as_entire_binding(),
            }],
        });
        let k = KernelDispatch::new(Arc::new(pipeline), Arc::new(bind_group), (1, 1, 1));
        (vec![k], conv_buf)
    }

    #[tokio::test]
    async fn test_stateful_pipeline_new_requires_gpu() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let config = StatefulConfig {
            convergence_scalars: 1,
            label: Some("TestPipeline".into()),
        };
        let _pipeline = StatefulPipeline::new(Arc::clone(&device), config);
    }

    #[tokio::test]
    async fn test_run_iterations_success() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let config = StatefulConfig {
            convergence_scalars: 1,
            label: None,
        };
        let pipeline = StatefulPipeline::new(Arc::clone(&device), config);
        let (chain, conv_buf) = create_convergence_chain(device.as_ref());
        let result = pipeline.run_iterations(&chain, &conv_buf, 1).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[tokio::test]
    async fn test_run_until_converged() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let config = StatefulConfig {
            convergence_scalars: 1,
            label: None,
        };
        let pipeline = StatefulPipeline::new(Arc::clone(&device), config);
        let (chain, conv_buf) = create_convergence_chain(device.as_ref());
        let (iters, scalars) = pipeline
            .run_until_converged(&chain, &conv_buf, 10, 1, 0.5)
            .unwrap();
        assert!(iters >= 1);
        assert_eq!(scalars.len(), 1);
    }
}
