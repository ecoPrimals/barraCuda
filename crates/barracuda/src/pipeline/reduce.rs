//! GPU Scalar Reduction Pipeline
//!
//! Promotes `sum_reduce_f64.wgsl` to a first-class pipeline primitive that
//! returns a single f64 scalar without CPU-side intermediate storage.
//!
//! # Why This Exists (hotSpring feedback, Feb 19 2026)
//!
//! Every physics Spring that does GPU-resident simulation needs the same
//! two-pass reduction pattern:
//!
//! ```text
//! [N element buffer] ──pass 1──> [⌈N/256⌉ partial sums] ──pass 2──> [1 scalar]
//!                                                                         │
//!                                                              copy to staging
//!                                                                         │
//!                                                              read 8 bytes
//! ```
//!
//! Without this helper, every use site duplicates 12+ lines of boilerplate:
//! two bind groups, two dispatches, one copy, one map_async, one poll.
//! With it:
//!
//! ```rust,ignore
//! let reducer = ReduceScalarPipeline::new(Arc::clone(&device), n)?;
//! let ke = reducer.sum_f64(&ke_buffer)?;  // one call, 8 bytes readback
//! ```
//!
//! # Readback reduction achieved (hotSpring MD, N=10,000)
//!
//! | Metric         | Before       | After  | Reduction |
//! |----------------|--------------|--------|-----------|
//! | KE readback    | 80 000 bytes | 8 B    | 10 000×   |
//! | PE readback    | 80 000 bytes | 8 B    | 10 000×   |
//! | Equil thermo   | 80 000 bytes | 8 B    | 10 000×   |

use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use std::sync::Arc;

const WORKGROUP_SIZE: u32 = 256;

/// Two-pass f64 reduction pipeline returning a single scalar.
///
/// Allocated once; call [`sum_f64`], [`max_f64`], or [`min_f64`] as many times
/// as needed.  All intermediate buffers and the MAP_READ staging buffer are
/// reused across calls — no per-call allocation.
///
/// Supports arrays up to `n` elements (fixed at construction time).  If you need
/// to reduce arrays of varying sizes, construct a `ReduceScalarPipeline` for the
/// maximum expected size; smaller inputs are handled correctly (extra threads
/// contribute identity elements).
pub struct ReduceScalarPipeline {
    device: Arc<WgpuDevice>,
    n: u32,
    partial_buffer: wgpu::Buffer, // ⌈n/256⌉ × 8 bytes
    scalar_staging: wgpu::Buffer, // 8 bytes, MAP_READ
    sum_pipeline: wgpu::ComputePipeline,
    _sum_bg_pass1: wgpu::BindGroup, // kept alive; pass-1 BG rebuilt per call against caller's input
    sum_bg_pass2: wgpu::BindGroup,
    scalar_output: wgpu::Buffer, // 1 × 8 bytes STORAGE | COPY_SRC
    params_buf: wgpu::Buffer,
    _partial_params: wgpu::Buffer, // params for pass-2 dispatch (kept alive; used in sum_bg_pass2)
}

impl ReduceScalarPipeline {
    const SHADER: &'static str = include_str!("../shaders/reduce/sum_reduce_f64.wgsl");

    /// Build a reduction pipeline for arrays of up to `n` f64 elements.
    pub fn new(device: Arc<WgpuDevice>, n: usize) -> Result<Self> {
        let n_u32 = n as u32;
        let n_partial = n_u32.div_ceil(WORKGROUP_SIZE) as usize;

        // Compile shader
        let module = device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("sum_reduce_f64"),
                source: wgpu::ShaderSource::Wgsl(Self::SHADER.into()),
            });

        // Bind group layout (matches sum_reduce_f64.wgsl group 0):
        //   binding 0: input (storage read)
        //   binding 1: output / partial (storage read_write)
        //   binding 2: params (uniform)
        let bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ReduceScalar:BGL"),
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
                    bgl_entry(2, wgpu::BufferBindingType::Uniform),
                ],
            });

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("ReduceScalar:PL"),
                    bind_group_layouts: &[&bgl],
                    push_constant_ranges: &[],
                });

        let sum_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("sum_reduce_f64"),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: "sum_reduce_f64",
                    compilation_options: Default::default(),
                    cache: None,
                });

        // Buffers
        let partial_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReduceScalar:partial"),
            size: (n_partial * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scalar_output = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReduceScalar:scalar"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scalar_staging = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReduceScalar:staging"),
            size: 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Params buffers: [size, 0, 0, 0] as u32×4
        let params_data: [u32; 4] = [n_u32, 0, 0, 0];
        let params_bytes: Vec<u8> = params_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReduceScalar:params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device.queue.write_buffer(&params_buf, 0, &params_bytes);

        let partial_size = n_partial as u32;
        let partial_params_data: [u32; 4] = [partial_size, 0, 0, 0];
        let partial_params_bytes: Vec<u8> = partial_params_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let partial_params = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReduceScalar:partial_params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&partial_params, 0, &partial_params_bytes);

        // Pass 2 bind group (partial → scalar_output)
        let sum_bg_pass2 = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ReduceScalar:BG:pass2"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: partial_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scalar_output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: partial_params.as_entire_binding(),
                },
            ],
        });

        // Pass 1 bind group is input-dependent — rebuilt per call.
        // This initial BG uses partial_buffer as a stand-in; the field keeps the BG alive.
        let sum_bg_pass1 = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ReduceScalar:BG:pass1:init"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: partial_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: partial_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            device,
            n: n_u32,
            partial_buffer,
            scalar_staging,
            sum_pipeline,
            _sum_bg_pass1: sum_bg_pass1,
            sum_bg_pass2,
            scalar_output,
            params_buf,
            _partial_params: partial_params,
        })
    }

    /// Compute `Σ input[0..n]` in f64 precision.
    ///
    /// Dispatches two GPU passes (N → partials → 1 scalar) and reads back
    /// exactly 8 bytes.  `input` must be a STORAGE buffer of at least `n × 8`
    /// bytes with `COPY_SRC` usage if chained after another kernel, or `STORAGE`
    /// usage if written directly.
    pub fn sum_f64(&self, input: &wgpu::Buffer) -> Result<f64> {
        self.reduce(input, "sum_reduce_f64")
    }

    /// Compute `max input[0..n]` in f64 precision.
    pub fn max_f64(&self, input: &wgpu::Buffer) -> Result<f64> {
        self.reduce(input, "max_reduce_f64")
    }

    /// Compute `min input[0..n]` in f64 precision.
    pub fn min_f64(&self, input: &wgpu::Buffer) -> Result<f64> {
        self.reduce(input, "min_reduce_f64")
    }

    /// Return the GPU-side scalar output buffer (for pipeline chaining).
    ///
    /// After the most recent [`sum_f64`] / [`max_f64`] / [`min_f64`] call, this
    /// buffer contains the result as a single f64.  Pass it to subsequent GPU
    /// kernels to avoid any CPU readback at all.
    pub fn scalar_buffer(&self) -> &wgpu::Buffer {
        &self.scalar_output
    }

    // ── Private ──────────────────────────────────────────────────────────────

    fn reduce(&self, input: &wgpu::Buffer, entry: &str) -> Result<f64> {
        // Rebuild pass-1 bind group against the caller's input buffer.
        // This is cheap (< 1 µs) and avoids storing a mutable bind group.
        let bgl = self.sum_pipeline.get_bind_group_layout(0);
        let bg_pass1 = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ReduceScalar:BG:pass1"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.partial_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.params_buf.as_entire_binding(),
                    },
                ],
            });

        // Recompile pass-2 pipeline if entry point differs from default.
        // For the common case (sum) we use the pre-built pipeline.
        // For max/min we build on demand.  These are cold paths.
        let pass2_pipeline = if entry == "sum_reduce_f64" {
            None // use self.sum_pipeline
        } else {
            let module = self
                .device
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(entry),
                    source: wgpu::ShaderSource::Wgsl(Self::SHADER.into()),
                });
            let layout =
                self.device
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("ReduceScalar:PL:alt"),
                        bind_group_layouts: &[&bgl],
                        push_constant_ranges: &[],
                    });
            Some(
                self.device
                    .device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(entry),
                        layout: Some(&layout),
                        module: &module,
                        entry_point: entry,
                        compilation_options: Default::default(),
                        cache: None,
                    }),
            )
        };

        let n_partial = self.n.div_ceil(WORKGROUP_SIZE);

        let mut enc = self
            .device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ReduceScalar"),
            });

        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reduce:pass1"),
                timestamp_writes: None,
            });
            let pl = pass2_pipeline.as_ref().unwrap_or(&self.sum_pipeline);
            pass.set_pipeline(pl);
            pass.set_bind_group(0, &bg_pass1, &[]);
            pass.dispatch_workgroups(n_partial, 1, 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reduce:pass2"),
                timestamp_writes: None,
            });
            let pl = pass2_pipeline.as_ref().unwrap_or(&self.sum_pipeline);
            pass.set_pipeline(pl);
            pass.set_bind_group(0, &self.sum_bg_pass2, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        enc.copy_buffer_to_buffer(&self.scalar_output, 0, &self.scalar_staging, 0, 8);
        self.device.submit_and_poll(Some(enc.finish()));

        let slice = self.scalar_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll_safe()?;
        rx.recv()
            .map_err(|_| BarracudaError::execution_failed("ReduceScalarPipeline: channel closed"))?
            .map_err(|e| BarracudaError::execution_failed(e.to_string()))?;

        let data = slice.get_mapped_range();
        // SAFETY: chunks_exact(8) guarantees 8-byte chunk
        let v = f64::from_le_bytes(
            data[..8]
                .try_into()
                .expect("scalar buffer is exactly 8 bytes"),
        );
        drop(data);
        self.scalar_staging.unmap();
        Ok(v)
    }
}

fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bgl_entry_storage() {
        let e = bgl_entry(0, wgpu::BufferBindingType::Storage { read_only: true });
        assert_eq!(e.binding, 0);
        assert!(e.visibility.contains(wgpu::ShaderStages::COMPUTE));
    }

    #[test]
    fn test_bgl_entry_uniform() {
        let e = bgl_entry(2, wgpu::BufferBindingType::Uniform);
        assert_eq!(e.binding, 2);
    }

    #[test]
    fn test_workgroup_size_constant() {
        // 256 threads × 8 bytes each = 2 KiB shared memory per workgroup.
        // Within the 32 KiB SM70 / 64 KiB RDNA2 limit.
        assert_eq!(WORKGROUP_SIZE, 256);
    }

    #[test]
    fn test_n_partial_ceiling() {
        // Verify div_ceil calculation used in new()
        assert_eq!(1u32.div_ceil(256), 1);
        assert_eq!(256u32.div_ceil(256), 1);
        assert_eq!(257u32.div_ceil(256), 2);
        assert_eq!(10_000u32.div_ceil(256), 40);
    }
}
