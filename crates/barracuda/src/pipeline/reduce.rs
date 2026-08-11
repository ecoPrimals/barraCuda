// SPDX-License-Identifier: AGPL-3.0-or-later
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
//! two bind groups, two dispatches, one copy, one `map_async`, one poll.
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
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::error::{BarracudaError, Result};
use crate::utils::chunk_to_array;
use bytemuck;
use std::sync::Arc;

/// DF64 sum-reduce shader (workgroup shared memory uses f32 pairs).
const SHADER_DF64: &str = include_str!("../shaders/reduce/sum_reduce_df64.wgsl");
/// Scalar f64 storage reduction (no workgroup memory — fallback path).
const SHADER_SCALAR_F64: &str = include_str!("../shaders/reduce/sum_reduce_scalar_f64.wgsl");
/// Subgroup-accelerated f64 tree reduction (fewest barriers, fastest path).
const SHADER_SUBGROUP_F64: &str = include_str!("../shaders/reduce/sum_reduce_subgroup_f64.wgsl");

/// Select the best reduce shader for this device based on probed capabilities.
///
/// Three tiers, highest performance first:
///
/// 1. **Subgroup** (`subgroupAdd` + shared memory): fewest barriers, full f64
///    precision. Requires `SUBGROUP` device feature AND verified f64 builtins.
///    RTX 3090 (sg=32): 3 barrier steps. RX 6950 XT (sg=64): 2 barrier steps.
///
/// 2. **DF64** (f32-pair workgroup tree): good throughput, ~48-bit precision.
///    Requires verified `df64_workgroup_reduce` probe.
///
/// 3. **Scalar** (sequential per-workgroup): always correct on any device
///    with `SHADER_F64` storage support. Slowest path.
fn shader_for_device(device: &WgpuDevice) -> &'static str {
    let has_subgroups = device.has_subgroups();
    let f64_builtins = crate::device::probe::cached_f64_builtins(device);

    if has_subgroups && f64_builtins.is_some() {
        tracing::info!(
            "Subgroup f64 reduction on {} (max_subgroup_size={})",
            device.adapter_info().name,
            device.adapter_info().subgroup_max_size,
        );
        return SHADER_SUBGROUP_F64;
    }

    if let Some(caps) = f64_builtins
        && !caps.df64_workgroup_reduce
    {
        tracing::info!(
            "DF64 workgroup reduce not verified on {} — using scalar f64 storage reduction",
            device.adapter_info().name
        );
        return SHADER_SCALAR_F64;
    }
    static DF64_COMBINED: std::sync::LazyLock<String> =
        std::sync::LazyLock::new(|| crate::shaders::df64_source(SHADER_DF64));
    &DF64_COMBINED
}

/// Multi-pass f64 reduction pipeline returning a single scalar.
///
/// Allocated once; call [`sum_f64`], [`max_f64`], or [`min_f64`] as many times
/// as needed.  All intermediate buffers and the `MAP_READ` staging buffer are
/// reused across calls — no per-call allocation.
///
/// Uses iterative reduction passes (ping-pong between two partial buffers)
/// to correctly handle arbitrary input sizes. Previous 2-pass design silently
/// dropped partial sums when `n > 256²` (65536) — e.g. kinetic energy at 16⁴
/// (262144 links) produced 1024 partials but pass 2 only read the first 256.
///
/// Supports arrays up to `n` elements (fixed at construction time).  If you need
/// to reduce arrays of varying sizes, construct a `ReduceScalarPipeline` for the
/// maximum expected size; smaller inputs are handled correctly (extra threads
/// contribute identity elements).
pub struct ReduceScalarPipeline {
    device: Arc<WgpuDevice>,
    n: u32,
    partial_buffer_a: wgpu::Buffer, // ⌈n/256⌉ × 8 bytes
    partial_buffer_b: wgpu::Buffer, // same size, for ping-pong intermediate passes
    scalar_staging: wgpu::Buffer,   // 8 bytes, MAP_READ
    sum_pipeline: wgpu::ComputePipeline,
    scalar_output: wgpu::Buffer, // 1 × 8 bytes STORAGE | COPY_SRC
    params_buf: wgpu::Buffer,
}

impl ReduceScalarPipeline {
    /// Build a reduction pipeline for arrays of up to `n` f64 elements.
    /// # Errors
    /// Returns [`Err`] if shader compilation fails or the device is lost.
    pub fn new(device: Arc<WgpuDevice>, n: usize) -> Result<Self> {
        let n_u32 = n as u32;
        let n_partial = n_u32.div_ceil(WORKGROUP_SIZE_1D) as usize;

        let module = device.compile_shader_f64(shader_for_device(&device), Some("sum_reduce_f64"));

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
                    immediate_size: 0,
                });

        let sum_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("sum_reduce_f64"),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some("sum_reduce_f64"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let partial_buf_size = (n_partial.max(1) * 8) as u64;
        let partial_buffer_a = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReduceScalar:partial_a"),
            size: partial_buf_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let partial_buffer_b = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReduceScalar:partial_b"),
            size: partial_buf_size,
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

        let params_data: [u32; 4] = [n_u32, 0, 0, 0];
        let params_bytes: &[u8] = bytemuck::cast_slice(&params_data);
        let params_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReduceScalar:params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device.queue.write_buffer(&params_buf, 0, params_bytes);

        Ok(Self {
            device,
            n: n_u32,
            partial_buffer_a,
            partial_buffer_b,
            scalar_staging,
            sum_pipeline,
            scalar_output,
            params_buf,
        })
    }

    /// Compute `Σ input[0..n]` in f64 precision.
    /// Dispatches two GPU passes (N → partials → 1 scalar) and reads back
    /// exactly 8 bytes.  `input` must be a STORAGE buffer of at least `n × 8`
    /// bytes with `COPY_SRC` usage if chained after another kernel, or `STORAGE`
    /// usage if written directly.
    /// # Errors
    /// Returns [`Err`] if GPU buffer mapping or readback fails (e.g., device lost).
    pub fn sum_f64(&self, input: &wgpu::Buffer) -> Result<f64> {
        self.reduce(input, "sum_reduce_f64")
    }

    /// Compute `max input[0..n]` in f64 precision.
    /// # Errors
    /// Returns [`Err`] if GPU buffer mapping or readback fails (e.g., device lost).
    pub fn max_f64(&self, input: &wgpu::Buffer) -> Result<f64> {
        self.reduce(input, "max_reduce_f64")
    }

    /// Compute `min input[0..n]` in f64 precision.
    /// # Errors
    /// Returns [`Err`] if GPU buffer mapping or readback fails (e.g., device lost).
    pub fn min_f64(&self, input: &wgpu::Buffer) -> Result<f64> {
        self.reduce(input, "min_reduce_f64")
    }

    /// Return the GPU-side scalar output buffer (for pipeline chaining).
    /// After the most recent [`sum_f64`] / [`max_f64`] / [`min_f64`] call, this
    /// buffer contains the result as a single f64.  Pass it to subsequent GPU
    /// kernels to avoid any CPU readback at all.
    #[must_use]
    pub fn scalar_buffer(&self) -> &wgpu::Buffer {
        &self.scalar_output
    }

    /// Encode a sum reduction into an existing command encoder WITHOUT
    /// submitting or reading back. The result stays GPU-resident in
    /// [`scalar_buffer()`]. Uses iterative multi-pass reduction.
    ///
    /// Use this for GPU-resident CG solvers and multi-kernel pipelines
    /// where CPU round-trips between reductions are unacceptable.
    ///
    /// After encoding, the caller submits the encoder and either:
    /// - Chains the scalar buffer into a subsequent GPU kernel, or
    /// - Calls [`readback_scalar`] to copy and map the result.
    pub fn encode_reduce_to_buffer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
    ) {
        let bgl = self.sum_pipeline.get_bind_group_layout(0);

        // Pass 1: input → partial_buffer_a
        let n_partial = self.n.div_ceil(WORKGROUP_SIZE_1D);
        {
            let bg = self
                .device
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ReduceScalar:BG:pass1:encode"),
                    layout: &bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: input.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.partial_buffer_a.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.params_buf.as_entire_binding(),
                        },
                    ],
                });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reduce:encode:pass1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sum_pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_partial, 1, 1);
        }

        // Iterative intermediate passes (ping-pong)
        let mut remaining = n_partial;
        let mut read_from_a = true;
        while remaining > WORKGROUP_SIZE_1D {
            let next_partial = remaining.div_ceil(WORKGROUP_SIZE_1D);
            let params = self.make_params_buf(remaining);
            let (src, dst) = if read_from_a {
                (&self.partial_buffer_a, &self.partial_buffer_b)
            } else {
                (&self.partial_buffer_b, &self.partial_buffer_a)
            };
            let bg = self
                .device
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ReduceScalar:BG:encode:intermediate"),
                    layout: &bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: src.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: dst.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params.as_entire_binding(),
                        },
                    ],
                });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reduce:encode:intermediate"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sum_pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(next_partial, 1, 1);
            drop(pass);
            remaining = next_partial;
            read_from_a = !read_from_a;
        }

        // Final pass → scalar_output
        {
            let final_src = if read_from_a {
                &self.partial_buffer_a
            } else {
                &self.partial_buffer_b
            };
            let params = self.make_params_buf(remaining);
            let bg = self
                .device
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ReduceScalar:BG:encode:final"),
                    layout: &bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: final_src.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.scalar_output.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params.as_entire_binding(),
                        },
                    ],
                });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reduce:encode:final"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sum_pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    /// Read back the scalar result after a previous [`encode_reduce_to_buffer`]
    /// + submit cycle. Copies `scalar_output` → staging → CPU.
    /// # Errors
    /// Returns [`Err`] if GPU buffer mapping or readback fails (e.g., device lost).
    pub fn readback_scalar(&self) -> Result<f64> {
        let mut enc = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("ReduceScalar:readback"),
            });
        enc.copy_buffer_to_buffer(&self.scalar_output, 0, &self.scalar_staging, 0, 8);
        self.device.submit_commands(Some(enc.finish()));

        let slice = self.scalar_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll_safe()?;
        rx.recv()
            .map_err(|_| {
                BarracudaError::execution_failed("ReduceScalarPipeline: readback channel closed")
            })?
            .map_err(|e| BarracudaError::execution_failed(e.to_string()))?;

        let data = slice.get_mapped_range();
        let v = f64::from_le_bytes(chunk_to_array::<8>(&data[..8])?);
        drop(data);
        self.scalar_staging.unmap();
        Ok(v)
    }

    // ── Private ──────────────────────────────────────────────────────────────

    fn make_params_buf(&self, size: u32) -> wgpu::Buffer {
        let data: [u32; 4] = [size, 0, 0, 0];
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        let buf = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReduceScalar:dyn_params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.device.queue.write_buffer(&buf, 0, bytes);
        buf
    }

    fn reduce(&self, input: &wgpu::Buffer, entry: &str) -> Result<f64> {
        let bgl = self.sum_pipeline.get_bind_group_layout(0);

        let pipeline = if entry == "sum_reduce_f64" {
            None
        } else {
            let module = self
                .device
                .compile_shader_f64(shader_for_device(&self.device), Some(entry));
            let layout =
                self.device
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("ReduceScalar:PL:alt"),
                        bind_group_layouts: &[&bgl],
                        immediate_size: 0,
                    });
            Some(
                self.device
                    .device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(entry),
                        layout: Some(&layout),
                        module: &module,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        cache: None,
                    }),
            )
        };
        let pl = pipeline.as_ref().unwrap_or(&self.sum_pipeline);

        let mut enc = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("ReduceScalar"),
            });

        // Pass 1: input → partial_buffer_a
        let n_partial = self.n.div_ceil(WORKGROUP_SIZE_1D);
        {
            let bg = self
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
                            resource: self.partial_buffer_a.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.params_buf.as_entire_binding(),
                        },
                    ],
                });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reduce:pass1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pl);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_partial, 1, 1);
        }

        // Iterative intermediate passes: ping-pong between partial_a and partial_b
        // until the remaining count fits in a single workgroup (≤ 256).
        let mut remaining = n_partial;
        let mut read_from_a = true;
        while remaining > WORKGROUP_SIZE_1D {
            let next_partial = remaining.div_ceil(WORKGROUP_SIZE_1D);
            let params = self.make_params_buf(remaining);
            let (src, dst) = if read_from_a {
                (&self.partial_buffer_a, &self.partial_buffer_b)
            } else {
                (&self.partial_buffer_b, &self.partial_buffer_a)
            };
            let bg = self
                .device
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ReduceScalar:BG:intermediate"),
                    layout: &bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: src.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: dst.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params.as_entire_binding(),
                        },
                    ],
                });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reduce:intermediate"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pl);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(next_partial, 1, 1);
            drop(pass);
            remaining = next_partial;
            read_from_a = !read_from_a;
        }

        // Final pass: remaining ≤ 256 partials → scalar_output (1 workgroup)
        {
            let final_src = if read_from_a {
                &self.partial_buffer_a
            } else {
                &self.partial_buffer_b
            };
            let params = self.make_params_buf(remaining);
            let bg = self
                .device
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ReduceScalar:BG:final"),
                    layout: &bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: final_src.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.scalar_output.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params.as_entire_binding(),
                        },
                    ],
                });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reduce:final"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pl);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        enc.copy_buffer_to_buffer(&self.scalar_output, 0, &self.scalar_staging, 0, 8);
        self.device.submit_commands(Some(enc.finish()));

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
        let v = f64::from_le_bytes(chunk_to_array::<8>(&data[..8])?);
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
        assert_eq!(WORKGROUP_SIZE_1D, 256);
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
