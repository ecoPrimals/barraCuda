// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compute Graph - Lazy Execution for Operation Batching
//!
//! **Problem**: wgpu has significant per-dispatch overhead (~50-100μs).
//! Individual operations like `tensor.add(&b)` each submit a separate command buffer.
//!
//! **Solution**: Record operations lazily, then batch them into a single submission.
//!
//! **Usage**:
//! ```rust,ignore
//! # use barracuda::prelude::*;
//! # async fn example() -> Result<()> {
//! let device = Arc::new(WgpuDevice::new().await?);
//! let mut graph = ComputeGraph::new(device.as_ref());
//! // Use record_add/record_mul with buffers, then graph.execute()
//! # Ok(())
//! # }
//! ```
//!
//! **Performance Impact**: Batching 100 operations reduces overhead from
//! ~10ms (100 × 100μs) to ~200μs (1 × 100μs + 100 × 1μs kernel dispatch).

use crate::device::compute_pipeline::{storage_bgl_entry, uniform_bgl_entry};
use crate::device::WgpuDevice;
use crate::error::Result;
use std::sync::Arc;

/// Recorded operation in the compute graph.
#[derive(Debug)]
pub enum RecordedOp {
    /// Element-wise add: output = a + b
    Add {
        /// First input buffer
        input_a: wgpu::Buffer,
        /// Second input buffer
        input_b: wgpu::Buffer,
        /// Output buffer
        output: wgpu::Buffer,
        /// Element count
        size: usize,
    },
    /// Element-wise multiply: output = a * b
    Mul {
        /// First input buffer
        input_a: wgpu::Buffer,
        /// Second input buffer
        input_b: wgpu::Buffer,
        /// Output buffer
        output: wgpu::Buffer,
        /// Element count
        size: usize,
    },
    /// Fused multiply-add: output = a * b + c
    Fma {
        /// First input buffer
        input_a: wgpu::Buffer,
        /// Second input buffer
        input_b: wgpu::Buffer,
        /// Third input buffer
        input_c: wgpu::Buffer,
        /// Output buffer
        output: wgpu::Buffer,
        /// Element count
        size: usize,
    },
    /// Scale: output = a * scalar
    Scale {
        /// Input buffer
        input: wgpu::Buffer,
        /// Scale factor
        scalar: f32,
        /// Output buffer
        output: wgpu::Buffer,
        /// Element count
        size: usize,
    },
    /// Custom shader operation
    Custom {
        /// WGSL shader source
        shader_source: String,
        /// Input/output buffers
        buffers: Vec<wgpu::Buffer>,
        /// Workgroup counts (x, y, z)
        workgroups: (u32, u32, u32),
    },
}

/// Compute graph for batching GPU operations
///
/// Records operations without executing them, then batches
/// all operations into a single command buffer submission.
pub struct ComputeGraph {
    wgpu_device: Arc<WgpuDevice>,
    device: crate::device::wgpu_device::GuardedDeviceHandle,
    queue: wgpu::Queue,
    device_name: String,
    ops: Vec<RecordedOp>,
    optimal_workgroup_size: u32,
}

impl ComputeGraph {
    /// Create a new compute graph for a device
    pub fn new(wgpu_device: &WgpuDevice) -> Self {
        let optimal_wg = wgpu_device.optimal_workgroup_size();
        Self {
            wgpu_device: Arc::new(wgpu_device.clone()),
            device: wgpu_device.device.clone(),
            queue: wgpu_device.queue.clone(),
            device_name: wgpu_device.name().to_string(),
            ops: Vec::new(),
            optimal_workgroup_size: optimal_wg,
        }
    }

    /// Device name (for debug logging and diagnostics).
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Record an add operation
    pub fn record_add(
        &mut self,
        input_a: wgpu::Buffer,
        input_b: wgpu::Buffer,
        output: wgpu::Buffer,
        size: usize,
    ) {
        self.ops.push(RecordedOp::Add {
            input_a,
            input_b,
            output,
            size,
        });
    }

    /// Record a multiply operation
    pub fn record_mul(
        &mut self,
        input_a: wgpu::Buffer,
        input_b: wgpu::Buffer,
        output: wgpu::Buffer,
        size: usize,
    ) {
        self.ops.push(RecordedOp::Mul {
            input_a,
            input_b,
            output,
            size,
        });
    }

    /// Record a fused multiply-add operation
    pub fn record_fma(
        &mut self,
        input_a: wgpu::Buffer,
        input_b: wgpu::Buffer,
        input_c: wgpu::Buffer,
        output: wgpu::Buffer,
        size: usize,
    ) {
        self.ops.push(RecordedOp::Fma {
            input_a,
            input_b,
            input_c,
            output,
            size,
        });
    }

    /// Number of recorded operations
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Check if graph is empty
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Clear all recorded operations
    pub fn clear(&mut self) {
        self.ops.clear();
    }

    /// Execute all recorded operations in a single batch
    ///
    /// This is where the performance magic happens - all operations
    /// are encoded into a single command buffer and submitted together.
    pub fn execute(&mut self) -> Result<()> {
        if self.ops.is_empty() {
            return Ok(());
        }

        let mut encoder =
            self.wgpu_device
                .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                    label: Some("ComputeGraph Batch"),
                });

        let add_shader = self.compile_elementwise("output[idx] = a[idx] + b[idx];", "Add");
        let mul_shader = self.compile_elementwise("output[idx] = a[idx] * b[idx];", "Mul");
        let fma_shader = self.compile_shader(
            &format!(
                r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> c: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) {{ return; }}
    output[idx] = fma(a[idx], b[idx], c[idx]);
}}"#,
                wg = self.optimal_workgroup_size
            ),
            "FMA",
        );
        let scale_shader = self.compile_shader(
            &format!(
                r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<uniform> scalar: f32;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= arrayLength(&output)) {{ return; }}
    output[idx] = input[idx] * scalar;
}}"#,
                wg = self.optimal_workgroup_size
            ),
            "Scale",
        );

        for op in &self.ops {
            match op {
                RecordedOp::Add {
                    input_a,
                    input_b,
                    output,
                    size,
                } => {
                    self.dispatch_pass(
                        &mut encoder,
                        &add_shader,
                        &[
                            storage_bgl_entry(0, true),
                            storage_bgl_entry(1, true),
                            storage_bgl_entry(2, false),
                        ],
                        &[input_a, input_b, output],
                        *size,
                    );
                }
                RecordedOp::Mul {
                    input_a,
                    input_b,
                    output,
                    size,
                } => {
                    self.dispatch_pass(
                        &mut encoder,
                        &mul_shader,
                        &[
                            storage_bgl_entry(0, true),
                            storage_bgl_entry(1, true),
                            storage_bgl_entry(2, false),
                        ],
                        &[input_a, input_b, output],
                        *size,
                    );
                }
                RecordedOp::Fma {
                    input_a,
                    input_b,
                    input_c,
                    output,
                    size,
                } => {
                    self.dispatch_pass(
                        &mut encoder,
                        &fma_shader,
                        &[
                            storage_bgl_entry(0, true),
                            storage_bgl_entry(1, true),
                            storage_bgl_entry(2, true),
                            storage_bgl_entry(3, false),
                        ],
                        &[input_a, input_b, input_c, output],
                        *size,
                    );
                }
                RecordedOp::Scale {
                    input,
                    scalar,
                    output,
                    size,
                } => {
                    let scalar_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Scalar Uniform"),
                        size: 4,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    self.queue
                        .write_buffer(&scalar_buffer, 0, bytemuck::bytes_of(scalar));
                    self.dispatch_pass(
                        &mut encoder,
                        &scale_shader,
                        &[
                            storage_bgl_entry(0, true),
                            uniform_bgl_entry(1),
                            storage_bgl_entry(2, false),
                        ],
                        &[input, &scalar_buffer, output],
                        *size,
                    );
                }
                RecordedOp::Custom {
                    shader_source,
                    buffers,
                    workgroups,
                } => {
                    self.encode_custom_op(&mut encoder, shader_source, buffers, *workgroups);
                }
            }
        }

        self.wgpu_device
            .submit_and_poll_inner(Some(encoder.finish()));
        self.ops.clear();
        if self.wgpu_device.is_lost() {
            return Err(crate::error::BarracudaError::device(
                "GPU device lost during compute graph execution",
            ));
        }

        Ok(())
    }

    // ── Shader compilation helpers ──────────────────────────────────────

    fn compile_shader(&self, source: &str, label: &str) -> wgpu::ShaderModule {
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
    }

    fn compile_elementwise(&self, body: &str, label: &str) -> wgpu::ShaderModule {
        self.compile_shader(
            &format!(
                r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) {{ return; }}
    {body}
}}"#,
                wg = self.optimal_workgroup_size
            ),
            label,
        )
    }

    // ── Dispatch helpers ────────────────────────────────────────────────

    fn dispatch_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        shader: &wgpu::ShaderModule,
        bgl_entries: &[wgpu::BindGroupLayoutEntry],
        buffers: &[&wgpu::Buffer],
        size: usize,
    ) {
        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: bgl_entries,
            });

        let bind_entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            })
            .collect();

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &bind_entries,
        });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let workgroups = (size as u32)
            .div_ceil(self.optimal_workgroup_size)
            .min(65_535);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }

    fn encode_custom_op(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        shader_source: &str,
        buffers: &[wgpu::Buffer],
        workgroups: (u32, u32, u32),
    ) {
        let shader = self.compile_shader(shader_source, "Custom");

        let bgl_entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, _)| storage_bgl_entry(i as u32, i < buffers.len() - 1))
            .collect();

        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &bgl_entries,
            });

        let bind_entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            })
            .collect();

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &bind_entries,
        });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool;

    #[tokio::test]
    async fn test_compute_graph_batching() {
        let Some(device) = test_pool::get_test_device_if_gpu_available().await else {
            return;
        };

        let mut graph = ComputeGraph::new(&device);

        let data_a: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let data_b: Vec<f32> = (0..1000).map(|i| (i * 2) as f32).collect();

        let buf_a = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("A"),
                contents: bytemuck::cast_slice(&data_a),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let buf_b = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("B"),
                contents: bytemuck::cast_slice(&data_b),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let buf_out = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Out"),
            size: (1000 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        graph.record_add(buf_a, buf_b, buf_out, 1000);

        assert_eq!(graph.len(), 1);

        graph.execute().unwrap();

        assert!(graph.is_empty());
    }
}
