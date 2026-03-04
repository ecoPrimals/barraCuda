// SPDX-License-Identifier: AGPL-3.0-or-later
//! Batched command encoder — fuses multiple compute dispatches into one submission.
//!
//! Reduces per-op `queue.submit()` overhead for multi-op pipelines (MLP, transformer layers).
//! neuralSpring V64 demonstrated 46–78× speedups from batching submits.

use crate::device::WgpuDevice;

struct Binding<'a> {
    index: u32,
    buffer: &'a wgpu::Buffer,
    read_only: bool,
    is_uniform: bool,
}

/// A single compute pass within a batched submission.
pub struct BatchedPass<'a> {
    label: &'a str,
    shader_source: &'a str,
    entry_point: &'a str,
    f64_shader: bool,
    bindings: Vec<Binding<'a>>,
    workgroups: (u32, u32, u32),
}

/// Batches multiple compute dispatches into a single command encoder submission.
///
/// ## Example
/// ```ignore
/// let mut batch = BatchedEncoder::new(&device);
/// batch.dispatch("op1", SHADER1, "main").storage_read(0, &buf_a).storage_rw(1, &buf_b)
///     .uniform(2, &params1).workgroups(64, 1, 1);
/// batch.dispatch("op2", SHADER2, "main").storage_read(0, &buf_b).storage_rw(1, &buf_c)
///     .uniform(2, &params2).workgroups(32, 1, 1);
/// batch.submit(); // Single queue.submit() for all ops
/// ```
pub struct BatchedEncoder<'a> {
    device: &'a WgpuDevice,
    passes: Vec<BatchedPass<'a>>,
}

/// Builder for a single pass; call `.workgroups()` to add it to the batch.
pub struct BatchedPassBuilder<'a, 'b> {
    encoder: &'b mut BatchedEncoder<'a>,
    pass: BatchedPass<'a>,
}

impl<'a> BatchedEncoder<'a> {
    pub fn new(device: &'a WgpuDevice) -> Self {
        Self {
            device,
            passes: Vec::new(),
        }
    }

    /// Start a new dispatch; chain bindings and call `.workgroups()` to add it.
    pub fn dispatch<'b>(
        &'b mut self,
        label: &'a str,
        shader_source: &'a str,
        entry_point: &'a str,
    ) -> BatchedPassBuilder<'a, 'b> {
        BatchedPassBuilder {
            encoder: self,
            pass: BatchedPass {
                label,
                shader_source,
                entry_point,
                f64_shader: false,
                bindings: Vec::new(),
                workgroups: (1, 1, 1),
            },
        }
    }

    /// Create encoder, record passes, and submit in one call.
    pub fn submit(self) {
        let _permit = self.device.acquire_dispatch();
        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("BatchedEncoder"),
            });

        let dev = &self.device;
        for pass in &self.passes {
            let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> = pass
                .bindings
                .iter()
                .map(|b| wgpu::BindGroupLayoutEntry {
                    binding: b.index,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: if b.is_uniform {
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        }
                    } else {
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: b.read_only,
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        }
                    },
                    count: None,
                })
                .collect();
            let bgl = dev
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(pass.label),
                    entries: &bgl_entries,
                });
            let bg_entries: Vec<wgpu::BindGroupEntry<'_>> = pass
                .bindings
                .iter()
                .map(|b| wgpu::BindGroupEntry {
                    binding: b.index,
                    resource: b.buffer.as_entire_binding(),
                })
                .collect();
            let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(pass.label),
                layout: &bgl,
                entries: &bg_entries,
            });
            let module = if pass.f64_shader {
                dev.compile_shader_f64(pass.shader_source, Some(pass.label))
            } else {
                dev.compile_shader(pass.shader_source, Some(pass.label))
            };
            let pl = dev
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(pass.label),
                    bind_group_layouts: &[&bgl],
                    push_constant_ranges: &[],
                });
            let pipeline = dev
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(pass.label),
                    layout: Some(&pl),
                    module: &module,
                    entry_point: pass.entry_point,
                    cache: dev.pipeline_cache(),
                    compilation_options: Default::default(),
                });
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(pass.label),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups(pass.workgroups.0, pass.workgroups.1, pass.workgroups.2);
        }

        self.device.submit_and_poll_inner(Some(encoder.finish()));
    }
}

impl<'a, 'b> BatchedPassBuilder<'a, 'b> {
    pub fn f64(mut self) -> Self {
        self.pass.f64_shader = true;
        self
    }

    pub fn storage_read(mut self, index: u32, buffer: &'a wgpu::Buffer) -> Self {
        self.pass.bindings.push(Binding {
            index,
            buffer,
            read_only: true,
            is_uniform: false,
        });
        self
    }

    pub fn storage_rw(mut self, index: u32, buffer: &'a wgpu::Buffer) -> Self {
        self.pass.bindings.push(Binding {
            index,
            buffer,
            read_only: false,
            is_uniform: false,
        });
        self
    }

    pub fn uniform(mut self, index: u32, buffer: &'a wgpu::Buffer) -> Self {
        self.pass.bindings.push(Binding {
            index,
            buffer,
            read_only: true,
            is_uniform: true,
        });
        self
    }

    pub fn workgroups(mut self, x: u32, y: u32, z: u32) -> &'b mut BatchedEncoder<'a> {
        self.pass.workgroups = (x, y, z);
        self.encoder.passes.push(self.pass);
        self.encoder
    }
}
