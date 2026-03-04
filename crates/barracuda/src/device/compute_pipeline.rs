// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU compute pipeline builder — eliminates bind-group/pipeline boilerplate.
//!
//! The ~80 lines of create_bind_group_layout → create_bind_group →
//! create_pipeline_layout → create_compute_pipeline → create_command_encoder →
//! begin_compute_pass → set_pipeline → set_bind_group → dispatch pattern is
//! repeated in 50+ ops. This module captures that pattern as a builder.
//!
//! # Example
//!
//! ```ignore
//! use barracuda::device::compute_pipeline::ComputeDispatch;
//!
//! ComputeDispatch::new(&device, "MyOp")
//!     .shader(source, "main")
//!     .storage_read(0, &input_buf)
//!     .storage_rw(1, &output_buf)
//!     .uniform(2, &params_buf)
//!     .dispatch_1d(element_count)
//!     .submit();
//! ```

use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::WgpuDevice;

/// Buffer binding declaration for the pipeline builder.
struct Binding<'a> {
    index: u32,
    buffer: &'a wgpu::Buffer,
    read_only: bool,
    is_uniform: bool,
}

/// Ergonomic one-shot compute dispatch builder.
///
/// Builds the bind-group layout, bind group, pipeline layout, pipeline,
/// command encoder, and compute pass from a fluent API, then submits.
pub struct ComputeDispatch<'a> {
    device: &'a WgpuDevice,
    label: &'a str,
    bindings: Vec<Binding<'a>>,
    shader_source: Option<&'a str>,
    entry_point: &'a str,
    workgroups: (u32, u32, u32),
    f64_shader: bool,
}

impl<'a> ComputeDispatch<'a> {
    pub fn new(device: &'a WgpuDevice, label: &'a str) -> Self {
        Self {
            device,
            label,
            bindings: Vec::new(),
            shader_source: None,
            entry_point: "main",
            workgroups: (1, 1, 1),
            f64_shader: false,
        }
    }

    /// Set the WGSL shader source and entry point.
    pub fn shader(mut self, source: &'a str, entry_point: &'a str) -> Self {
        self.shader_source = Some(source);
        self.entry_point = entry_point;
        self
    }

    /// Use f64 shader compilation path (with precision patching + ILP optimizer).
    pub fn f64(mut self) -> Self {
        self.f64_shader = true;
        self
    }

    /// Bind a read-only storage buffer at `index`.
    pub fn storage_read(mut self, index: u32, buffer: &'a wgpu::Buffer) -> Self {
        self.bindings.push(Binding {
            index,
            buffer,
            read_only: true,
            is_uniform: false,
        });
        self
    }

    /// Bind a read-write storage buffer at `index`.
    pub fn storage_rw(mut self, index: u32, buffer: &'a wgpu::Buffer) -> Self {
        self.bindings.push(Binding {
            index,
            buffer,
            read_only: false,
            is_uniform: false,
        });
        self
    }

    /// Bind a uniform buffer at `index`.
    pub fn uniform(mut self, index: u32, buffer: &'a wgpu::Buffer) -> Self {
        self.bindings.push(Binding {
            index,
            buffer,
            read_only: true,
            is_uniform: true,
        });
        self
    }

    /// Set dispatch to `ceil(n / WORKGROUP_SIZE_1D)` workgroups in x.
    pub fn dispatch_1d(mut self, element_count: u32) -> Self {
        self.workgroups = (element_count.div_ceil(WORKGROUP_SIZE_1D), 1, 1);
        self
    }

    /// Set explicit workgroup counts.
    pub fn dispatch(mut self, x: u32, y: u32, z: u32) -> Self {
        self.workgroups = (x, y, z);
        self
    }

    /// Build everything and submit the compute pass to the device queue.
    ///
    /// Holds a dispatch permit for the full compile→bind→submit lifecycle,
    /// respecting the device's hardware-aware concurrency budget.
    /// All resource creation is covered by the encoder barrier to prevent
    /// wgpu-core races between poll cleanup and resource allocation.
    pub fn submit(self) {
        let _permit = self.device.acquire_dispatch();
        self.device.encoding_guard();
        let source = self
            .shader_source
            .expect("ComputeDispatch: shader source required");

        let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> = self
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

        let bgl = self
            .device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(self.label),
                entries: &bgl_entries,
            });

        let bg_entries: Vec<wgpu::BindGroupEntry<'_>> = self
            .bindings
            .iter()
            .map(|b| wgpu::BindGroupEntry {
                binding: b.index,
                resource: b.buffer.as_entire_binding(),
            })
            .collect();

        let bg = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(self.label),
                layout: &bgl,
                entries: &bg_entries,
            });

        let module = if self.f64_shader {
            self.device.compile_shader_f64(source, Some(self.label))
        } else {
            self.device.compile_shader(source, Some(self.label))
        };

        let pl = self
            .device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(self.label),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });

        let pipeline =
            self.device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(self.label),
                    layout: Some(&pl),
                    module: &module,
                    entry_point: Some(self.entry_point),
                    cache: self.device.pipeline_cache(),
                    compilation_options: Default::default(),
                });

        let mut encoder =
            self.device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(self.label),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(self.label),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(self.workgroups.0, self.workgroups.1, self.workgroups.2);
        }

        let commands = encoder.finish();
        self.device.encoding_complete();
        self.device.submit_and_poll_inner(Some(commands));
    }
}

/// Helper to create a storage bind-group-layout entry.
///
/// Replaces the 7-line `BindGroupLayoutEntry` struct literal that appears
/// in every GPU op. Use for ops that need the BGL separately (e.g.,
/// cached pipelines that outlive a single dispatch).
pub fn storage_bgl_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Helper to create a uniform bind-group-layout entry.
pub fn uniform_bgl_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
