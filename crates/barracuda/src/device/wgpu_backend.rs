// SPDX-License-Identifier: AGPL-3.0-only
//! `GpuBackend` implementation for `WgpuDevice`.
//!
//! Delegates to existing `WgpuDevice` methods. The dispatch path moves the
//! wgpu boilerplate (bind group layout → bind group → pipeline → encoder →
//! compute pass → submit → poll) from `ComputeDispatch::submit()` into
//! `GpuBackend::dispatch_compute()`, making `ComputeDispatch` a thin builder
//! that works with any backend.

use super::backend::{DispatchDescriptor, GpuBackend};
use super::wgpu_device::WgpuDevice;
use crate::error::{BarracudaError, Result};

impl GpuBackend for WgpuDevice {
    type Buffer = wgpu::Buffer;

    fn name(&self) -> &str {
        WgpuDevice::name(self)
    }

    fn has_f64_shaders(&self) -> bool {
        WgpuDevice::has_f64_shaders(self)
    }

    fn is_lost(&self) -> bool {
        WgpuDevice::is_lost(self)
    }

    fn alloc_buffer(&self, label: &str, size: u64) -> Result<wgpu::Buffer> {
        self.encoding_guard();
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.encoding_complete();
        Ok(buf)
    }

    fn alloc_buffer_init(&self, label: &str, contents: &[u8]) -> Result<wgpu::Buffer> {
        self.encoding_guard();
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        self.encoding_complete();
        Ok(buf)
    }

    fn alloc_uniform(&self, label: &str, contents: &[u8]) -> Result<wgpu::Buffer> {
        self.encoding_guard();
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        self.encoding_complete();
        Ok(buf)
    }

    fn upload(&self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        self.encoding_guard();
        self.queue.write_buffer(buffer, offset, data);
        self.encoding_complete();
    }

    fn download(&self, buffer: &wgpu::Buffer, size: u64) -> Result<bytes::Bytes> {
        if size == 0 {
            return Ok(bytes::Bytes::new());
        }
        if self.is_lost() {
            return Err(BarracudaError::device_lost(
                "cannot download buffer — device lost",
            ));
        }

        let _permit = self.acquire_dispatch();

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("backend readback staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("backend readback"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.submit_and_poll_inner(Some(encoder.finish()));

        if self.is_lost() {
            return Err(BarracudaError::device_lost(
                "device lost during readback copy",
            ));
        }

        let vec_data = self.map_staging_buffer::<u8>(&staging, size as usize)?;
        Ok(bytes::Bytes::from(vec_data))
    }

    fn dispatch_compute(&self, desc: DispatchDescriptor<'_, Self>) -> Result<()> {
        let _permit = self.acquire_dispatch();
        self.encoding_guard();

        let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> = desc
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
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(desc.label),
                entries: &bgl_entries,
            });

        let bg_entries: Vec<wgpu::BindGroupEntry<'_>> = desc
            .bindings
            .iter()
            .map(|b| wgpu::BindGroupEntry {
                binding: b.index,
                resource: b.buffer.as_entire_binding(),
            })
            .collect();

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(desc.label),
            layout: &bgl,
            entries: &bg_entries,
        });

        let module = if desc.df64_shader {
            self.compile_shader_df64(desc.shader_source, Some(desc.label))
        } else if desc.f64_shader {
            self.compile_shader_f64(desc.shader_source, Some(desc.label))
        } else {
            self.compile_shader(desc.shader_source, Some(desc.label))
        };

        let pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(desc.label),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(desc.label),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(desc.entry_point),
                cache: self.pipeline_cache(),
                compilation_options: Default::default(),
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(desc.label),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(desc.label),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(desc.workgroups.0, desc.workgroups.1, desc.workgroups.2);
        }

        let commands = encoder.finish();
        self.encoding_complete();
        self.submit_and_poll_inner(Some(commands));
        Ok(())
    }

    fn dispatch_compute_batch(&self, descs: Vec<DispatchDescriptor<'_, Self>>) -> Result<()> {
        if descs.is_empty() {
            return Ok(());
        }
        if descs.len() == 1 {
            return self
                .dispatch_compute(descs.into_iter().next().expect("len == 1 checked above"));
        }

        let _permit = self.acquire_dispatch();
        self.encoding_guard();

        struct Compiled {
            pipeline: wgpu::ComputePipeline,
            bg: wgpu::BindGroup,
            workgroups: (u32, u32, u32),
        }

        let mut compiled = Vec::with_capacity(descs.len());
        for desc in &descs {
            let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> = desc
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
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(desc.label),
                    entries: &bgl_entries,
                });

            let bg_entries: Vec<wgpu::BindGroupEntry<'_>> = desc
                .bindings
                .iter()
                .map(|b| wgpu::BindGroupEntry {
                    binding: b.index,
                    resource: b.buffer.as_entire_binding(),
                })
                .collect();

            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(desc.label),
                layout: &bgl,
                entries: &bg_entries,
            });

            let module = if desc.df64_shader {
                self.compile_shader_df64(desc.shader_source, Some(desc.label))
            } else if desc.f64_shader {
                self.compile_shader_f64(desc.shader_source, Some(desc.label))
            } else {
                self.compile_shader(desc.shader_source, Some(desc.label))
            };

            let pl = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(desc.label),
                    bind_group_layouts: &[&bgl],
                    immediate_size: 0,
                });

            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(desc.label),
                    layout: Some(&pl),
                    module: &module,
                    entry_point: Some(desc.entry_point),
                    cache: self.pipeline_cache(),
                    compilation_options: Default::default(),
                });

            compiled.push(Compiled {
                pipeline,
                bg,
                workgroups: desc.workgroups,
            });
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("batched_dispatch"),
            });

        for c in &compiled {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&c.pipeline);
            pass.set_bind_group(0, Some(&c.bg), &[]);
            pass.dispatch_workgroups(c.workgroups.0, c.workgroups.1, c.workgroups.2);
        }

        let commands = encoder.finish();
        self.encoding_complete();
        self.submit_and_poll_inner(Some(commands));
        Ok(())
    }
}
