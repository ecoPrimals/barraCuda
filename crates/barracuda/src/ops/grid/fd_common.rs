//! Finite-Difference Common Utilities
//!
//! **Smart Refactoring**: Shared infrastructure for all FD operators
//!
//! Reduces code duplication across gradient and Laplacian operators by
//! providing common patterns for bind group layouts, buffer creation,
//! and staging readback.

use crate::device::WgpuDevice;
use crate::error::Result;
use std::sync::Arc;

/// Common shader source for all FD operations
pub const FD_SHADER_SOURCE: &str = include_str!("../../shaders/grid/fd_gradient_f64.wgsl");

/// Standard workgroup size for FD operations
pub const FD_WORKGROUP_SIZE: u32 = 256;

/// Create a uniform buffer bind group layout entry
pub fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

/// Create a read-only storage buffer bind group layout entry
pub fn storage_readonly_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Create a read-write storage buffer bind group layout entry
pub fn storage_readwrite_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Create an f64 GPU buffer from a slice
pub fn create_f64_buffer(
    device: &wgpu::Device,
    data: &[f64],
    label: &str,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: &bytes,
        usage,
    })
}

/// Create an empty f64 GPU buffer
pub fn create_empty_f64_buffer(
    device: &wgpu::Device,
    count: usize,
    label: &str,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (count * 8) as u64,
        usage,
        mapped_at_creation: false,
    })
}

/// Create a staging buffer for readback
pub fn create_staging_buffer(device: &wgpu::Device, size_bytes: u64, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Builder for FD compute pipelines
///
/// Reduces boilerplate when creating gradient/Laplacian operators.
pub struct FdPipelineBuilder<'a> {
    device: &'a wgpu::Device,
    shader_source: &'a str,
    entries: Vec<wgpu::BindGroupLayoutEntry>,
    label_prefix: &'a str,
}

impl<'a> FdPipelineBuilder<'a> {
    /// Create a new pipeline builder
    pub fn new(device: &'a wgpu::Device, label_prefix: &'a str) -> Self {
        Self {
            device,
            shader_source: FD_SHADER_SOURCE,
            entries: Vec::new(),
            label_prefix,
        }
    }

    /// Add a uniform buffer binding
    #[must_use]
    pub fn with_uniform(mut self, binding: u32) -> Self {
        self.entries.push(uniform_entry(binding));
        self
    }

    /// Add a read-only storage buffer binding (input)
    #[must_use]
    pub fn with_input(mut self, binding: u32) -> Self {
        self.entries.push(storage_readonly_entry(binding));
        self
    }

    /// Add a read-write storage buffer binding (output)
    #[must_use]
    pub fn with_output(mut self, binding: u32) -> Self {
        self.entries.push(storage_readwrite_entry(binding));
        self
    }

    /// Build the pipeline with the specified entry point
    pub fn build(
        self,
        entry_point: &str,
    ) -> Result<(wgpu::ComputePipeline, wgpu::BindGroupLayout)> {
        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{}_shader", self.label_prefix)),
                source: wgpu::ShaderSource::Wgsl(self.shader_source.into()),
            });

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bgl", self.label_prefix)),
                    entries: &self.entries,
                });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{}_layout", self.label_prefix)),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{}_pipeline", self.label_prefix)),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point,
                cache: None,
                compilation_options: Default::default(),
            });

        Ok((pipeline, bind_group_layout))
    }
}

/// Runner for FD compute operations
///
/// Handles the common dispatch pattern for all FD operators.
pub struct FdComputeRunner<'a> {
    device: &'a Arc<WgpuDevice>,
    pipeline: &'a wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    workgroups: u32,
    label: &'a str,
}

impl<'a> FdComputeRunner<'a> {
    /// Create a new compute runner
    pub fn new(
        device: &'a Arc<WgpuDevice>,
        pipeline: &'a wgpu::ComputePipeline,
        bind_group: wgpu::BindGroup,
        total_elements: usize,
        label: &'a str,
    ) -> Self {
        Self {
            device,
            pipeline,
            bind_group,
            workgroups: total_elements.div_ceil(FD_WORKGROUP_SIZE as usize) as u32,
            label,
        }
    }

    /// Execute the compute pass and read back a single output buffer
    pub fn execute_single(
        self,
        output_buffer: &wgpu::Buffer,
        output_count: usize,
    ) -> Result<Vec<f64>> {
        let buffer_size = (output_count * 8) as u64;

        let mut encoder = self
            .device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(self.label),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.workgroups, 1, 1);
        }

        let staging = create_staging_buffer(self.device.device(), buffer_size, "staging");
        encoder.copy_buffer_to_buffer(output_buffer, 0, &staging, 0, buffer_size);
        self.device.queue().submit(Some(encoder.finish()));

        self.device
            .map_staging_buffer::<f64>(&staging, output_count)
    }

    /// Execute the compute pass and read back two output buffers
    pub fn execute_dual(
        self,
        output_buffer_a: &wgpu::Buffer,
        output_buffer_b: &wgpu::Buffer,
        output_count: usize,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let buffer_size = (output_count * 8) as u64;

        let mut encoder = self
            .device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(self.label),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.workgroups, 1, 1);
        }

        let staging_a = create_staging_buffer(self.device.device(), buffer_size, "staging_a");
        let staging_b = create_staging_buffer(self.device.device(), buffer_size, "staging_b");
        encoder.copy_buffer_to_buffer(output_buffer_a, 0, &staging_a, 0, buffer_size);
        encoder.copy_buffer_to_buffer(output_buffer_b, 0, &staging_b, 0, buffer_size);
        self.device.queue().submit(Some(encoder.finish()));

        let result_a = self
            .device
            .map_staging_buffer::<f64>(&staging_a, output_count)?;
        let result_b = self
            .device
            .map_staging_buffer::<f64>(&staging_b, output_count)?;

        Ok((result_a, result_b))
    }
}
