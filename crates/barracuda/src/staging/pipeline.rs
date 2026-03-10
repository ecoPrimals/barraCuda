// SPDX-License-Identifier: AGPL-3.0-only
//! metalForge Streaming Pipeline Builder
//!
//! Composes GPU compute stages into a topology that executes as a single
//! GPU command submission without CPU readback between stages.
//!
//! # Design (hotSpring Forge v0.3–v0.5)
//!
//! ```text
//! Stage A ──→ Stage B ──→ Stage C
//!   (FFT)       (filter)     (IFFT)
//!   output A = input B, output B = input C
//!   Single command encoder, zero CPU round-trips
//! ```

use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use std::sync::Arc;

/// A single compute stage in a streaming pipeline.
pub struct Stage {
    label: String,
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group: Arc<wgpu::BindGroup>,
    workgroups: (u32, u32, u32),
}

impl Stage {
    /// Create a new compute stage with the given pipeline, bind group, and workgroup dimensions.
    pub fn new(
        label: impl Into<String>,
        pipeline: Arc<wgpu::ComputePipeline>,
        bind_group: Arc<wgpu::BindGroup>,
        workgroups: (u32, u32, u32),
    ) -> Self {
        Self {
            label: label.into(),
            pipeline,
            bind_group,
            workgroups,
        }
    }

    /// Get the stage label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Get the workgroup dimensions.
    #[must_use]
    pub fn workgroups(&self) -> (u32, u32, u32) {
        self.workgroups
    }
}

/// Intermediate buffer connecting two stages.
/// The output of stage N is the input of stage N+1.
pub struct StageLink {
    buffer: wgpu::Buffer,
    size: u64,
    label: String,
}

impl StageLink {
    /// Create a new stage link connecting two pipeline stages.
    pub fn new(label: impl Into<String>, buffer: wgpu::Buffer, size: u64) -> Self {
        Self {
            buffer,
            size,
            label: label.into(),
        }
    }

    /// Get the underlying buffer.
    #[must_use]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Get the buffer size in bytes.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Get the link label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }
}

/// Builder for constructing a streaming pipeline topology.
pub struct PipelineBuilder {
    device: Arc<WgpuDevice>,
    stages: Vec<Stage>,
    label: String,
}

impl PipelineBuilder {
    /// Create a new pipeline builder for the given device.
    pub fn new(device: Arc<WgpuDevice>, label: impl Into<String>) -> Self {
        Self {
            device,
            stages: Vec::new(),
            label: label.into(),
        }
    }

    /// Add a stage to the pipeline.
    #[must_use]
    pub fn stage(mut self, stage: Stage) -> Self {
        self.stages.push(stage);
        self
    }

    /// Build the pipeline topology.
    /// # Errors
    /// Returns [`Err`] if no stages have been added.
    pub fn build(self) -> Result<StreamingPipeline> {
        if self.stages.is_empty() {
            return Err(BarracudaError::InvalidInput {
                message: "StreamingPipeline: at least one stage required".into(),
            });
        }
        Ok(StreamingPipeline {
            device: self.device,
            stages: self.stages,
            label: self.label,
        })
    }
}

/// A compiled streaming pipeline ready for execution.
/// All stages execute in a single GPU command encoder submission.
pub struct StreamingPipeline {
    device: Arc<WgpuDevice>,
    stages: Vec<Stage>,
    label: String,
}

impl StreamingPipeline {
    /// Execute all stages as a single GPU submission.
    /// No CPU readback occurs between stages.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if the GPU device is lost or if buffer allocation or
    /// GPU dispatch fails.
    pub fn execute(&self) -> Result<()> {
        if self.device.is_lost() {
            return Err(BarracudaError::device(
                "GPU device lost — cannot execute pipeline",
            ));
        }

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{}:encoder", self.label)),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{}:compute", self.label)),
                timestamp_writes: None,
            });
            for stage in &self.stages {
                pass.set_pipeline(&stage.pipeline);
                pass.set_bind_group(0, Some(&*stage.bind_group), &[]);
                pass.dispatch_workgroups(
                    stage.workgroups.0,
                    stage.workgroups.1,
                    stage.workgroups.2,
                );
            }
        }

        self.device.submit_and_poll(Some(encoder.finish()));
        Ok(())
    }

    /// Execute N iterations of the full pipeline chain.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if the GPU device is lost or if buffer allocation or
    /// GPU dispatch fails.
    pub fn execute_iterations(&self, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        if self.device.is_lost() {
            return Err(BarracudaError::device(
                "GPU device lost — cannot execute pipeline",
            ));
        }

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{}:iter_encoder", self.label)),
            });

        for _ in 0..n {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{}:iter_pass", self.label)),
                timestamp_writes: None,
            });
            for stage in &self.stages {
                pass.set_pipeline(&stage.pipeline);
                pass.set_bind_group(0, Some(&*stage.bind_group), &[]);
                pass.dispatch_workgroups(
                    stage.workgroups.0,
                    stage.workgroups.1,
                    stage.workgroups.2,
                );
            }
        }

        self.device.submit_and_poll(Some(encoder.finish()));
        Ok(())
    }

    /// Execute and read back the final stage output.
    /// Executes the pipeline once, then copies `output_buffer` to a staging
    /// buffer and maps it to read `count` elements of type `T`.
    /// # Errors
    /// Returns [`Err`] if pipeline execution fails (device lost) or buffer
    /// readback fails (e.g., `map_async` error, device lost during poll).
    pub fn execute_and_read<T: bytemuck::Pod>(
        &self,
        output_buffer: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<T>> {
        self.execute()?;
        self.device.read_buffer(output_buffer, count)
    }

    /// Get the number of stages in the pipeline.
    #[must_use]
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Get the pipeline label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool;

    const MINIMAL_PASSTHROUGH_WGSL: &str = r"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    output[i] = input[i];
}
";

    fn create_minimal_stage(device: &WgpuDevice) -> Option<Stage> {
        let shader = device.compile_shader(MINIMAL_PASSTHROUGH_WGSL, Some("test_passthrough"));
        let bind_group_layout =
            device
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
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("test_pl"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });
        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("test_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });
        let size = 256 * 4; // 256 f32s
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
        Some(Stage::new(
            "passthrough",
            Arc::new(pipeline),
            Arc::new(bind_group),
            (4, 1, 1), // 256 elements / 64 = 4 workgroups
        ))
    }

    #[test]
    fn test_pipeline_builder_empty() {
        let device = test_pool::get_test_device_sync();
        let result = PipelineBuilder::new(Arc::clone(&device), "empty").build();
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("at least one stage required"));
        }
    }

    #[test]
    fn test_pipeline_builder_stage_count() {
        let device = test_pool::get_test_device_sync();
        let Some(stage) = create_minimal_stage(&device) else {
            return;
        };
        let pipeline = PipelineBuilder::new(Arc::clone(&device), "two_stages")
            .stage(stage)
            .build()
            .expect("build with one stage should succeed");
        assert_eq!(pipeline.stage_count(), 1);
        assert_eq!(pipeline.label(), "two_stages");

        let stage2 = create_minimal_stage(&device).expect("second stage");
        let pipeline2 = PipelineBuilder::new(Arc::clone(&device), "two_stages")
            .stage(stage2)
            .stage(create_minimal_stage(&device).expect("stage"))
            .build()
            .expect("build with two stages should succeed");
        assert_eq!(pipeline2.stage_count(), 2);
    }

    #[test]
    fn test_stage_creation() {
        let device = test_pool::get_test_device_sync();
        let Some(stage) = create_minimal_stage(&device) else {
            return;
        };
        assert_eq!(stage.label(), "passthrough");
        assert_eq!(stage.workgroups(), (4, 1, 1));
    }

    #[test]
    fn test_stage_link_creation() {
        let device = test_pool::get_test_device_sync();
        let size = 256u64 * 4;
        let buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_stage_link"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let link = StageLink::new("link_a_to_b", buffer, size);
        assert_eq!(link.label(), "link_a_to_b");
        assert_eq!(link.size(), size);
        assert_eq!(link.buffer().size(), size);
    }

    #[test]
    fn test_pipeline_builder_chaining() {
        let device = test_pool::get_test_device_sync();
        let s1 = create_minimal_stage(&device).expect("stage 1");
        let s2 = create_minimal_stage(&device).expect("stage 2");
        let s3 = create_minimal_stage(&device).expect("stage 3");
        let pipeline = PipelineBuilder::new(Arc::clone(&device), "chained")
            .stage(s1)
            .stage(s2)
            .stage(s3)
            .build()
            .expect("build with three stages");
        assert_eq!(pipeline.stage_count(), 3);
    }

    #[test]
    fn test_pipeline_label() {
        let device = test_pool::get_test_device_sync();
        let stage = create_minimal_stage(&device).expect("stage");
        let pipeline = PipelineBuilder::new(Arc::clone(&device), "my_custom_label")
            .stage(stage)
            .build()
            .expect("build");
        assert_eq!(pipeline.label(), "my_custom_label");
    }
}
