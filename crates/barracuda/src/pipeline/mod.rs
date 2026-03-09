// SPDX-License-Identifier: AGPL-3.0-or-later
//! Multi-Kernel Pipeline — GPU Operation Chaining Without CPU Round-Trips
//!
//! Enables chaining GPU operations where the output buffer of one operation
//! becomes the input buffer of the next, without CPU involvement.
//!
//! **Problem**: Traditional pattern requires CPU readback between dependent ops:
//!
//! ```text
//! Current:  H-build → [readback] → eigensolve → [readback] → BCS → [readback] → density
//! ```
//!
//! **Solution**: Chain ops with buffer handles, single GPU submit:
//!
//! ```text
//! Target:   H-build ──────────────> eigensolve ────────────> BCS ────────────> density
//!                    (GPU buffer)              (GPU buffer)       (GPU buffer)
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! let pipeline = PipelineBuilder::new(&device)
//!     .create_buffer("input", BufferSpec::f64(1000))
//!     .create_buffer("intermediate", BufferSpec::f64(500))
//!     .create_buffer("output", BufferSpec::f64(100))
//!     .add_stage(Stage {
//!         name: "transform",
//!         pipeline: &transform_pipeline,
//!         bind_group_layout: &transform_bgl,
//!         inputs: &["input"],
//!         outputs: &["intermediate"],
//!         workgroups: (4, 1, 1),
//!     })
//!     .add_stage(Stage {
//!         name: "reduce",
//!         pipeline: &reduce_pipeline,
//!         bind_group_layout: &reduce_bgl,
//!         inputs: &["intermediate"],
//!         outputs: &["output"],
//!         workgroups: (1, 1, 1),
//!     })
//!     .build()?;
//!
//! // Upload initial data
//! pipeline.write_buffer("input", &input_data)?;
//!
//! // Execute all stages with single GPU submit
//! pipeline.execute()?;
//!
//! // Read final result
//! let result = pipeline.read_buffer("output")?;
//! ```
//!
//! # Deep Debt Principles
//!
//! - Zero CPU↔GPU round-trips during pipeline execution
//! - Safe Rust (no unsafe code)
//! - Composable stages
//! - Capability-based buffer management

pub mod batched_stateful;
pub mod gpu_view;
pub mod reduce;
pub mod stateful;

pub use gpu_view::{GpuView, GpuViewElement, GpuViewF32, GpuViewF64, GpuViewU32};
pub use reduce::ReduceScalarPipeline;
pub use stateful::{PipelineStage, StatefulPipeline, WaterBalanceState};

use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use crate::utils::chunk_to_array;
use bytemuck;
use std::collections::HashMap;
use std::sync::Arc;

/// Buffer specification for pipeline allocation
#[derive(Debug, Clone)]
pub struct BufferSpec {
    /// Size in bytes
    pub size: u64,
    /// Buffer usage flags
    pub usage: wgpu::BufferUsages,
    /// Label for debugging
    pub label: Option<String>,
}

impl BufferSpec {
    /// Create a buffer spec for f64 array
    #[must_use]
    pub fn f64(count: usize) -> Self {
        Self {
            size: (count * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            label: None,
        }
    }

    /// Create a buffer spec for f32 array
    #[must_use]
    pub fn f32(count: usize) -> Self {
        Self {
            size: (count * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            label: None,
        }
    }

    /// Create a buffer spec for u32 array
    #[must_use]
    pub fn u32(count: usize) -> Self {
        Self {
            size: (count * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            label: None,
        }
    }

    /// Create a buffer spec with explicit size
    #[must_use]
    pub fn bytes(size: u64) -> Self {
        Self {
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            label: None,
        }
    }

    /// Set label for debugging
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Add uniform usage (for parameter buffers)
    #[must_use]
    pub fn with_uniform(mut self) -> Self {
        self.usage |= wgpu::BufferUsages::UNIFORM;
        self
    }
}

/// Stage definition for a compute pipeline stage
pub struct Stage {
    /// Stage name for debugging
    pub name: String,
    /// Compute pipeline (shared ownership)
    pub pipeline: Arc<wgpu::ComputePipeline>,
    /// Bind group layout (shared ownership)
    pub bind_group_layout: Arc<wgpu::BindGroupLayout>,
    /// Input buffer names (in binding order)
    pub inputs: Vec<String>,
    /// Output buffer names (in binding order, after inputs)
    pub outputs: Vec<String>,
    /// Workgroup dispatch dimensions
    pub workgroups: (u32, u32, u32),
}

impl Stage {
    /// Create a new stage
    pub fn new(
        name: impl Into<String>,
        pipeline: Arc<wgpu::ComputePipeline>,
        bind_group_layout: Arc<wgpu::BindGroupLayout>,
    ) -> Self {
        Self {
            name: name.into(),
            pipeline,
            bind_group_layout,
            inputs: Vec::new(),
            outputs: Vec::new(),
            workgroups: (1, 1, 1),
        }
    }

    /// Add input buffer names
    #[must_use]
    pub fn with_inputs(mut self, inputs: &[&str]) -> Self {
        self.inputs = inputs
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
        self
    }

    /// Add output buffer names
    #[must_use]
    pub fn with_outputs(mut self, outputs: &[&str]) -> Self {
        self.outputs = outputs
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
        self
    }

    /// Set workgroup dimensions
    #[must_use]
    pub fn with_workgroups(mut self, x: u32, y: u32, z: u32) -> Self {
        self.workgroups = (x, y, z);
        self
    }
}

/// A stage ready for execution (with pre-built bind group)
struct CompiledStage {
    name: String,
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group: wgpu::BindGroup,
    workgroups: (u32, u32, u32),
}

/// Builder for constructing multi-kernel pipelines
pub struct PipelineBuilder {
    device: Arc<WgpuDevice>,
    buffer_specs: HashMap<String, BufferSpec>,
    stages: Vec<Stage>,
}

impl PipelineBuilder {
    /// Create a new pipeline builder
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self {
            device,
            buffer_specs: HashMap::new(),
            stages: Vec::new(),
        }
    }

    /// Create a named buffer for the pipeline
    #[must_use]
    pub fn create_buffer(mut self, name: &str, spec: BufferSpec) -> Self {
        self.buffer_specs.insert(name.to_string(), spec);
        self
    }

    /// Add a compute stage to the pipeline
    #[must_use]
    pub fn add_stage(mut self, stage: Stage) -> Self {
        self.stages.push(stage);
        self
    }

    /// Build the pipeline
    /// # Errors
    /// Returns [`Err`] if any stage references an unknown input or output buffer
    /// name. Buffer allocation is infallible in wgpu (OOM surfaces as device lost).
    pub fn build(self) -> Result<ComputePipeline> {
        // Allocate all buffers
        let mut buffers = HashMap::new();
        for (name, spec) in &self.buffer_specs {
            let label = spec
                .label
                .clone()
                .unwrap_or_else(|| format!("Pipeline:{name}"));
            let buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&label),
                size: spec.size,
                usage: spec.usage,
                mapped_at_creation: false,
            });
            buffers.insert(name.clone(), Arc::new(buffer));
        }

        // Build stages with bind groups
        let mut compiled_stages = Vec::new();
        for stage in self.stages {
            // Validate all buffers exist
            for input in &stage.inputs {
                if !buffers.contains_key(input) {
                    return Err(BarracudaError::InvalidInput {
                        message: format!(
                            "Stage '{}' references unknown input buffer '{}'",
                            stage.name, input
                        ),
                    });
                }
            }
            for output in &stage.outputs {
                if !buffers.contains_key(output) {
                    return Err(BarracudaError::InvalidInput {
                        message: format!(
                            "Stage '{}' references unknown output buffer '{}'",
                            stage.name, output
                        ),
                    });
                }
            }

            // Build bind group entries
            let mut entries = Vec::new();
            let mut binding = 0u32;

            // Inputs first
            for input in &stage.inputs {
                entries.push(wgpu::BindGroupEntry {
                    binding,
                    resource: buffers[input].as_entire_binding(),
                });
                binding += 1;
            }

            // Then outputs
            for output in &stage.outputs {
                entries.push(wgpu::BindGroupEntry {
                    binding,
                    resource: buffers[output].as_entire_binding(),
                });
                binding += 1;
            }

            let bind_group = self
                .device
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("Pipeline:{}:BG", stage.name)),
                    layout: &stage.bind_group_layout,
                    entries: &entries,
                });

            compiled_stages.push(CompiledStage {
                name: stage.name,
                pipeline: stage.pipeline,
                bind_group,
                workgroups: stage.workgroups,
            });
        }

        Ok(ComputePipeline {
            device: self.device,
            buffers,
            stages: compiled_stages,
        })
    }
}

/// An executable multi-kernel compute pipeline
pub struct ComputePipeline {
    device: Arc<WgpuDevice>,
    buffers: HashMap<String, Arc<wgpu::Buffer>>,
    stages: Vec<CompiledStage>,
}

impl ComputePipeline {
    /// Write data to a pipeline buffer
    /// # Errors
    /// Returns [`Err`] if the buffer name is unknown.
    pub fn write_f64(&self, name: &str, data: &[f64]) -> Result<()> {
        let buffer = self
            .buffers
            .get(name)
            .ok_or_else(|| BarracudaError::InvalidInput {
                message: format!("Unknown buffer: {name}"),
            })?;

        let bytes: &[u8] = bytemuck::cast_slice(data);
        self.device.queue.write_buffer(buffer, 0, bytes);
        Ok(())
    }

    /// Write f32 data to a pipeline buffer
    /// # Errors
    /// Returns [`Err`] if the buffer name is unknown.
    pub fn write_f32(&self, name: &str, data: &[f32]) -> Result<()> {
        let buffer = self
            .buffers
            .get(name)
            .ok_or_else(|| BarracudaError::InvalidInput {
                message: format!("Unknown buffer: {name}"),
            })?;

        let bytes: &[u8] = bytemuck::cast_slice(data);
        self.device.queue.write_buffer(buffer, 0, bytes);
        Ok(())
    }

    /// Write raw bytes to a pipeline buffer
    /// # Errors
    /// Returns [`Err`] if the buffer name is unknown.
    pub fn write_bytes(&self, name: &str, data: &[u8]) -> Result<()> {
        let buffer = self
            .buffers
            .get(name)
            .ok_or_else(|| BarracudaError::InvalidInput {
                message: format!("Unknown buffer: {name}"),
            })?;

        self.device.queue.write_buffer(buffer, 0, data);
        Ok(())
    }

    /// Execute all pipeline stages with a single GPU submit
    /// # Errors
    /// Returns [`Err`] if command encoder creation fails, submit fails, or the
    /// device is lost (e.g. during poll).
    pub fn execute(&self) -> Result<()> {
        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("Pipeline Encoder"),
            });

        for stage in &self.stages {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&stage.name),
                timestamp_writes: None,
            });
            pass.set_pipeline(&stage.pipeline);
            pass.set_bind_group(0, Some(&stage.bind_group), &[]);
            pass.dispatch_workgroups(stage.workgroups.0, stage.workgroups.1, stage.workgroups.2);
        }

        self.device.submit_and_poll(Some(encoder.finish()));
        Ok(())
    }

    /// Read f64 data from a pipeline buffer
    /// # Errors
    /// Returns [`Err`] if the buffer name is unknown, if buffer mapping fails
    /// (e.g. device lost), or if the mapping channel is closed before completion.
    /// # Panics
    /// Panics if the buffer size is not a multiple of 8 bytes (internal invariant
    /// violation; should not occur with correctly-sized f64 buffers).
    pub fn read_f64(&self, name: &str) -> Result<Vec<f64>> {
        let buffer = self
            .buffers
            .get(name)
            .ok_or_else(|| BarracudaError::InvalidInput {
                message: format!("Unknown buffer: {name}"),
            })?;

        let size = buffer.size() as usize;
        let staging = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pipeline Read Staging"),
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("Pipeline Read"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size as u64);
        self.device.submit_and_poll(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll_safe()?;
        receiver
            .recv()
            .map_err(|_| BarracudaError::execution_failed("GPU buffer mapping channel closed"))?
            .map_err(|e| BarracudaError::execution_failed(e.to_string()))?;

        let data = slice.get_mapped_range();
        let result: Vec<f64> = data
            .chunks_exact(8)
            .map(|chunk| chunk_to_array::<8>(chunk).map(f64::from_le_bytes))
            .collect::<Result<Vec<_>>>()?;
        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Read f32 data from a pipeline buffer
    /// # Errors
    /// Returns [`Err`] if the buffer name is unknown, if buffer mapping fails
    /// (e.g. device lost), or if the mapping channel is closed before completion.
    /// # Panics
    /// Panics if the buffer size is not a multiple of 4 bytes (internal invariant
    /// violation; should not occur with correctly-sized f32 buffers).
    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>> {
        let buffer = self
            .buffers
            .get(name)
            .ok_or_else(|| BarracudaError::InvalidInput {
                message: format!("Unknown buffer: {name}"),
            })?;

        let size = buffer.size() as usize;
        let staging = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pipeline Read Staging"),
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("Pipeline Read"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size as u64);
        self.device.submit_and_poll(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll_safe()?;
        receiver
            .recv()
            .map_err(|_| BarracudaError::execution_failed("GPU buffer mapping channel closed"))?
            .map_err(|e| BarracudaError::execution_failed(e.to_string()))?;

        let data = slice.get_mapped_range();
        let result: Vec<f32> = data
            .chunks_exact(4)
            .map(|chunk| chunk_to_array::<4>(chunk).map(f32::from_le_bytes))
            .collect::<Result<Vec<_>>>()?;
        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Get a buffer reference (for use with external bind groups)
    #[must_use]
    pub fn buffer(&self, name: &str) -> Option<&wgpu::Buffer> {
        self.buffers.get(name).map(std::convert::AsRef::as_ref)
    }

    /// Get stage count
    #[must_use]
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Get buffer count
    #[must_use]
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full integration tests require actual compute pipelines.
    // These tests verify the builder API and buffer management.

    #[test]
    fn test_buffer_spec_f64() {
        let spec = BufferSpec::f64(100);
        assert_eq!(spec.size, 800); // 100 * 8 bytes
    }

    #[test]
    fn test_buffer_spec_f32() {
        let spec = BufferSpec::f32(100);
        assert_eq!(spec.size, 400); // 100 * 4 bytes
    }

    #[test]
    fn test_buffer_spec_with_label() {
        let spec = BufferSpec::f64(10).with_label("my_buffer");
        assert_eq!(spec.label, Some("my_buffer".to_string()));
    }

    #[test]
    fn test_buffer_spec_with_uniform() {
        let spec = BufferSpec::f64(10).with_uniform();
        assert!(spec.usage.contains(wgpu::BufferUsages::UNIFORM));
        assert!(spec.usage.contains(wgpu::BufferUsages::STORAGE));
    }
}
