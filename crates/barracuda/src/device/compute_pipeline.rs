// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU compute pipeline builder — eliminates bind-group/pipeline boilerplate.
//!
//! The ~80 lines of `create_bind_group_layout` → `create_bind_group` →
//! `create_pipeline_layout` → `create_compute_pipeline` → `create_command_encoder` →
//! `begin_compute_pass` → `set_pipeline` → `set_bind_group` → dispatch pattern is
//! repeated in 50+ ops. This module captures that pattern as a builder.
//!
//! `ComputeDispatch` is generic over `GpuBackend`, defaulting to `WgpuDevice`.
//! Existing callers need no changes — the type parameter is inferred. Future
//! backends (e.g. `CoralReefDevice`) work through the same builder.
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
//!     .submit()?;
//! ```

use crate::device::WgpuDevice;
use crate::device::backend::{BufferBinding, DispatchDescriptor, GpuBackend};
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::error::{BarracudaError, Result};

/// Buffer binding declaration for the pipeline builder.
struct Binding<'a, B: GpuBackend> {
    index: u32,
    buffer: &'a B::Buffer,
    read_only: bool,
    is_uniform: bool,
}

/// Ergonomic one-shot compute dispatch builder.
///
/// Builds the bind-group layout, bind group, pipeline layout, pipeline,
/// command encoder, and compute pass from a fluent API, then submits.
///
/// Generic over `GpuBackend`, defaulting to `WgpuDevice`. The type parameter
/// is inferred from the `device` argument — existing callers compile unchanged.
pub struct ComputeDispatch<'a, B: GpuBackend = WgpuDevice> {
    device: &'a B,
    label: &'a str,
    bindings: Vec<Binding<'a, B>>,
    shader_source: Option<&'a str>,
    entry_point: &'a str,
    workgroups: (u32, u32, u32),
    f64_shader: bool,
    df64_shader: bool,
}

impl<'a, B: GpuBackend> ComputeDispatch<'a, B> {
    /// Creates a new compute dispatch builder for the given device and label.
    #[must_use]
    pub fn new(device: &'a B, label: &'a str) -> Self {
        Self {
            device,
            label,
            bindings: Vec::new(),
            shader_source: None,
            entry_point: "main",
            workgroups: (1, 1, 1),
            f64_shader: false,
            df64_shader: false,
        }
    }

    /// Set the WGSL shader source and entry point.
    #[must_use]
    pub fn shader(mut self, source: &'a str, entry_point: &'a str) -> Self {
        self.shader_source = Some(source);
        self.entry_point = entry_point;
        self
    }

    /// Use f64 shader compilation path (with precision patching + ILP optimizer).
    #[must_use]
    pub fn f64(mut self) -> Self {
        self.f64_shader = true;
        self
    }

    /// Use DF64 shader compilation path (f32-pair arithmetic, ~48-bit mantissa).
    ///
    /// The source must already contain DF64 bridge functions (from naga rewrite).
    /// This method prepends `df64_core` + `df64_transcendentals` and compiles.
    #[must_use]
    pub fn df64(mut self) -> Self {
        self.df64_shader = true;
        self
    }

    /// Bind a read-only storage buffer at `index`.
    #[must_use]
    pub fn storage_read(mut self, index: u32, buffer: &'a B::Buffer) -> Self {
        self.bindings.push(Binding {
            index,
            buffer,
            read_only: true,
            is_uniform: false,
        });
        self
    }

    /// Bind a read-write storage buffer at `index`.
    #[must_use]
    pub fn storage_rw(mut self, index: u32, buffer: &'a B::Buffer) -> Self {
        self.bindings.push(Binding {
            index,
            buffer,
            read_only: false,
            is_uniform: false,
        });
        self
    }

    /// Bind a uniform buffer at `index`.
    #[must_use]
    pub fn uniform(mut self, index: u32, buffer: &'a B::Buffer) -> Self {
        self.bindings.push(Binding {
            index,
            buffer,
            read_only: true,
            is_uniform: true,
        });
        self
    }

    /// Set dispatch to `ceil(n / WORKGROUP_SIZE_1D)` workgroups in x.
    #[must_use]
    pub fn dispatch_1d(mut self, element_count: u32) -> Self {
        self.workgroups = (element_count.div_ceil(WORKGROUP_SIZE_1D), 1, 1);
        self
    }

    /// Set explicit workgroup counts.
    #[must_use]
    pub fn dispatch(mut self, x: u32, y: u32, z: u32) -> Self {
        self.workgroups = (x, y, z);
        self
    }

    /// Build everything and submit the compute pass to the device queue.
    ///
    /// Delegates to [`GpuBackend::dispatch_compute`], which handles the full
    /// compile → bind → submit lifecycle using the backend's native API.
    ///
    /// # Errors
    /// Returns [`Err`] if shader source was not set via [`shader`](Self::shader),
    /// or if the backend's dispatch fails.
    pub fn submit(self) -> Result<()> {
        let source = self
            .shader_source
            .ok_or_else(|| BarracudaError::InvalidInput {
                message: "ComputeDispatch: shader source required".into(),
            })?;

        let bindings: Vec<BufferBinding<'_, B>> = self
            .bindings
            .iter()
            .map(|b| BufferBinding {
                index: b.index,
                buffer: b.buffer,
                read_only: b.read_only,
                is_uniform: b.is_uniform,
            })
            .collect();

        self.device.dispatch_compute(DispatchDescriptor {
            label: self.label,
            shader_source: source,
            entry_point: self.entry_point,
            bindings,
            workgroups: self.workgroups,
            f64_shader: self.f64_shader,
            df64_shader: self.df64_shader,
        })
    }
}

/// Helper to create a storage bind-group-layout entry.
///
/// Replaces the 7-line `BindGroupLayoutEntry` struct literal that appears
/// in every GPU op. Use for ops that need the BGL separately (e.g.,
/// cached pipelines that outlive a single dispatch).
#[must_use]
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
#[must_use]
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
