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
//! backends (e.g. `SovereignDevice`) work through the same builder.
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

    /// Build the dispatch descriptor without submitting.
    ///
    /// Use this with [`BatchedComputeDispatch`] to record multiple dispatches
    /// into a single GPU submission, reducing per-dispatch overhead (~1.8x on
    /// Vulkan for multi-kernel pipelines like MD force+kick+drift).
    ///
    /// # Errors
    /// Returns [`Err`] if shader source was not set via [`shader`](Self::shader).
    pub fn build(self) -> Result<DispatchDescriptor<'a, B>> {
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

        Ok(DispatchDescriptor {
            label: self.label,
            shader_source: source,
            entry_point: self.entry_point,
            bindings,
            workgroups: self.workgroups,
            f64_shader: self.f64_shader,
            df64_shader: self.df64_shader,
            hardware_hint: crate::device::backend::HardwareHint::default(),
        })
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
        let device = self.device;
        let desc = self.build()?;
        device.dispatch_compute(desc)
    }
}

/// Batched compute dispatch — multiple kernels in a single GPU submission.
///
/// Reduces per-dispatch overhead by recording all compute passes into one
/// command encoder before submitting. On Vulkan this avoids repeated
/// queue submissions (~1.8x overhead measured by hotSpring on RTX 3090).
///
/// Backends that don't support batching (e.g. sovereign/DRM) fall back to
/// sequential `dispatch_compute` calls via the default trait implementation.
///
/// # Example
///
/// ```ignore
/// let mut batch = BatchedComputeDispatch::new(&device);
/// batch.push(
///     ComputeDispatch::new(&device, "force")
///         .shader(FORCE_SHADER, "main").f64()
///         .storage_read(0, &pos).storage_rw(1, &force)
///         .dispatch_1d(n)
/// )?;
/// batch.push(
///     ComputeDispatch::new(&device, "kick_drift")
///         .shader(KD_SHADER, "main").f64()
///         .storage_rw(0, &pos).storage_rw(1, &vel)
///         .dispatch_1d(n)
/// )?;
/// batch.submit()?;
/// ```
pub struct BatchedComputeDispatch<'a, B: GpuBackend = WgpuDevice> {
    device: &'a B,
    descriptors: Vec<DispatchDescriptor<'a, B>>,
}

impl<'a, B: GpuBackend> BatchedComputeDispatch<'a, B> {
    /// Create a new batched dispatch for the given device.
    #[must_use]
    pub fn new(device: &'a B) -> Self {
        Self {
            device,
            descriptors: Vec::new(),
        }
    }

    /// Add a dispatch to the batch.
    ///
    /// The dispatch is built (validated) immediately but not submitted until
    /// [`submit`](Self::submit) is called.
    ///
    /// # Errors
    /// Returns [`Err`] if the dispatch builder is missing a shader source.
    pub fn push(&mut self, dispatch: ComputeDispatch<'a, B>) -> Result<()> {
        self.descriptors.push(dispatch.build()?);
        Ok(())
    }

    /// Number of dispatches queued so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    /// Whether the batch is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }

    /// Submit all queued dispatches in a single GPU submission.
    ///
    /// On backends that support batching (e.g. wgpu/Vulkan), all compute
    /// passes are recorded into one command encoder. On backends without
    /// batching support, dispatches are executed sequentially.
    ///
    /// # Errors
    /// Returns [`Err`] if any dispatch fails (compilation, dispatch, or sync).
    pub fn submit(self) -> Result<()> {
        self.device.dispatch_compute_batch(self.descriptors)
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

/// Declarative bind-group layout builder for reusable compute pipelines.
///
/// Many ops create identical bind-group layouts every dispatch. This builder
/// lets callers declare the layout once and cache the resulting
/// `wgpu::BindGroupLayout` for reuse across dispatches.
///
/// # Example
///
/// ```ignore
/// use barracuda::device::compute_pipeline::BglBuilder;
///
/// let bgl = BglBuilder::new("my_op")
///     .storage_read(0)
///     .storage_rw(1)
///     .uniform(2)
///     .build(&device.device);
/// ```
pub struct BglBuilder {
    label: &'static str,
    entries: Vec<wgpu::BindGroupLayoutEntry>,
}

impl BglBuilder {
    /// Create a new builder with the given label.
    #[must_use]
    pub fn new(label: &'static str) -> Self {
        Self {
            label,
            entries: Vec::new(),
        }
    }

    /// Add a read-only storage buffer binding at `index`.
    #[must_use]
    pub fn storage_read(mut self, index: u32) -> Self {
        self.entries.push(storage_bgl_entry(index, true));
        self
    }

    /// Add a read-write storage buffer binding at `index`.
    #[must_use]
    pub fn storage_rw(mut self, index: u32) -> Self {
        self.entries.push(storage_bgl_entry(index, false));
        self
    }

    /// Add a uniform buffer binding at `index`.
    #[must_use]
    pub fn uniform(mut self, index: u32) -> Self {
        self.entries.push(uniform_bgl_entry(index));
        self
    }

    /// Add an arbitrary `BindGroupLayoutEntry` for advanced use cases.
    #[must_use]
    pub fn entry(mut self, entry: wgpu::BindGroupLayoutEntry) -> Self {
        self.entries.push(entry);
        self
    }

    /// Number of entries declared so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no entries have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Build the `BindGroupLayout` on the given device.
    #[must_use]
    pub fn build(self, device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(self.label),
            entries: &self.entries,
        })
    }
}
