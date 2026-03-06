// SPDX-License-Identifier: AGPL-3.0-or-later
//! Atomic encoder barrier and guarded device handle.
//!
//! Prevents `device.poll()` from racing with resource creation on software
//! rasterizers (llvmpipe, lavapipe). The guard pattern uses an `AtomicU32`
//! counter: encoding is lock-free, polling waits for the counter to reach 0.

use crate::error::{BarracudaError, Result};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use wgpu::util::DeviceExt;

/// RAII command encoder that prevents `device.poll()` from running.
///
/// While this encoder exists, an atomic counter is non-zero, which causes
/// poll to spin-wait. Multiple `GuardedEncoder`s can coexist with zero
/// contention (just atomic increments). Encoding is never blocked.
///
/// Call `.finish()` to consume the encoder, decrement the counter, and
/// obtain a `wgpu::CommandBuffer` ready for submission.
pub struct GuardedEncoder {
    encoder: Option<wgpu::CommandEncoder>,
    active_encoders: Arc<AtomicU32>,
}

impl std::ops::Deref for GuardedEncoder {
    type Target = wgpu::CommandEncoder;

    fn deref(&self) -> &Self::Target {
        // Invariant: encoder is Some from construction until finish() consumes self.
        // finish() takes self by value so Deref cannot be called post-finish.
        self.encoder
            .as_ref()
            .expect("GuardedEncoder::deref after finish (unreachable by ownership)")
    }
}

impl std::ops::DerefMut for GuardedEncoder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.encoder
            .as_mut()
            .expect("GuardedEncoder::deref_mut after finish (unreachable by ownership)")
    }
}

impl GuardedEncoder {
    pub(crate) fn new(encoder: wgpu::CommandEncoder, active_encoders: Arc<AtomicU32>) -> Self {
        Self {
            encoder: Some(encoder),
            active_encoders,
        }
    }

    /// Get immutable reference to the encoder.
    ///
    /// # Errors
    /// Returns [`Err`] if the encoder was already finished.
    pub fn encoder(&self) -> Result<&wgpu::CommandEncoder> {
        self.encoder
            .as_ref()
            .ok_or_else(|| BarracudaError::Internal("encoder already finished".into()))
    }

    /// Get mutable reference to the encoder.
    ///
    /// # Errors
    /// Returns [`Err`] if the encoder was already finished.
    pub fn encoder_mut(&mut self) -> Result<&mut wgpu::CommandEncoder> {
        self.encoder
            .as_mut()
            .ok_or_else(|| BarracudaError::Internal("encoder already finished".into()))
    }

    /// Finish encoding and decrement the active encoder count.
    ///
    /// # Panics
    ///
    /// Unreachable by construction: `finish` consumes `self` so it cannot be
    /// called twice.
    #[must_use]
    pub fn finish(mut self) -> wgpu::CommandBuffer {
        // Invariant: encoder is Some until take(); finish consumes self so only called once.
        let encoder = self
            .encoder
            .take()
            .expect("GuardedEncoder::finish called twice (unreachable by ownership)");
        self.active_encoders.fetch_sub(1, Ordering::Release);
        encoder.finish()
    }
}

impl Drop for GuardedEncoder {
    fn drop(&mut self) {
        if self.encoder.is_some() {
            self.active_encoders.fetch_sub(1, Ordering::Release);
        }
    }
}

/// Wrapper around `wgpu::Device` that auto-protects `create_*` calls.
///
/// wgpu 28 `Device` is internally Arc-wrapped and `Clone`, so no
/// outer `Arc` is needed. Inherent methods shadow `Device::create_*`,
/// incrementing the active-encoder counter before the call and
/// decrementing after. This prevents `device.poll()` from racing with
/// resource creation on software rasterizers (llvmpipe).
#[derive(Clone)]
pub struct GuardedDeviceHandle {
    inner: wgpu::Device,
    active_encoders: Arc<AtomicU32>,
}

impl std::fmt::Debug for GuardedDeviceHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuardedDeviceHandle")
            .field("inner", &"<wgpu::Device>")
            .finish()
    }
}

impl std::ops::Deref for GuardedDeviceHandle {
    type Target = wgpu::Device;
    fn deref(&self) -> &wgpu::Device {
        &self.inner
    }
}

impl GuardedDeviceHandle {
    pub(crate) fn new(inner: wgpu::Device, active_encoders: Arc<AtomicU32>) -> Self {
        Self {
            inner,
            active_encoders,
        }
    }

    fn guard(&self) {
        self.active_encoders.fetch_add(1, Ordering::Acquire);
    }

    fn unguard(&self) {
        self.active_encoders.fetch_sub(1, Ordering::Release);
    }

    /// Create a buffer (guarded against poll races).
    #[must_use]
    pub fn create_buffer(&self, desc: &wgpu::BufferDescriptor<'_>) -> wgpu::Buffer {
        self.guard();
        let r = self.inner.create_buffer(desc);
        self.unguard();
        r
    }

    /// Create and initialize a buffer (guarded).
    #[must_use]
    pub fn create_buffer_init(&self, desc: &wgpu::util::BufferInitDescriptor<'_>) -> wgpu::Buffer {
        self.guard();
        let r = self.inner.create_buffer_init(desc);
        self.unguard();
        r
    }

    /// Create a bind group layout (guarded).
    #[must_use]
    pub fn create_bind_group_layout(
        &self,
        desc: &wgpu::BindGroupLayoutDescriptor<'_>,
    ) -> wgpu::BindGroupLayout {
        self.guard();
        let r = self.inner.create_bind_group_layout(desc);
        self.unguard();
        r
    }

    /// Create a bind group (guarded).
    #[must_use]
    pub fn create_bind_group(&self, desc: &wgpu::BindGroupDescriptor<'_>) -> wgpu::BindGroup {
        self.guard();
        let r = self.inner.create_bind_group(desc);
        self.unguard();
        r
    }

    /// Create a pipeline layout (guarded).
    #[must_use]
    pub fn create_pipeline_layout(
        &self,
        desc: &wgpu::PipelineLayoutDescriptor<'_>,
    ) -> wgpu::PipelineLayout {
        self.guard();
        let r = self.inner.create_pipeline_layout(desc);
        self.unguard();
        r
    }

    /// Create a compute pipeline (guarded).
    #[must_use]
    pub fn create_compute_pipeline(
        &self,
        desc: &wgpu::ComputePipelineDescriptor<'_>,
    ) -> wgpu::ComputePipeline {
        self.guard();
        let r = self.inner.create_compute_pipeline(desc);
        self.unguard();
        r
    }

    /// Create a shader module (guarded).
    #[must_use]
    pub fn create_shader_module(
        &self,
        desc: wgpu::ShaderModuleDescriptor<'_>,
    ) -> wgpu::ShaderModule {
        self.guard();
        let r = self.inner.create_shader_module(desc);
        self.unguard();
        r
    }

    /// Create a command encoder (guarded).
    #[must_use]
    pub fn create_command_encoder(
        &self,
        desc: &wgpu::CommandEncoderDescriptor<'_>,
    ) -> wgpu::CommandEncoder {
        self.guard();
        let r = self.inner.create_command_encoder(desc);
        self.unguard();
        r
    }
}
