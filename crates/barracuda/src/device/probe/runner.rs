// SPDX-License-Identifier: AGPL-3.0-only
//! Core probe runner — compile, dispatch, validate
//!
//! Runs a single probe shader and returns whether it produced the expected result.

use super::probes::ProbeShader;
use crate::device::WgpuDevice;

/// Run a single probe shader, catching compilation and dispatch errors.
///
/// Returns `true` if the shader compiled, dispatched, and produced the expected
/// numeric result. Returns `false` on any failure.
pub(super) async fn run_single_probe(wgpu_device: &WgpuDevice, probe: &ProbeShader) -> bool {
    let device = wgpu_device.device();
    // Phase 1: shader compilation
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(probe.name),
        source: wgpu::ShaderSource::Wgsl(probe.wgsl.into()),
    });
    if scope.pop().await.is_some() {
        return false;
    }

    // Phase 2: pipeline and buffers
    let scope2 = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe_out"),
        size: 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe_staging"),
        size: 8,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(probe.name),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("probe"),
        cache: None,
        compilation_options: Default::default(),
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: out_buf.as_entire_binding(),
        }],
    });

    if scope2.pop().await.is_some() {
        return false;
    }

    // Phase 3: dispatch
    let scope3 = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, 8);
    wgpu_device.submit_and_poll_inner(Some(enc.finish()));

    if scope3.pop().await.is_some() {
        return false;
    }

    // Phase 4: read and validate numeric result
    let slice = staging.slice(..);
    let (tx, rx) =
        std::sync::mpsc::sync_channel::<std::result::Result<(), wgpu::BufferAsyncError>>(1);
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let poll_ok = wgpu_device.poll_safe().is_ok();

    if !poll_ok || rx.recv().ok().and_then(std::result::Result::ok).is_none() {
        return false;
    }

    let bytes = slice.get_mapped_range();
    let result = f64::from_le_bytes(bytes[0..8].try_into().unwrap_or([0u8; 8]));
    drop(bytes);

    (result - probe.expected).abs() < probe.tolerance
}
