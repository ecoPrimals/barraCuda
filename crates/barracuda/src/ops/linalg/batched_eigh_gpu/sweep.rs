// SPDX-License-Identifier: AGPL-3.0-only
//! Core sweep pass execution for Jacobi eigensolver

use crate::device::WgpuDevice;
use std::sync::Arc;

/// Execute a single sweep sub-pass (compute angles, rotate A, update blocks, or rotate V)
pub(crate) fn run_sweep_pass(
    device: &Arc<WgpuDevice>,
    sweep_bg: &wgpu::BindGroup,
    pipeline: &wgpu::ComputePipeline,
    dispatch: (u32, u32, u32),
) {
    let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
        label: Some("Jacobi sweep pass"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Jacobi sweep sub-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, Some(sweep_bg), &[]);
        pass.dispatch_workgroups(dispatch.0, dispatch.1, dispatch.2);
    }
    device.submit_and_poll(Some(encoder.finish()));
}
