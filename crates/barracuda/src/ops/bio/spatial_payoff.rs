// SPDX-License-Identifier: AGPL-3.0-or-later

//! Spatial Prisoner's Dilemma Payoff — GPU kernel.
//!
//! Computes cumulative payoff for each cell in a 2D grid using
//! Moore neighborhood (8 neighbors) with periodic boundary conditions.
//! Grid: 1 = cooperator, 0 = defector.
//!
//! Payoff rules:
//! - Both cooperate: b - c
//! - Cooperator exploited: -c
//! - Defector exploits: b
//! - Both defect: 0
//!
//! Provenance: neuralSpring metalForge → toadStool absorption

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;

const WGSL_SPATIAL_PAYOFF: &str = include_str!("../../shaders/math/spatial_payoff_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PayoffParams {
    grid_size: u32,
    b_x1000: u32,
    c_x1000: u32,
    _pad: u32,
}

/// GPU spatial payoff (evolutionary game) computation.
pub struct SpatialPayoffGpu {
    device: Arc<WgpuDevice>,
}

impl SpatialPayoffGpu {
    /// Create spatial payoff calculator.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Compute spatial PD payoffs for a `grid_size × grid_size` grid.
    ///
    /// `grid_buf`: `[grid_size²]` u32 (0 = defector, 1 = cooperator)
    /// `fitness_buf`: `[grid_size²]` f32 (cumulative payoff)
    /// `benefit` / `cost`: PD parameters (encoded as x1000 integers internally)
    #[expect(
        clippy::missing_panics_doc,
        reason = "dispatch submit is infallible on valid device"
    )]
    pub fn dispatch(
        &self,
        grid_buf: &wgpu::Buffer,
        fitness_buf: &wgpu::Buffer,
        grid_size: u32,
        benefit: f32,
        cost: f32,
    ) {
        let d = self.device.device();

        let params = PayoffParams {
            grid_size,
            b_x1000: (benefit * 1000.0) as u32,
            c_x1000: (cost * 1000.0) as u32,
            _pad: 0,
        };
        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SpatialPayoff Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let total = grid_size * grid_size;

        ComputeDispatch::new(&self.device, "SpatialPayoff")
            .shader(WGSL_SPATIAL_PAYOFF, "spatial_payoff")
            .storage_read(0, grid_buf)
            .storage_rw(1, fitness_buf)
            .uniform(2, &params_buf)
            .dispatch_1d(total)
            .submit()
            .expect("SpatialPayoff GPU dispatch failed");
    }
}
