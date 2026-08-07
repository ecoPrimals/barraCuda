// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cyclical learning rate operation
//!
//! Cycles learning rate between bounds for better convergence
//! Reference: "Cyclical Learning Rates for Training Neural Networks" by Smith (2017)

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CyclicalLrParams {
    current_iter: u32,
    step_size: u32,
    base_lr: f32,
    max_lr: f32,
    mode: u32,
    gamma: f32,
    _padding: [u32; 2],
}

/// Cyclical learning rate operation
pub struct CyclicalLr {
    current_iter: u32,
    step_size: u32,
    base_lr: f32,
    max_lr: f32,
    mode: CyclicalLrMode,
    gamma: f32,
}

/// Cyclical learning rate mode
#[derive(Copy, Clone, Debug)]
pub enum CyclicalLrMode {
    /// Linear cycle between base and max LR.
    Triangular = 0,
    /// Halve max LR each cycle.
    Triangular2 = 1,
    /// Exponentially decaying range.
    ExpRange = 2,
}

impl CyclicalLr {
    /// Create cyclical learning rate operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(
        current_iter: u32,
        step_size: u32,
        base_lr: f32,
        max_lr: f32,
        mode: CyclicalLrMode,
        gamma: f32,
    ) -> Result<Self> {
        if step_size == 0 {
            return Err(BarracudaError::invalid_op(
                "cyclical_lr",
                "step_size must be greater than 0",
            ));
        }

        if base_lr < 0.0 || max_lr < 0.0 {
            return Err(BarracudaError::invalid_op(
                "cyclical_lr",
                "base_lr and max_lr must be non-negative",
            ));
        }

        if base_lr > max_lr {
            return Err(BarracudaError::invalid_op(
                "cyclical_lr",
                format!("base_lr {base_lr} must be <= max_lr {max_lr}"),
            ));
        }

        if matches!(mode, CyclicalLrMode::ExpRange) && gamma <= 0.0 {
            return Err(BarracudaError::invalid_op(
                "cyclical_lr",
                "gamma must be positive for ExpRange mode",
            ));
        }

        Ok(Self {
            current_iter,
            step_size,
            base_lr,
            max_lr,
            mode,
            gamma,
        })
    }

    /// Execute cyclical learning rate computation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self, device: &crate::device::WgpuDevice) -> Result<Tensor> {
        let output_buffer = device.create_buffer_f32(1)?;

        let params = CyclicalLrParams {
            current_iter: self.current_iter,
            step_size: self.step_size,
            base_lr: self.base_lr,
            max_lr: self.max_lr,
            mode: self.mode as u32,
            gamma: self.gamma,
            _padding: [0; 2],
        };

        let params_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CyclicalLr Params"),
            size: std::mem::size_of::<CyclicalLrParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        ComputeDispatch::new(device, "CyclicalLr")
            .shader(include_str!("../shaders/optimizer/cyclical_lr_f64.wgsl"), "main")
            .storage_rw(0, &output_buffer)
            .uniform(1, &params_buffer)
            .dispatch(1, 1, 1)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![1],
            std::sync::Arc::new(device.clone()),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_triangular_succeeds() {
        assert!(CyclicalLr::new(0, 100, 0.001, 0.01, CyclicalLrMode::Triangular, 0.9).is_ok());
    }

    #[test]
    fn new_triangular2_succeeds() {
        assert!(CyclicalLr::new(0, 50, 0.0, 0.1, CyclicalLrMode::Triangular2, 0.9).is_ok());
    }

    #[test]
    fn new_exp_range_succeeds() {
        assert!(CyclicalLr::new(0, 200, 0.001, 0.01, CyclicalLrMode::ExpRange, 0.99).is_ok());
    }

    #[test]
    fn rejects_zero_step_size() {
        assert!(CyclicalLr::new(0, 0, 0.001, 0.01, CyclicalLrMode::Triangular, 0.9).is_err());
    }

    #[test]
    fn rejects_negative_base_lr() {
        assert!(CyclicalLr::new(0, 100, -0.001, 0.01, CyclicalLrMode::Triangular, 0.9).is_err());
    }

    #[test]
    fn rejects_negative_max_lr() {
        assert!(CyclicalLr::new(0, 100, 0.001, -0.01, CyclicalLrMode::Triangular, 0.9).is_err());
    }

    #[test]
    fn rejects_base_lr_greater_than_max_lr() {
        assert!(CyclicalLr::new(0, 100, 0.01, 0.001, CyclicalLrMode::Triangular, 0.9).is_err());
    }

    #[test]
    fn rejects_exp_range_with_zero_gamma() {
        assert!(CyclicalLr::new(0, 100, 0.001, 0.01, CyclicalLrMode::ExpRange, 0.0).is_err());
    }

    #[test]
    fn rejects_exp_range_with_negative_gamma() {
        assert!(CyclicalLr::new(0, 100, 0.001, 0.01, CyclicalLrMode::ExpRange, -1.0).is_err());
    }

    #[test]
    fn allows_equal_lr_range() {
        assert!(CyclicalLr::new(0, 100, 0.01, 0.01, CyclicalLrMode::Triangular, 0.9).is_ok());
    }
}
