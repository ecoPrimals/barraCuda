// SPDX-License-Identifier: AGPL-3.0-only

//! `BatchedStatefulF64` — GPU-resident state buffer for sequential multi-step pipelines.
//!
//! Manages a persistent state buffer on GPU that carries state across
//! sequential dispatch steps (e.g., multi-day water balance, time series
//! accumulation). Each step reads the current state, computes new outputs,
//! and updates the state buffer — all without CPU readback between steps.
//!
//! The key insight is that many scientific pipelines need "day-over-day" or
//! "step-over-step" state carry: θ(t) depends on θ(t-1). Rather than
//! reading back to CPU between each step, `BatchedStatefulF64` keeps the
//! state buffer GPU-resident and provides a buffer-swap mechanism.
//!
//! Provenance: airSpring V045 (`SeasonalPipeline::GpuFused` multi-day request)

use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use bytemuck;
use std::sync::Arc;

/// GPU-resident state buffer for sequential multi-step dispatches.
///
/// Holds a pair of ping-pong buffers so that shaders can read from
/// `state_in` and write to `state_out`, then swap for the next step.
pub struct BatchedStatefulF64 {
    device: Arc<WgpuDevice>,
    state_a: wgpu::Buffer,
    state_b: wgpu::Buffer,
    /// Which buffer is currently "current" state (true = A, false = B)
    current_is_a: bool,
    n_cells: usize,
    n_state_per_cell: usize,
}

impl BatchedStatefulF64 {
    /// Create a new batched state buffer.
    /// # Arguments
    /// * `device` - GPU device
    /// * `n_cells` - Number of spatial cells / batch elements
    /// * `n_state_per_cell` - Number of f64 state values per cell
    /// * `initial_state` - Initial state values [`n_cells` * `n_state_per_cell`]
    /// # Errors
    /// Returns [`Err`] if `initial_state.len()` does not equal
    /// `n_cells * n_state_per_cell`.
    pub fn new(
        device: Arc<WgpuDevice>,
        n_cells: usize,
        n_state_per_cell: usize,
        initial_state: &[f64],
    ) -> Result<Self> {
        let expected = n_cells * n_state_per_cell;
        if initial_state.len() != expected {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "initial_state length {} != n_cells({}) * n_state_per_cell({})",
                    initial_state.len(),
                    n_cells,
                    n_state_per_cell,
                ),
            });
        }

        let bytes: &[u8] = bytemuck::cast_slice(initial_state);
        let buf_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let state_a = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("BatchedStateful:A"),
                contents: bytes,
                usage: buf_usage,
            });
        let state_b = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BatchedStateful:B"),
            size: (expected * 8) as u64,
            usage: buf_usage,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            state_a,
            state_b,
            current_is_a: true,
            n_cells,
            n_state_per_cell,
        })
    }

    /// The buffer to read current state FROM (bind as `storage_read`).
    #[must_use]
    pub fn state_in(&self) -> &wgpu::Buffer {
        if self.current_is_a {
            &self.state_a
        } else {
            &self.state_b
        }
    }

    /// The buffer to write new state TO (bind as `storage_rw`).
    #[must_use]
    pub fn state_out(&self) -> &wgpu::Buffer {
        if self.current_is_a {
            &self.state_b
        } else {
            &self.state_a
        }
    }

    /// Swap buffers after a dispatch step: the "out" becomes the new "in".
    pub fn swap(&mut self) {
        self.current_is_a = !self.current_is_a;
    }

    /// Read the current state back to CPU.
    /// # Errors
    /// Returns [`Err`] if buffer readback fails (e.g., device lost, `map_async` error).
    pub fn read_state(&self) -> Result<Vec<f64>> {
        self.device
            .read_f64_buffer(self.state_in(), self.n_cells * self.n_state_per_cell)
    }

    /// Write new state from CPU (e.g., for re-initialization).
    /// # Errors
    /// Returns [`Err`] if `state.len()` does not equal the expected buffer size.
    pub fn write_state(&self, state: &[f64]) -> Result<()> {
        let expected = self.n_cells * self.n_state_per_cell;
        if state.len() != expected {
            return Err(BarracudaError::InvalidInput {
                message: format!("state length {} != expected {expected}", state.len()),
            });
        }
        let bytes: &[u8] = bytemuck::cast_slice(state);
        self.device.queue.write_buffer(self.state_in(), 0, bytes);
        Ok(())
    }

    /// Total number of f64 values in the state buffer.
    #[must_use]
    pub fn state_len(&self) -> usize {
        self.n_cells * self.n_state_per_cell
    }

    /// Number of spatial cells.
    #[must_use]
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    /// Number of state values per cell.
    #[must_use]
    pub fn n_state_per_cell(&self) -> usize {
        self.n_state_per_cell
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_gpu_device;

    #[tokio::test]
    async fn test_batched_stateful_roundtrip() {
        let Some(device) = get_test_gpu_device().await else {
            return;
        };
        let initial = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let stateful = BatchedStatefulF64::new(device, 3, 2, &initial).unwrap();

        let readback = stateful.read_state().unwrap();
        assert_eq!(readback.len(), 6);
        for (i, (&got, &expected)) in readback.iter().zip(initial.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-15,
                "state[{i}] = {got}, expected {expected}"
            );
        }
    }

    #[tokio::test]
    async fn test_batched_stateful_swap() {
        let Some(device) = get_test_gpu_device().await else {
            return;
        };
        let initial = vec![10.0, 20.0];
        let mut stateful = BatchedStatefulF64::new(device, 2, 1, &initial).unwrap();

        assert!(std::ptr::eq(
            stateful.state_in(),
            &raw const stateful.state_a
        ));
        assert!(std::ptr::eq(
            stateful.state_out(),
            &raw const stateful.state_b
        ));

        stateful.swap();

        assert!(std::ptr::eq(
            stateful.state_in(),
            &raw const stateful.state_b
        ));
        assert!(std::ptr::eq(
            stateful.state_out(),
            &raw const stateful.state_a
        ));
    }

    #[tokio::test]
    async fn test_batched_stateful_write() {
        let Some(device) = get_test_gpu_device().await else {
            return;
        };
        let initial = vec![0.0; 4];
        let stateful = BatchedStatefulF64::new(device, 2, 2, &initial).unwrap();

        let new_state = vec![7.0, 8.0, 9.0, 10.0];
        stateful.write_state(&new_state).unwrap();
    }
}
