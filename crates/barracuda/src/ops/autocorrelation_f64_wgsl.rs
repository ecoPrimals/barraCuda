// SPDX-License-Identifier: AGPL-3.0-or-later
//! General 1D autocorrelation (f64) — single GPU dispatch for all lags
//!
//! Computes C(lag) = (1/(N-lag)) × Σ x\[t\] × x\[t+lag\] for lags `0..max_lag`.
//! One workgroup per lag, tree reduction within workgroup. Single dispatch,
//! zero CPU round-trips.
//!
//! Generalised from MD-specific VACF (velocity autocorrelation function) to
//! serve all springs: spectral analysis, time-series correlation, signal
//! processing.

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("../shaders/stats/autocorrelation_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    n: u32,
    max_lag: u32,
    _pad0: u32,
    _pad1: u32,
}

/// General 1D autocorrelation on GPU.
pub struct AutocorrelationF64 {
    device: Arc<WgpuDevice>,
}

impl AutocorrelationF64 {
    /// Create a new autocorrelation evaluator.
    /// # Errors
    /// Returns [`Err`] if device initialization fails.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Compute autocorrelation C(lag) for lags 0..`max_lag`.
    ///
    /// Returns a vector of length `max_lag` where `out[k]` = C(k).
    /// Single GPU dispatch for all lags simultaneously.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if the input is shorter than `max_lag`, if buffer
    /// allocation fails, the device is lost, or buffer readback fails.
    pub fn autocorrelation(&self, data: &[f64], max_lag: usize) -> Result<Vec<f64>> {
        let n = data.len();
        if n == 0 || max_lag == 0 {
            return Ok(Vec::new());
        }
        if max_lag > n {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: format!("max_lag ({max_lag}) exceeds input length ({n})"),
            });
        }

        let input_buffer =
            self.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Autocorr Input"),
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let output_buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Autocorr Output"),
            size: (max_lag * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = Params {
            n: n as u32,
            max_lag: max_lag as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer =
            self.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Autocorr Params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        ComputeDispatch::new(&self.device, "autocorrelation_f64")
            .shader(SHADER, "main")
            .f64()
            .storage_read(0, &input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch(max_lag as u32, 1, 1)
            .submit()?;

        self.device.read_buffer::<f64>(&output_buffer, max_lag)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn autocorrelation_cpu(data: &[f64], max_lag: usize) -> Vec<f64> {
        let n = data.len();
        (0..max_lag)
            .map(|lag| {
                let valid = n - lag;
                if valid == 0 {
                    return 0.0;
                }
                let sum: f64 = (0..valid).map(|t| data[t] * data[t + lag]).sum();
                sum / valid as f64
            })
            .collect()
    }

    #[tokio::test]
    async fn test_autocorrelation_constant_signal() {
        let Some(dev) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };
        let op = AutocorrelationF64::new(dev).unwrap();
        let data = vec![1.0_f64; 256];
        let result = op.autocorrelation(&data, 10).unwrap();
        for c in &result {
            assert!(
                (*c - 1.0).abs() < 1e-10,
                "Constant signal autocorrelation should be 1.0, got {c}"
            );
        }
    }

    #[tokio::test]
    async fn test_autocorrelation_vs_cpu() {
        let Some(dev) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };
        let op = AutocorrelationF64::new(dev).unwrap();
        let data: Vec<f64> = (0..512).map(|i| (i as f64 * 0.1).sin()).collect();
        let max_lag = 32;
        let gpu_result = op.autocorrelation(&data, max_lag).unwrap();
        let cpu_result = autocorrelation_cpu(&data, max_lag);
        for (lag, (g, c)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
            assert!(
                (g - c).abs() < 1e-6,
                "Mismatch at lag {lag}: gpu={g}, cpu={c}"
            );
        }
    }
}
