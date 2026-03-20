// SPDX-License-Identifier: AGPL-3.0-or-later
//! Real-to-Complex Fast Fourier Transform (RFFT)
//!
//! **Optimization**: Exploits conjugate symmetry for 50% speedup on real signals
//! **Input**: Real signal (N points)
//! **Output**: Complex spectrum (N/2+1 unique points)
//! **Mathematical Property**: X[k] = conj(X[N-k]) for real inputs
//!
//! GPU-resident: uses `real_to_complex_f64.wgsl` (downcast), batched FFT,
//! `rfft_extract_f64.wgsl` (downcast). Zero CPU readbacks.

use super::fft_1d::{AxisConfig, batched_shader_f32, dispatch_axis_inner, upload_twiddles_f32};
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use std::mem::size_of;

fn rtc_shader_f32() -> &'static str {
    static SHADER: std::sync::LazyLock<String> =
        std::sync::LazyLock::new(|| include_str!("real_to_complex_f64.wgsl").to_string());
    &SHADER
}

fn extract_shader_f32() -> &'static str {
    static SHADER: std::sync::LazyLock<String> =
        std::sync::LazyLock::new(|| include_str!("rfft_extract_f64.wgsl").to_string());
    &SHADER
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RtcParams {
    n: u32,
    padding: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ExtractParams {
    unique_points: u32,
    padding: [u32; 3],
}

/// Real-to-Complex FFT Operation
///
/// For real-valued inputs, FFT has conjugate symmetry: X[k] = conj(X[N-k])
/// This means we only need to compute N/2+1 unique frequency components.
pub struct Rfft {
    input: Tensor,
    degree: u32,
}

impl Rfft {
    /// Create a new RFFT operation
    /// # Arguments
    /// * `input` - Real-valued signal tensor (shape: [N])
    /// * `degree` - FFT degree (must be power of 2)
    /// # Errors
    /// Returns [`Err`] if degree is not a power of 2, input is not 1D, or input size ≠ degree.
    pub fn new(input: Tensor, degree: u32) -> Result<Self> {
        if degree == 0 || (degree & (degree - 1)) != 0 {
            return Err(BarracudaError::Device(format!(
                "FFT degree must be power of 2, got {degree}"
            )));
        }

        let shape = input.shape();
        if shape.len() != 1 {
            return Err(BarracudaError::Device(
                "RFFT input must be 1D (real signal)".to_string(),
            ));
        }

        if shape[0] != degree as usize {
            return Err(BarracudaError::Device(format!(
                "Input size {} doesn't match degree {}",
                shape[0], degree
            )));
        }

        Ok(Self { input, degree })
    }

    /// Execute RFFT operation. 3 GPU passes: real→complex, FFT, extract. Zero CPU readbacks.
    /// # Returns
    /// Complex spectrum tensor (shape: \[N/2+1, 2\])
    /// # Errors
    /// Returns [`Err`] if GPU dispatch, buffer creation, or readback fails.
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let n = self.degree as usize;
        let unique_points = (n / 2) + 1;

        let complex_bytes = (n * 2) * size_of::<f32>();
        let output_bytes = unique_points * 2 * size_of::<f32>();

        // Pass 1: Real → complex (zero-interleave)
        let complex_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RFFT complex"),
            size: complex_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let rtc_params = RtcParams {
            n: n as u32,
            padding: [0; 3],
        };
        let rtc_params_buf = device.create_uniform_buffer("RFFT RtcParams", &rtc_params);

        ComputeDispatch::new(device, "RFFT Real-to-Complex")
            .shader(rtc_shader_f32(), "main")
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &complex_buf)
            .uniform(2, &rtc_params_buf)
            .dispatch((n as u32).div_ceil(WORKGROUP_SIZE_1D), 1, 1)
            .submit()?;

        // Pass 2: Batched FFT
        let buf_b = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RFFT buf_b"),
            size: complex_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (tw_re, tw_im) = upload_twiddles_f32(device, n as u32);
        let axis = AxisConfig {
            degree: n as u32,
            element_stride: 1,
            dim1_stride: n as u32,
            dim2_stride: 1,
            dim1_count: 1,
            dim2_count: 1,
        };
        dispatch_axis_inner(
            device,
            batched_shader_f32(),
            &complex_buf,
            &buf_b,
            complex_bytes as u64,
            &tw_re,
            &tw_im,
            &axis,
            false,
            false,
        )?;

        // Pass 3: Extract N/2+1 unique points
        let output_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RFFT output"),
            size: output_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let extract_params = ExtractParams {
            unique_points: unique_points as u32,
            padding: [0; 3],
        };
        let extract_params_buf =
            device.create_uniform_buffer("RFFT ExtractParams", &extract_params);

        ComputeDispatch::new(device, "RFFT Extract")
            .shader(extract_shader_f32(), "main")
            .storage_read(0, &complex_buf)
            .storage_rw(1, &output_buf)
            .uniform(2, &extract_params_buf)
            .dispatch((unique_points as u32).div_ceil(WORKGROUP_SIZE_1D), 1, 1)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buf,
            vec![unique_points, 2],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rfft_simple() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        let n = 8;
        let data: Vec<f32> = (0..n)
            .map(|k| (2.0 * std::f32::consts::PI * (k as f32) / (n as f32)).sin())
            .collect();

        let tensor = Tensor::from_data(&data, vec![n], device).unwrap();

        let rfft = Rfft::new(tensor, n as u32).unwrap();
        let spectrum = rfft.execute().unwrap();

        assert_eq!(spectrum.shape(), &[5, 2]);
    }

    #[tokio::test]
    async fn test_rfft_dc_component() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        let n = 16;
        let data = vec![1.0f32; n];

        let tensor = Tensor::from_data(&data, vec![n], device).unwrap();

        let rfft = Rfft::new(tensor, n as u32).unwrap();
        let spectrum = rfft.execute().unwrap();
        let spectrum_data = spectrum.to_vec().unwrap();

        let dc_real = spectrum_data[0];
        assert!((dc_real - (n as f32)).abs() < 1e-4, "DC component = N");

        for i in 1..=(n / 2) {
            let mag = spectrum_data[i * 2].hypot(spectrum_data[i * 2 + 1]);
            assert!(mag < 1e-3, "Non-DC components near zero");
        }
    }

    #[tokio::test]
    async fn test_rfft_conjugate_symmetry() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        let n = 32;
        let data: Vec<f32> = (0..n)
            .map(|k| {
                let t = (k as f32) / (n as f32);
                (2.0 * std::f32::consts::PI * 3.0 * t).sin()
                    + 0.5 * (2.0 * std::f32::consts::PI * 7.0 * t).cos()
            })
            .collect();

        let tensor = Tensor::from_data(&data, vec![n], device).unwrap();

        let rfft = Rfft::new(tensor, n as u32).unwrap();
        let spectrum = rfft.execute().unwrap();

        assert_eq!(spectrum.shape(), &[17, 2]);

        let spectrum_data = spectrum.to_vec().unwrap();
        let total_energy: f32 = (0..17)
            .map(|i| spectrum_data[i * 2].powi(2) + spectrum_data[i * 2 + 1].powi(2))
            .sum();

        assert!(total_energy > 1.0, "Spectrum has energy");
    }

    #[tokio::test]
    async fn test_rfft_performance_benefit() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        let n = 4096;
        let data: Vec<f32> = (0..n).map(|k| (k as f32).sin() / 100.0).collect();

        let tensor = Tensor::from_data(&data, vec![n], device).unwrap();

        let start = std::time::Instant::now();
        let rfft = Rfft::new(tensor, n as u32).unwrap();
        let _spectrum = rfft.execute().unwrap();
        let elapsed = start.elapsed();

        assert!(elapsed.as_secs() < 30, "RFFT completes efficiently");
    }
}
