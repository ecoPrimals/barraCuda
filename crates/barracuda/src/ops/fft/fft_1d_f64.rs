// SPDX-License-Identifier: AGPL-3.0-or-later
//! 1D Fast Fourier Transform Operation (f64 Precision)
//!
//! **Evolution**: f64 version for PPPM/Ewald long-range electrostatics
//! **Critical**: Unblocks full plasma simulation (Coulomb κ=0 cases)
//!
//! Uses the shared batched engine from `fft_1d` with f64 buffers.

use super::fft_1d::{AxisConfig, dispatch_axis_f64, upload_twiddles_f64};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use std::mem::size_of;

/// 1D Complex FFT operation (f64 precision)
///
/// Transforms complex signal from time/spatial domain to frequency domain
/// with double precision for PPPM/Ewald accuracy.
pub struct Fft1DF64 {
    input: Tensor,
    degree: u32,
}

impl Fft1DF64 {
    /// Create a new 1D FFT operation (f64)
    /// ## Parameters
    /// - `input`: Complex tensor (shape [..., N, 2] where last dim is (real, imag))
    /// - `degree`: FFT size N (must be power of 2)
    /// # Errors
    /// Returns [`Err`] if the input's last dimension is not 2 (complex representation), or if
    /// `degree` is not a power of 2.
    pub fn new(input: Tensor, degree: u32) -> Result<Self> {
        let shape = input.shape();
        if shape.last() != Some(&2) {
            return Err(BarracudaError::Device(
                "FFT input must have last dimension = 2 (complex)".to_string(),
            ));
        }

        if degree & (degree - 1) != 0 {
            return Err(BarracudaError::Device(format!(
                "FFT degree {degree} must be power of 2"
            )));
        }

        Ok(Self { input, degree })
    }

    /// Execute the FFT (forward transform).
    /// # Errors
    /// Returns [`Err`] if GPU dispatch, buffer creation, or readback fails.
    pub async fn execute(&self) -> Result<Tensor> {
        self.execute_internal(false).await
    }

    /// Execute the inverse FFT.
    /// Note: Result must be scaled by `1/N` for proper normalization.
    /// # Errors
    /// Returns [`Err`] if GPU dispatch, buffer creation, or readback fails.
    pub async fn execute_inverse(&self) -> Result<Tensor> {
        self.execute_internal(true).await
    }

    async fn execute_internal(&self, inverse: bool) -> Result<Tensor> {
        let device = self.input.device();
        let n = self.degree;
        let buffer_bytes = (n as u64) * 2 * size_of::<f64>() as u64;

        let buf_a = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFT1D f64 buf_a"),
            size: buffer_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        {
            let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("FFT1D f64 Copy Input"),
            });
            encoder.copy_buffer_to_buffer(self.input.buffer(), 0, &buf_a, 0, buffer_bytes);
            device.submit_commands(std::iter::once(encoder.finish()));
        }

        let buf_b = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFT1D f64 buf_b"),
            size: buffer_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (tw_re, tw_im) = upload_twiddles_f64(device, n);

        let axis = AxisConfig {
            degree: n,
            element_stride: 1,
            dim1_stride: n,
            dim2_stride: 1,
            dim1_count: 1,
            dim2_count: 1,
        };

        dispatch_axis_f64(
            device,
            &buf_a,
            &buf_b,
            buffer_bytes,
            &tw_re,
            &tw_im,
            &axis,
            inverse,
        )?;

        Ok(Tensor::from_buffer(
            buf_a,
            vec![n as usize, 2],
            device.clone(),
        ))
    }

    /// Get the FFT degree (size N)
    #[must_use]
    pub fn degree(&self) -> u32 {
        self.degree
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_f64_gpu_available;

    #[test]
    fn test_twiddle_factors() {
        let n = 8u32;
        let pi = std::f64::consts::PI;

        let mut twiddle_re = Vec::new();
        let mut twiddle_im = Vec::new();

        for k in 0..n {
            let angle = -2.0 * pi * (k as f64) / (n as f64);
            twiddle_re.push(angle.cos());
            twiddle_im.push(angle.sin());
        }

        assert!((twiddle_re[0] - 1.0).abs() < 1e-10);
        assert!((twiddle_im[0]).abs() < 1e-10);

        assert!((twiddle_re[2]).abs() < 1e-10);
        assert!((twiddle_im[2] - (-1.0)).abs() < 1e-10);

        assert!((twiddle_re[4] - (-1.0)).abs() < 1e-10);
        assert!((twiddle_im[4]).abs() < 1e-10);
    }

    #[test]
    fn test_power_of_2_validation() {
        fn is_power_of_two(n: usize) -> bool {
            n.is_power_of_two()
        }
        assert!(is_power_of_two(16));
        assert!(is_power_of_two(256));
        assert!(is_power_of_two(1024));
        assert!(!is_power_of_two(15));
        assert!(!is_power_of_two(100));
    }

    #[tokio::test]
    async fn test_fft_1d_f64_impulse_spectrum() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let n = 8u32;
        let mut data = vec![0.0f64; (n * 2) as usize];
        data[0] = 1.0;

        let tensor = Tensor::from_f64_data(&data, vec![n as usize, 2], device).unwrap();
        let fft = Fft1DF64::new(tensor, n).unwrap();
        let spectrum = fft.execute().await.unwrap();
        let freq = spectrum.to_f64_vec().unwrap();

        for k in 0..n as usize {
            let re = freq[k * 2];
            let im = freq[k * 2 + 1];
            let mag = re.hypot(im);
            assert!(
                (mag - 1.0).abs() < 1e-10,
                "Bin {k}: expected magnitude 1.0, got {mag:.15e}"
            );
        }
    }

    #[tokio::test]
    async fn test_fft_1d_f64_roundtrip() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let n = 16u32;
        let pi = std::f64::consts::PI;
        let mut data = vec![0.0f64; (n * 2) as usize];
        for i in 0..n as usize {
            data[i * 2] = 0.5f64.mul_add(
                (4.0 * pi * i as f64 / n as f64).cos(),
                (2.0 * pi * i as f64 / n as f64).sin(),
            );
        }
        let original = data.clone();

        let fwd_tensor = Tensor::from_f64_data(&data, vec![n as usize, 2], device.clone()).unwrap();
        let fft = Fft1DF64::new(fwd_tensor, n).unwrap();
        let spectrum = fft.execute().await.unwrap();

        let spec_data = spectrum.to_f64_vec().unwrap();
        let inv_tensor = Tensor::from_f64_data(&spec_data, vec![n as usize, 2], device).unwrap();
        let ifft = Fft1DF64::new(inv_tensor, n).unwrap();
        let recovered_raw = ifft.execute_inverse().await.unwrap();
        let recovered = recovered_raw.to_f64_vec().unwrap();

        for i in 0..n as usize {
            let re = recovered[i * 2] / n as f64;
            let im = recovered[i * 2 + 1] / n as f64;
            assert!(
                (re - original[i * 2]).abs() < 1e-10,
                "Sample {i} real: expected {:.15e}, got {re:.15e}",
                original[i * 2]
            );
            assert!(
                im.abs() < 1e-10,
                "Sample {i} imag: expected ~0, got {im:.15e}"
            );
        }
    }

    #[tokio::test]
    async fn test_fft_1d_f64_single_frequency() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let n = 8u32;
        let target_bin = 2usize;
        let pi = std::f64::consts::PI;
        let mut data = vec![0.0f64; (n * 2) as usize];
        for i in 0..n as usize {
            let angle = 2.0 * pi * target_bin as f64 * i as f64 / n as f64;
            data[i * 2] = angle.cos();
            data[i * 2 + 1] = angle.sin();
        }

        let tensor = Tensor::from_f64_data(&data, vec![n as usize, 2], device).unwrap();
        let fft = Fft1DF64::new(tensor, n).unwrap();
        let spectrum = fft.execute().await.unwrap();
        let freq = spectrum.to_f64_vec().unwrap();

        let re_t = freq[target_bin * 2];
        let im_t = freq[target_bin * 2 + 1];
        let mag_t = re_t.hypot(im_t);
        assert!(
            (mag_t - n as f64).abs() < 1e-8,
            "Bin {target_bin}: expected magnitude {n}, got {mag_t:.15e}"
        );

        for k in 0..n as usize {
            if k == target_bin {
                continue;
            }
            let re = freq[k * 2];
            let im = freq[k * 2 + 1];
            let mag = re.hypot(im);
            assert!(
                mag < 1e-8,
                "Bin {k}: expected near-zero magnitude, got {mag:.15e}"
            );
        }
    }
}
