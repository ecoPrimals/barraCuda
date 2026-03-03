// SPDX-License-Identifier: AGPL-3.0-or-later
//! 1D Fast Fourier Transform Operation (f64 Precision)
//!
//! **Evolution**: f64 version for PPPM/Ewald long-range electrostatics
//! **Critical**: Unblocks full plasma simulation (Coulomb κ=0 cases)
//!
//! ## Differences from f32 Version
//!
//! - Uses `f64` storage instead of `f32`
//! - Twiddle factors stored in separate re/im arrays (no `vec2<f64>`)
//! - Uses `math_f64.wgsl` for sin/cos (prepended via `ShaderTemplate`)
//! - Double precision throughout for energy conservation

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use wgpu::util::DeviceExt;

/// 1D Complex FFT operation (f64 precision)
///
/// Transforms complex signal from time/spatial domain to frequency domain
/// with double precision for PPPM/Ewald accuracy.
pub struct Fft1DF64 {
    input: Tensor,
    degree: u32,
    twiddle_re: Vec<f64>, // Precomputed real parts of exp(-2πik/N)
    twiddle_im: Vec<f64>, // Precomputed imag parts of exp(-2πik/N)
}

impl Fft1DF64 {
    /// Create a new 1D FFT operation (f64)
    ///
    /// ## Parameters
    ///
    /// - `input`: Complex tensor (shape [..., N, 2] where last dim is (real, imag))
    /// - `degree`: FFT size N (must be power of 2)
    ///
    /// ## Constraints
    ///
    /// - N must be a power of 2 (for Cooley-Tukey radix-2 FFT)
    /// - Input must have last dimension = 2 (complex representation)
    /// - Device must support f64 shaders (SHADER_F64 feature)
    pub fn new(input: Tensor, degree: u32) -> Result<Self> {
        // Validate input
        let shape = input.shape();
        if shape.last() != Some(&2) {
            return Err(BarracudaError::Device(
                "FFT input must have last dimension = 2 (complex)".to_string(),
            ));
        }

        // Validate degree is power of 2
        if degree & (degree - 1) != 0 {
            return Err(BarracudaError::Device(format!(
                "FFT degree {degree} must be power of 2"
            )));
        }

        let _device = input.device();

        // ================================================================
        // PRECOMPUTE TWIDDLE FACTORS (f64)
        // ================================================================
        // twiddle[k] = exp(-2πik/N) for k = 0 to N-1
        // Stored as separate arrays for re/im (no vec2<f64> in WGSL)

        let mut twiddle_re = Vec::with_capacity(degree as usize);
        let mut twiddle_im = Vec::with_capacity(degree as usize);
        let pi = std::f64::consts::PI;

        for k in 0..degree {
            let angle = -2.0 * pi * (k as f64) / (degree as f64);
            twiddle_re.push(angle.cos());
            twiddle_im.push(angle.sin());
        }

        Ok(Self {
            input,
            degree,
            twiddle_re,
            twiddle_im,
        })
    }

    /// Execute the FFT (forward transform)
    ///
    /// Returns the frequency-domain representation of the input signal.
    pub async fn execute(&self) -> Result<Tensor> {
        self.execute_internal(false).await
    }

    /// Execute the inverse FFT
    ///
    /// Returns the time-domain signal from frequency representation.
    /// Note: Result must be scaled by 1/N for proper normalization.
    pub async fn execute_inverse(&self) -> Result<Tensor> {
        self.execute_internal(true).await
    }

    async fn execute_internal(&self, inverse: bool) -> Result<Tensor> {
        let device = self.input.device();
        let n = self.degree;
        let log_n = (n as f32).log2() as u32;

        // ================================================================
        // CREATE BUFFERS
        // ================================================================

        let buffer_size = (n * 2) as u64 * std::mem::size_of::<f64>() as u64;
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFT 1D f64 Output"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let twiddle_re_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("FFT 1D f64 Twiddle Re"),
                    contents: bytemuck::cast_slice(&self.twiddle_re),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let twiddle_im_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("FFT 1D f64 Twiddle Im"),
                    contents: bytemuck::cast_slice(&self.twiddle_im),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let working_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFT 1D f64 Working"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader_source = include_str!("fft_1d_f64.wgsl");

        // Copy input to working buffer
        {
            let mut encoder =
                device
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("FFT 1D f64 Copy Encoder"),
                    });
            encoder.copy_buffer_to_buffer(self.input.buffer(), 0, &working_buffer, 0, buffer_size);
            device.submit_and_poll(std::iter::once(encoder.finish()));
        }

        // Step 1: Bit-reverse permutation
        let params = Fft64Params {
            degree: n,
            stage: 0,
            inverse: if inverse { 1 } else { 0 },
            _padding: 0,
        };

        let params_buffer = device.create_uniform_buffer("FFT 1D f64 Params", &params);

        ComputeDispatch::new(device, "FFT 1D f64 Bit Reverse")
            .shader(shader_source, "bit_reverse")
            .f64()
            .storage_read(0, &working_buffer)
            .storage_rw(1, &output_buffer)
            .storage_read(2, &twiddle_re_buffer)
            .storage_read(3, &twiddle_im_buffer)
            .uniform(4, &params_buffer)
            .dispatch_1d(n)
            .submit();

        // Copy result back to working buffer
        {
            let mut encoder =
                device
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("FFT 1D f64 Copy Encoder 2"),
                    });
            encoder.copy_buffer_to_buffer(&output_buffer, 0, &working_buffer, 0, buffer_size);
            device.submit_and_poll(std::iter::once(encoder.finish()));
        }

        // Step 2: Butterfly stages
        for stage in 0..log_n {
            let stage_params = Fft64Params {
                degree: n,
                stage,
                inverse: if inverse { 1 } else { 0 },
                _padding: 0,
            };

            let stage_params_buffer = device
                .create_uniform_buffer(&format!("FFT 1D f64 Params Stage {stage}"), &stage_params);

            ComputeDispatch::new(device, &format!("FFT 1D f64 Butterfly Stage {stage}"))
                .shader(shader_source, "main")
                .f64()
                .storage_read(0, &working_buffer)
                .storage_rw(1, &output_buffer)
                .storage_read(2, &twiddle_re_buffer)
                .storage_read(3, &twiddle_im_buffer)
                .uniform(4, &stage_params_buffer)
                .dispatch_1d(n / 2)
                .submit();

            // Ping-pong: copy output to working for next stage
            if stage < log_n - 1 {
                let mut encoder =
                    device
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some(&format!("FFT 1D f64 Copy Stage {stage}")),
                        });
                encoder.copy_buffer_to_buffer(&output_buffer, 0, &working_buffer, 0, buffer_size);
                device.submit_and_poll(std::iter::once(encoder.finish()));
            }
        }

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![n as usize, 2],
            device.clone(),
        ))
    }

    /// Get the FFT degree (size N)
    pub fn degree(&self) -> u32 {
        self.degree
    }
}

/// FFT parameters (matches WGSL struct)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Fft64Params {
    degree: u32,
    stage: u32,
    inverse: u32,
    _padding: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_f64_gpu_available;

    // -------------------------------------------------------------------------
    // CPU-only tests (run everywhere)
    // -------------------------------------------------------------------------

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

        // W_8^0 = 1 + 0i
        assert!((twiddle_re[0] - 1.0).abs() < 1e-10);
        assert!((twiddle_im[0]).abs() < 1e-10);

        // W_8^2 = 0 - 1i
        assert!((twiddle_re[2]).abs() < 1e-10);
        assert!((twiddle_im[2] - (-1.0)).abs() < 1e-10);

        // W_8^4 = -1 + 0i
        assert!((twiddle_re[4] - (-1.0)).abs() < 1e-10);
        assert!((twiddle_im[4]).abs() < 1e-10);
    }

    #[test]
    fn test_power_of_2_validation() {
        fn is_power_of_two(n: usize) -> bool {
            n != 0 && (n & (n - 1)) == 0
        }
        assert!(is_power_of_two(16));
        assert!(is_power_of_two(256));
        assert!(is_power_of_two(1024));
        assert!(!is_power_of_two(15));
        assert!(!is_power_of_two(100));
    }

    // -------------------------------------------------------------------------
    // GPU tests — skipped gracefully when no f64-capable GPU is present.
    //
    // These validate the full path:
    //   Rust twiddle precompute → Tensor upload → WGSL butterfly via wgpu/Vulkan
    //   → readback → numerical check at 1e-10 tolerance.
    //
    // Three orthogonal properties tested:
    //   1. Impulse spectrum  — FFT of a unit impulse is a flat-magnitude spectrum
    //   2. Roundtrip         — FFT followed by IFFT recovers non-trivial signal
    //   3. Single frequency  — pure complex tone maps to a single bin
    //
    // The impulse-only test (test 1) would pass even with a broken inverse because
    // FFT(FFT(impulse))/N = impulse.  Tests 2 and 3 catch that regression.
    // -------------------------------------------------------------------------

    /// FFT of a unit impulse at t=0 must produce a flat-magnitude spectrum.
    /// Every bin X[k] = 1 (magnitude), with tolerance 1e-10.
    /// Validates: forward butterfly correctness, twiddle precompute, SHADER_F64 path.
    #[tokio::test]
    async fn test_fft_1d_f64_impulse_spectrum_gpu() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let n = 8u32;
        let mut data = vec![0.0f64; (n * 2) as usize];
        data[0] = 1.0; // x[0] = 1+0i; all others zero

        let tensor = Tensor::from_f64_data(&data, vec![n as usize, 2], device).unwrap();
        let fft = Fft1DF64::new(tensor, n).unwrap();
        let spectrum = fft.execute().await.unwrap();
        let freq = spectrum.to_f64_vec().unwrap();

        for k in 0..n as usize {
            let re = freq[k * 2];
            let im = freq[k * 2 + 1];
            let mag = (re * re + im * im).sqrt();
            assert!(
                (mag - 1.0).abs() < 1e-10,
                "Bin {k}: expected magnitude 1.0, got {mag:.15e} (re={re}, im={im})"
            );
        }
    }

    /// Forward FFT followed by inverse FFT must recover the original signal.
    /// Uses a non-trivial sinusoidal signal so the test is not degenerate
    /// (impulse roundtrip passes even with a broken inverse).
    /// Tolerance: 1e-10 on real part; imaginary residual < 1e-10.
    #[tokio::test]
    async fn test_fft_1d_f64_roundtrip_gpu() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let n = 16u32;
        let pi = std::f64::consts::PI;
        let mut data = vec![0.0f64; (n * 2) as usize];
        for i in 0..n as usize {
            // Real-valued signal with multiple harmonics; imaginary part = 0
            data[i * 2] = (2.0 * pi * i as f64 / n as f64).sin()
                + 0.5 * (4.0 * pi * i as f64 / n as f64).cos();
        }
        let original = data.clone();

        // Forward FFT
        let fwd_tensor = Tensor::from_f64_data(&data, vec![n as usize, 2], device.clone()).unwrap();
        let fft = Fft1DF64::new(fwd_tensor, n).unwrap();
        let spectrum = fft.execute().await.unwrap();

        // Inverse FFT
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

    /// A pure complex tone at frequency k=2 (x[n] = exp(2πi·2n/N)) must
    /// produce exactly one non-zero bin: |X[2]| = N, all others < 1e-8.
    /// Validates spectral concentration and confirms the Vulkan f64 path is live.
    #[tokio::test]
    async fn test_fft_1d_f64_single_frequency_gpu() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let n = 8u32;
        let target_bin = 2usize; // tone at k=2
        let pi = std::f64::consts::PI;
        let mut data = vec![0.0f64; (n * 2) as usize];
        for i in 0..n as usize {
            let angle = 2.0 * pi * target_bin as f64 * i as f64 / n as f64;
            data[i * 2] = angle.cos(); // real:  cos(2π·2·n/N)
            data[i * 2 + 1] = angle.sin(); // imag:  sin(2π·2·n/N)
        }

        let tensor = Tensor::from_f64_data(&data, vec![n as usize, 2], device).unwrap();
        let fft = Fft1DF64::new(tensor, n).unwrap();
        let spectrum = fft.execute().await.unwrap();
        let freq = spectrum.to_f64_vec().unwrap();

        // Bin `target_bin` should have magnitude = N
        let re_t = freq[target_bin * 2];
        let im_t = freq[target_bin * 2 + 1];
        let mag_t = (re_t * re_t + im_t * im_t).sqrt();
        assert!(
            (mag_t - n as f64).abs() < 1e-8,
            "Bin {target_bin}: expected magnitude {n}, got {mag_t:.15e}"
        );

        // All other bins must be near zero
        for k in 0..n as usize {
            if k == target_bin {
                continue;
            }
            let re = freq[k * 2];
            let im = freq[k * 2 + 1];
            let mag = (re * re + im * im).sqrt();
            assert!(
                mag < 1e-8,
                "Bin {k}: expected near-zero magnitude, got {mag:.15e}"
            );
        }
    }
}
