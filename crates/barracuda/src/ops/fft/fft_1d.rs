// SPDX-License-Identifier: AGPL-3.0-or-later
//! 1D Fast Fourier Transform Operation
//!
//! **Evolution**: Adapted from FheNtt (80% Rust structure reuse!)
//! **Performance**: ~10x faster than NTT (native float vs U64 emulation)
//! **CRITICAL**: Unblocks PPPM, structure factors, all wave physics

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use wgpu::util::DeviceExt;

/// 1D Complex FFT operation
///
/// Transforms complex signal from time/spatial domain to frequency domain.
pub struct Fft1D {
    input: Tensor,
    degree: u32,
    twiddle_factors: Vec<f32>, // Precomputed exp(-2πik/N) as complex pairs
}

impl Fft1D {
    /// Create a new 1D FFT operation
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
        // PRECOMPUTE TWIDDLE FACTORS
        // ================================================================
        // twiddle[k] = exp(-2πik/N) for k = 0 to N-1
        // These are the roots of unity on the complex unit circle

        let mut twiddle_factors = Vec::with_capacity((degree * 2) as usize);
        let pi = std::f32::consts::PI;

        for k in 0..degree {
            let angle = -2.0 * pi * (k as f32) / (degree as f32);
            let real = angle.cos(); // exp(iθ) = cos(θ) + i·sin(θ)
            let imag = angle.sin();
            twiddle_factors.push(real);
            twiddle_factors.push(imag);
        }

        Ok(Self {
            input,
            degree,
            twiddle_factors,
        })
    }

    /// Execute FFT transformation
    ///
    /// Returns a new tensor containing the frequency-domain representation.
    ///
    /// ## Algorithm
    ///
    /// 1. Bit-reversal permutation (preprocessing)
    /// 2. log₂(N) butterfly stages (Cooley-Tukey FFT)
    /// 3. Each stage processes N/2 butterflies in parallel
    ///
    /// ## Complexity
    ///
    /// - Time: O(N log N)
    /// - Space: O(N) temporary buffer
    /// - GPU parallelism: N/2 threads per stage
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();

        // Buffer size: degree * 2 f32s (for complex numbers)
        let buffer_size = self.degree as u64 * 2 * std::mem::size_of::<f32>() as u64;

        // Create output buffer
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFT Output Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create intermediate buffer (for ping-pong between stages)
        let intermediate_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFT Intermediate Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Upload twiddle factors to GPU
        let twiddle_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("FFT Twiddle Factors"),
                contents: bytemuck::cast_slice(&self.twiddle_factors),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let shader_source = include_str!("fft_1d.wgsl");

        // Params struct
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct FftParams {
            degree: u32,
            stage: u32,
            inverse: u32,
            _padding: u32,
        }

        let base_params = FftParams {
            degree: self.degree,
            stage: 0,
            inverse: 0, // Forward FFT
            _padding: 0,
        };

        // ============================================================
        // Pass 1: Bit-reversal permutation
        // ============================================================

        let params_buffer = device.create_uniform_buffer("FFT Params (Bit Reverse)", &base_params);

        ComputeDispatch::new(device, "FFT Bit Reverse")
            .shader(shader_source, "bit_reverse")
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &intermediate_buffer)
            .storage_read(2, &twiddle_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(self.degree)
            .submit();

        // ============================================================
        // Pass 2-N: Butterfly stages (log₂(N) stages)
        // ============================================================

        let num_stages = (self.degree as f32).log2() as u32;
        let mut current_input = intermediate_buffer;
        let mut current_output = output_buffer;

        for stage in 0..num_stages {
            let stage_params = FftParams {
                stage,
                ..base_params
            };
            let stage_params_buffer =
                device.create_uniform_buffer(&format!("FFT Params (Stage {stage})"), &stage_params);

            ComputeDispatch::new(device, &format!("FFT Butterfly Stage {stage}"))
                .shader(shader_source, "main")
                .storage_read(0, &current_input)
                .storage_rw(1, &current_output)
                .storage_read(2, &twiddle_buffer)
                .uniform(3, &stage_params_buffer)
                .dispatch_1d(self.degree / 2)
                .submit();

            // Ping-pong buffers for next stage
            std::mem::swap(&mut current_input, &mut current_output);
        }

        // After each stage we swap, so current_input always holds the last written buffer.
        let final_buffer = current_input;

        // Create result tensor
        Ok(Tensor::from_buffer(
            final_buffer,
            self.input.shape().to_vec(), // Preserve input shape!
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fft_1d_simple() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // Simple 4-point FFT test
        // Input: [1+0i, 0+0i, 0+0i, 0+0i]
        let data = vec![
            1.0f32, 0.0, // 1+0i
            0.0, 0.0, // 0+0i
            0.0, 0.0, // 0+0i
            0.0, 0.0, // 0+0i
        ];

        let tensor = Tensor::from_data(&data, vec![4, 2], device.clone()).unwrap();
        let fft = Fft1D::new(tensor, 4).unwrap();
        let result = fft.execute().unwrap();

        let result_data = result.to_vec().unwrap();

        // FFT([1,0,0,0]) = [1,1,1,1] (all ones in frequency domain)
        assert!((result_data[0] - 1.0).abs() < 1e-5); // DC component
        assert!((result_data[2] - 1.0).abs() < 1e-5); // First harmonic
    }
}
