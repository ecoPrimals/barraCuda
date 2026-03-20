// SPDX-License-Identifier: AGPL-3.0-or-later
//! Quantize - Convert FP32 to INT8/INT4 quantization
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute
//!
//! Quantizes floating point values to low-precision integers.
//! Used for model compression and efficient inference.

/// WGSL kernel for quantization parameter computation (f64).
pub const WGSL_QUANTIZE_PARAMS_F64: &str = include_str!("../shaders/misc/quantize_params_f64.wgsl");

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct QuantizeParams {
    size: u32,
    scale: f32,
    zero_point: f32,
    num_bits: u32,
    _padding: u32,
}

/// Quantize operation
pub struct Quantize {
    input: Tensor,
    scale: f32,
    zero_point: f32,
    num_bits: u32,
}

impl Quantize {
    /// Create quantize operation
    /// # Arguments
    /// * `input` - Input tensor (FP32)
    /// * `scale` - Quantization scale (inverse of quantization scale)
    /// * `zero_point` - Quantization zero point
    /// * `num_bits` - Number of bits (4 for INT4, 8 for INT8)
    /// # Errors
    /// Returns [`Err`] if scale <= 0 or `num_bits` is not 4 or 8.
    pub fn new(input: Tensor, scale: f32, zero_point: f32, num_bits: u32) -> Result<Self> {
        if scale <= 0.0 {
            return Err(BarracudaError::invalid_op(
                "quantize",
                format!("scale must be positive, got {scale}"),
            ));
        }
        if num_bits != 4 && num_bits != 8 {
            return Err(BarracudaError::invalid_op(
                "quantize",
                format!("num_bits must be 4 or 8, got {num_bits}"),
            ));
        }

        Ok(Self {
            input,
            scale,
            zero_point,
            num_bits,
        })
    }

    /// WGSL shader source (embedded at compile time)
    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!("../shaders/misc/quantize_f64.wgsl").to_string()
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Execute quantize on tensor
    /// Returns a tensor with i32 values (quantized integers)
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer readback fails (e.g. device lost).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size = self.input.len();

        // Create output buffer (i32 for quantized values)
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Quantize Output"),
            size: (size * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        // Create params
        let params = QuantizeParams {
            size: size as u32,
            scale: self.scale,
            zero_point: self.zero_point,
            num_bits: self.num_bits,
            _padding: 0,
        };

        let params_buffer = device.create_uniform_buffer("Quantize Params", &params);

        ComputeDispatch::new(device, "Quantize")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        // Create staging buffer for reading i32 data
        let staging_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Quantize Staging Buffer"),
            size: (size * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy output to staging buffer (compute completed via ComputeDispatch above)
        let mut copy_encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Quantize Copy Encoder"),
        });
        copy_encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (size * std::mem::size_of::<i32>()) as u64,
        );
        device.submit_commands(Some(copy_encoder.finish()));

        let i32_data: Vec<i32> = device.map_staging_buffer(&staging_buffer, size)?;
        let f32_data: Vec<f32> = i32_data.iter().map(|&x| x as f32).collect();

        // Create tensor from f32 data (values represent quantized integers)
        Tensor::from_data(&f32_data, self.input.shape().to_vec(), device.clone())
    }
}

/// Compute affine i8 quantization parameters from data range.
///
/// Returns `(scale, zero_point)` for symmetric quantization around zero,
/// mapping `[-abs_max, abs_max]` to `[-127, 127]`.
#[must_use]
pub fn compute_affine_i8_params(data: &[f32]) -> (f32, f32) {
    let abs_max = data
        .iter()
        .fold(0.0f32, |acc, &x| acc.max(x.abs()))
        .max(1e-10);
    let scale = 127.0 / abs_max;
    (scale, 0.0)
}

/// Quantize a tensor to affine int8 in one call.
///
/// Computes scale from the tensor's range and applies symmetric quantization.
/// For asymmetric quantization or custom `scale/zero_point`, use `Quantize::new`.
///
/// # Errors
///
/// Returns [`Err`] if tensor readback fails for scale computation, or if [`Quantize::new`] or [`Quantize::execute`] fails.
pub fn quantize_affine_i8(input: Tensor) -> Result<(Tensor, f32, f32)> {
    let data = input.to_vec().map_err(|e| BarracudaError::InvalidInput {
        message: format!("failed to read tensor for scale computation: {e}"),
    })?;
    let (scale, zero_point) = compute_affine_i8_params(&data);
    let quantized = Quantize::new(input, scale, zero_point, 8)?.execute()?;
    Ok((quantized, scale, zero_point))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_quantize_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_vec_on(vec![-1.0, 0.0, 1.0], vec![3], device.clone())
            .await
            .unwrap();

        let output = Quantize::new(input, 0.01, 0.0, 8)
            .unwrap()
            .execute()
            .unwrap();
        let result = output.to_vec().unwrap();

        assert_eq!(result.len(), 3);
        // Values should be quantized (as f32 representation of i32)
    }

    #[tokio::test]
    async fn test_quantize_edge_cases() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Test clamping at boundaries
        let input = Tensor::from_vec_on(vec![-1000.0, 1000.0, 0.0], vec![3], device.clone())
            .await
            .unwrap();

        let output = Quantize::new(input, 1.0, 0.0, 8)
            .unwrap()
            .execute()
            .unwrap();
        let result = output.to_vec().unwrap();

        // Should clamp to INT8 range [-128, 127]
        assert_eq!(result[0] as i32, -128);
        assert_eq!(result[1] as i32, 127);
        assert_eq!(result[2] as i32, 0);
    }

    #[tokio::test]
    async fn test_quantize_int4() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_vec_on(vec![-10.0, 0.0, 10.0], vec![3], device.clone())
            .await
            .unwrap();

        let output = Quantize::new(input, 1.0, 0.0, 4)
            .unwrap()
            .execute()
            .unwrap();
        let result = output.to_vec().unwrap();

        // Should clamp to INT4 range [-8, 7]
        assert_eq!(result[0] as i32, -8);
        assert_eq!(result[1] as i32, 0);
        assert_eq!(result[2] as i32, 7);
    }
}
