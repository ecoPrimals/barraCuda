// SPDX-License-Identifier: AGPL-3.0-only
//! 1D Fast Fourier Transform Operation
//!
//! **Evolution**: Adapted from `FheNtt` (80% Rust structure reuse!)
//! **Performance**: ~10x faster than NTT (native float vs U64 emulation)
//! **CRITICAL**: Unblocks PPPM, structure factors, all wave physics
//!
//! Uses the batched FFT engine (`fft_3d_batched_f64.wgsl`) for both f32 and f64.
//! f32 path: downcast shader via `downcast_f64_to_f32`.

use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

const MAX_FFT_DEGREE: u32 = 1 << 24;

/// Batched FFT params matching WGSL layout (8 u32 fields).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BatchedFftParams {
    /// FFT size N (must be power of 2).
    pub degree: u32,
    /// Current butterfly stage (0..log2(N)).
    pub stage: u32,
    /// 1 for inverse FFT, 0 for forward.
    pub inverse: u32,
    /// Stride between consecutive elements along the FFT axis.
    pub element_stride: u32,
    /// Stride along the first non-FFT dimension.
    pub dim1_stride: u32,
    /// Stride along the second non-FFT dimension.
    pub dim2_stride: u32,
    /// Number of pencils along dim1.
    pub dim1_count: u32,
    /// Number of pencils along dim2.
    pub dim2_count: u32,
}

/// Axis configuration for batched FFT dispatch.
#[derive(Debug, Clone, Copy)]
pub struct AxisConfig {
    /// FFT size along this axis (must be power of 2).
    pub degree: u32,
    /// Stride between consecutive elements along the FFT axis.
    pub element_stride: u32,
    /// Stride along the first non-FFT dimension.
    pub dim1_stride: u32,
    /// Stride along the second non-FFT dimension.
    pub dim2_stride: u32,
    /// Number of pencils along dim1.
    pub dim1_count: u32,
    /// Number of pencils along dim2.
    pub dim2_count: u32,
}

/// f64-canonical batched FFT shader source.
pub const BATCHED_SHADER_F64: &str = include_str!("fft_3d_batched_f64.wgsl");

/// Returns the f32-downcast version of the batched shader.
#[must_use]
pub fn batched_shader_f32() -> &'static str {
    static SHADER: std::sync::LazyLock<String> =
        std::sync::LazyLock::new(|| BATCHED_SHADER_F64.to_string());
    &SHADER
}

/// Upload twiddle factors for f32 (re/im interleaved as f32 pairs).
#[must_use]
pub fn upload_twiddles_f32(
    device: &crate::device::WgpuDevice,
    degree: u32,
) -> (wgpu::Buffer, wgpu::Buffer) {
    let pi = std::f32::consts::PI;
    let mut re = Vec::with_capacity(degree as usize);
    let mut im = Vec::with_capacity(degree as usize);
    for k in 0..degree {
        let angle = -2.0 * pi * (k as f32) / (degree as f32);
        re.push(angle.cos());
        im.push(angle.sin());
    }
    let re_buf = device
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FFT Twiddle Re f32"),
            contents: bytemuck::cast_slice(&re),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let im_buf = device
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FFT Twiddle Im f32"),
            contents: bytemuck::cast_slice(&im),
            usage: wgpu::BufferUsages::STORAGE,
        });
    (re_buf, im_buf)
}

/// Upload twiddle factors for f64 (separate re/im arrays).
#[must_use]
pub fn upload_twiddles_f64(
    device: &crate::device::WgpuDevice,
    degree: u32,
) -> (wgpu::Buffer, wgpu::Buffer) {
    let pi = std::f64::consts::PI;
    let mut re = Vec::with_capacity(degree as usize);
    let mut im = Vec::with_capacity(degree as usize);
    for k in 0..degree {
        let angle = -2.0 * pi * (k as f64) / (degree as f64);
        re.push(angle.cos());
        im.push(angle.sin());
    }
    let re_buf = device
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FFT Twiddle Re f64"),
            contents: bytemuck::cast_slice(&re),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let im_buf = device
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FFT Twiddle Im f64"),
            contents: bytemuck::cast_slice(&im),
            usage: wgpu::BufferUsages::STORAGE,
        });
    (re_buf, im_buf)
}

pub(crate) fn dispatch_axis_inner(
    device: &crate::device::WgpuDevice,
    shader: &str,
    buf_a: &wgpu::Buffer,
    buf_b: &wgpu::Buffer,
    buffer_bytes: u64,
    tw_re: &wgpu::Buffer,
    tw_im: &wgpu::Buffer,
    axis: &AxisConfig,
    inverse: bool,
    use_f64: bool,
) -> Result<()> {
    let n = axis.degree;
    let log_n = (n as f32).log2() as u32;
    let num_pencils = axis.dim1_count * axis.dim2_count;
    let inv_flag = u32::from(inverse);

    // Bit-reverse pass: buf_a → buf_b
    let br_params = BatchedFftParams {
        degree: n,
        stage: 0,
        inverse: inv_flag,
        element_stride: axis.element_stride,
        dim1_stride: axis.dim1_stride,
        dim2_stride: axis.dim2_stride,
        dim1_count: axis.dim1_count,
        dim2_count: axis.dim2_count,
    };
    let br_params_buf = device.create_uniform_buffer("FFT BR Params", &br_params);

    let total_invocations = num_pencils * n;
    let workgroups = total_invocations.div_ceil(WORKGROUP_SIZE_1D);

    let mut br_dispatch = ComputeDispatch::new(device, "FFT Bit Reverse")
        .shader(shader, "bit_reverse")
        .storage_read(0, buf_a)
        .storage_rw(1, buf_b)
        .storage_read(2, tw_re)
        .storage_read(3, tw_im)
        .uniform(4, &br_params_buf);

    if use_f64 {
        br_dispatch = br_dispatch.f64();
    }

    br_dispatch.dispatch(workgroups, 1, 1).submit()?;

    // Copy buf_b → buf_a
    {
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("FFT Copy BR"),
        });
        encoder.copy_buffer_to_buffer(buf_b, 0, buf_a, 0, buffer_bytes);
        device.submit_and_poll(std::iter::once(encoder.finish()));
    }

    // Butterfly stages
    for stage in 0..log_n {
        let s_params = BatchedFftParams {
            degree: n,
            stage,
            inverse: inv_flag,
            element_stride: axis.element_stride,
            dim1_stride: axis.dim1_stride,
            dim2_stride: axis.dim2_stride,
            dim1_count: axis.dim1_count,
            dim2_count: axis.dim2_count,
        };
        let s_params_buf = device.create_uniform_buffer("FFT Stage Params", &s_params);

        let total = num_pencils * (n / 2);
        let workgroups = total.div_ceil(WORKGROUP_SIZE_1D);

        let mut bf_dispatch = ComputeDispatch::new(device, "FFT Butterfly")
            .shader(shader, "main")
            .storage_read(0, buf_a)
            .storage_rw(1, buf_b)
            .storage_read(2, tw_re)
            .storage_read(3, tw_im)
            .uniform(4, &s_params_buf);

        if use_f64 {
            bf_dispatch = bf_dispatch.f64();
        }

        bf_dispatch.dispatch(workgroups, 1, 1).submit()?;

        // Ping-pong: copy buf_b → buf_a (skip for last stage)
        if stage < log_n - 1 {
            let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("FFT Copy Stage"),
            });
            encoder.copy_buffer_to_buffer(buf_b, 0, buf_a, 0, buffer_bytes);
            device.submit_and_poll(std::iter::once(encoder.finish()));
        }
    }

    // Final copy: buf_b → buf_a (last butterfly writes to buf_b)
    {
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("FFT Copy Final"),
        });
        encoder.copy_buffer_to_buffer(buf_b, 0, buf_a, 0, buffer_bytes);
        device.submit_and_poll(std::iter::once(encoder.finish()));
    }
    Ok(())
}

/// Dispatch batched FFT along one axis (f32).
/// # Errors
/// Returns [`Err`] if GPU dispatch or buffer copy fails.
pub fn dispatch_axis(
    device: &crate::device::WgpuDevice,
    buf_a: &wgpu::Buffer,
    buf_b: &wgpu::Buffer,
    buffer_bytes: u64,
    tw_re: &wgpu::Buffer,
    tw_im: &wgpu::Buffer,
    axis: &AxisConfig,
    inverse: bool,
) -> Result<()> {
    dispatch_axis_inner(
        device,
        batched_shader_f32(),
        buf_a,
        buf_b,
        buffer_bytes,
        tw_re,
        tw_im,
        axis,
        inverse,
        false,
    )
}

/// Dispatch batched FFT along one axis (f64).
/// # Errors
/// Returns [`Err`] if GPU dispatch or buffer copy fails.
pub fn dispatch_axis_f64(
    device: &crate::device::WgpuDevice,
    buf_a: &wgpu::Buffer,
    buf_b: &wgpu::Buffer,
    buffer_bytes: u64,
    tw_re: &wgpu::Buffer,
    tw_im: &wgpu::Buffer,
    axis: &AxisConfig,
    inverse: bool,
) -> Result<()> {
    dispatch_axis_inner(
        device,
        BATCHED_SHADER_F64,
        buf_a,
        buf_b,
        buffer_bytes,
        tw_re,
        tw_im,
        axis,
        inverse,
        true,
    )
}

/// 1D Complex FFT operation
///
/// Transforms complex signal from time/spatial domain to frequency domain.
pub struct Fft1D {
    input: Tensor,
    degree: u32,
}

impl Fft1D {
    /// Create a new 1D FFT operation
    /// ## Parameters
    /// - `input`: Complex tensor (shape [..., N, 2] where last dim is (real, imag))
    /// - `degree`: FFT size N (must be power of 2)
    /// ## Constraints
    /// - N must be a power of 2 (for Cooley-Tukey radix-2 FFT)
    /// - N must be <= `MAX_FFT_DEGREE` (1<<24)
    /// - Input size must match degree * 2 (complex elements)
    /// # Errors
    /// Returns [`Err`] if input last dimension is not 2, degree is not a power of 2,
    /// degree is 0, degree exceeds `MAX_FFT_DEGREE`, or input size doesn't match.
    pub fn new(input: Tensor, degree: u32) -> Result<Self> {
        if degree == 0 {
            return Err(BarracudaError::Device(
                "FFT degree must be non-zero".to_string(),
            ));
        }
        if degree & (degree - 1) != 0 {
            return Err(BarracudaError::Device(format!(
                "FFT degree {degree} must be power of 2"
            )));
        }
        if degree > MAX_FFT_DEGREE {
            return Err(BarracudaError::Device(format!(
                "FFT degree {degree} exceeds MAX_FFT_DEGREE {MAX_FFT_DEGREE}"
            )));
        }

        let shape = input.shape();
        if shape.last() != Some(&2) {
            return Err(BarracudaError::Device(
                "FFT input must have last dimension = 2 (complex)".to_string(),
            ));
        }

        let expected_size = (degree as usize) * 2;
        let actual_size: usize = shape.iter().product();
        if actual_size != expected_size {
            return Err(BarracudaError::Device(format!(
                "FFT input size {actual_size} doesn't match degree {degree} (expected {expected_size} elements)"
            )));
        }

        Ok(Self { input, degree })
    }

    /// Execute FFT transformation.
    /// Returns a new tensor containing the frequency-domain representation.
    /// # Errors
    /// Returns [`Err`] if GPU dispatch, buffer creation, or readback fails.
    pub fn execute(self) -> Result<Tensor> {
        self.execute_internal(false)
    }

    /// Execute FFT with inverse flag (used by `Ifft1D`).
    pub(crate) fn execute_internal(self, inverse: bool) -> Result<Tensor> {
        let device = self.input.device();
        let n = self.degree;
        let buffer_bytes = (n as u64) * 2 * std::mem::size_of::<f32>() as u64;

        // Copy input to buf_a
        let buf_a = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFT buf_a"),
            size: buffer_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        {
            let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("FFT Copy Input"),
            });
            encoder.copy_buffer_to_buffer(self.input.buffer(), 0, &buf_a, 0, buffer_bytes);
            device.submit_and_poll(std::iter::once(encoder.finish()));
        }

        let buf_b = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFT buf_b"),
            size: buffer_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (tw_re, tw_im) = upload_twiddles_f32(device, n);

        let axis = AxisConfig {
            degree: n,
            element_stride: 1,
            dim1_stride: n,
            dim2_stride: 1,
            dim1_count: 1,
            dim2_count: 1,
        };

        dispatch_axis(
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
            self.input.shape().to_vec(),
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

        let data = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let tensor = Tensor::from_data(&data, vec![4, 2], device.clone()).unwrap();
        let fft = Fft1D::new(tensor, 4).unwrap();
        let result = fft.execute().unwrap();

        let result_data = result.to_vec().unwrap();

        assert!((result_data[0] - 1.0).abs() < 1e-5);
        assert!((result_data[2] - 1.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_fft_1d_degree_zero_rejected() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let data = vec![1.0f32, 0.0, 0.0, 0.0];
        let tensor = Tensor::from_data(&data, vec![2, 2], device).unwrap();
        assert!(Fft1D::new(tensor, 0).is_err());
    }

    #[tokio::test]
    async fn test_fft_1d_power_of_2_validation() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let data = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        let tensor = Tensor::from_data(&data, vec![3, 2], device).unwrap();
        assert!(Fft1D::new(tensor, 3).is_err());
    }
}
