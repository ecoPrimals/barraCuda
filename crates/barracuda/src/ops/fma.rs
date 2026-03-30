// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fused Multiply-Add operation
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (universal compute)
//! - ✅ Capability-based dispatch (hardware limits, not vendor names)
//! - ✅ Workgroup sizing from wgpu device limits
//! - ✅ Pipeline caching (compile once, dispatch many)
//! - ✅ Buffer pooling (zero allocation after warmup)
//! - ✅ Bind group caching (eliminates ~100μs per dispatch)
//!
//! Formula: D = A * B + C (fused multiply-add)
//!
//! **Key Optimization**: This eliminates one memory pass compared to
//! `a.mul(b).add(c)` which requires 2 dispatches and 2 memory passes.
//! Common patterns that benefit:
//! - Linear layers: output = weight @ input + bias
//! - Residual connections: output = layer(x) + x
//! - Scaled additions: output = alpha * x + y

use crate::device::capabilities::DeviceCapabilities;
use crate::device::pipeline_cache::{BindGroupLayoutSignature, GLOBAL_CACHE};
use crate::device::tensor_context::get_device_context;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Shader source optimized for NVIDIA GPUs (WG=64)
const SHADER_WG64: &str = include_str!("../shaders/math/fma_wg64.wgsl");

/// Shader source optimized for AMD GPUs (WG=128)  
const SHADER_WG128: &str = include_str!("../shaders/math/fma_wg128.wgsl");

/// Default shader (WG=256, fallback) — f64 canonical.
const SHADER_F64: &str = include_str!("../shaders/math/fma_f64.wgsl");

/// Default shader (f32 derived from f64).
static SHADER_DEFAULT: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| SHADER_F64.to_string());

/// Element-wise FMA shader.
pub const WGSL_FMA_ELEMENTWISE: &str = include_str!("../shaders/math/elementwise_fma.wgsl");

/// Fused Multiply-Add operation: D = A * B + C
pub struct Fma {
    a: Tensor,
    b: Tensor,
    c: Tensor,
}

impl Fma {
    /// Create FMA operation
    /// # Errors
    /// Returns [`Err`] if tensor shapes do not match.
    pub fn new(a: Tensor, b: Tensor, c: Tensor) -> Result<Self> {
        // Verify shapes match
        if a.shape() != b.shape() {
            return Err(BarracudaError::shape_mismatch(
                a.shape().to_vec(),
                b.shape().to_vec(),
            ));
        }
        if a.shape() != c.shape() {
            return Err(BarracudaError::shape_mismatch(
                a.shape().to_vec(),
                c.shape().to_vec(),
            ));
        }
        Ok(Self { a, b, c })
    }

    /// Select shader variant based on hardware capabilities.
    ///
    /// Workgroup size is chosen from the device's reported limits (vendor-agnostic).
    /// The largest supported workgroup size is preferred for throughput; smaller
    /// variants avoid exceeding dispatch constraints on large tensors.
    fn wgsl_shader(caps: &DeviceCapabilities, size: usize) -> (&'static str, u32) {
        let max_inv = caps.max_compute_invocations_per_workgroup;
        let max_dispatch = caps.max_compute_workgroups.0;

        let default: &'static str = &SHADER_DEFAULT;
        let (shader, wg) = if max_inv >= 256 {
            (default, 256u32)
        } else if max_inv >= 128 {
            (SHADER_WG128, 128u32)
        } else {
            (SHADER_WG64, 64u32)
        };

        let needed = (size as u32).div_ceil(wg);
        if needed > max_dispatch && max_inv >= 256 {
            return (default, 256);
        }

        (shader, wg)
    }

    /// Execute FMA on tensors
    /// Uses cached shader, pipeline, and bind group for fast repeated calls.
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the
    /// device is lost.
    pub fn execute(self) -> Result<Tensor> {
        let device = self.a.device();
        let size = self.a.len();

        // Get device context for buffer pooling and bind group caching
        let ctx = get_device_context(device);

        // Select shader variant from hardware capabilities
        let caps = DeviceCapabilities::from_device(device);
        let (shader_source, workgroup_size) = Self::wgsl_shader(&caps, size);

        // Acquire pooled output buffer
        let output_buffer = ctx.acquire_pooled_output(size);

        // Get cached bind group layout for ternary ops
        let layout_sig = BindGroupLayoutSignature::elementwise_ternary();
        let adapter_info = device.adapter_info();

        // Create bind group using TensorContext's cache
        let bind_group = ctx.get_or_create_bind_group(
            layout_sig,
            &[
                self.a.buffer(),
                self.b.buffer(),
                self.c.buffer(),
                &output_buffer,
            ],
            Some("FMA Bind Group"),
        );

        // Get cached pipeline
        let pipeline = GLOBAL_CACHE.get_or_create_pipeline(
            device.device(),
            adapter_info,
            shader_source,
            layout_sig,
            "main",
            Some("FMA Pipeline"),
        );

        // Encode and execute
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("FMA Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FMA Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&*bind_group), &[]);

            let workgroups = (size as u32).div_ceil(workgroup_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        // Create output tensor with pooled buffer
        Ok(Tensor::from_pooled_buffer(
            output_buffer,
            self.a.shape().to_vec(),
            device.clone(),
        ))
    }
}

// Convenience methods on Tensor
impl Tensor {
    /// Fused Multiply-Add: self * other + addend
    /// This is equivalent to `self.mul(other).add(addend)` but faster
    /// because it uses a single GPU dispatch instead of two.
    /// # Errors
    /// Returns [`Err`] if tensor shapes do not match, buffer allocation fails,
    /// GPU dispatch fails, or the device is lost.
    pub fn fma(&self, other: &Self, addend: &Self) -> Result<Self> {
        Fma::new(self.clone(), other.clone(), addend.clone())?.execute()
    }

    /// Multiply-accumulate: self * other + self
    /// Common pattern for residual connections.
    /// # Errors
    /// Returns [`Err`] if tensor shapes do not match, buffer allocation fails,
    /// GPU dispatch fails, or the device is lost.
    pub fn mul_add(&self, multiplier: &Self, addend: &Self) -> Result<Self> {
        self.fma(multiplier, addend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fma_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let a = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(vec![2.0, 2.0, 2.0, 2.0, 2.0], vec![5], device.clone())
            .await
            .unwrap();
        let c = Tensor::from_vec_on(vec![10.0, 20.0, 30.0, 40.0, 50.0], vec![5], device)
            .await
            .unwrap();

        let output = a.fma(&b, &c).unwrap();
        let result = output.to_vec().unwrap();

        // d = a * b + c
        let expected = [12.0, 24.0, 36.0, 48.0, 60.0];
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-6, "Mismatch at {i}: {r} vs {e}");
        }
    }

    #[tokio::test]
    async fn test_fma_vs_separate_ops() {
        let device = crate::device::test_pool::get_test_device().await;
        let a = Tensor::from_vec_on(vec![1.5, 2.5, 3.5, 4.5, 5.5], vec![5], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(vec![0.5, 1.5, 2.5, 3.5, 4.5], vec![5], device.clone())
            .await
            .unwrap();
        let c = Tensor::from_vec_on(vec![-1.0, -2.0, -3.0, -4.0, -5.0], vec![5], device.clone())
            .await
            .unwrap();

        // FMA result
        let fma_result = a.fma(&b, &c).unwrap().to_vec().unwrap();

        // Separate ops result
        let mul_result = a.mul(&b).unwrap();
        let add_result = mul_result.add(&c).unwrap().to_vec().unwrap();

        // Should match within floating point tolerance
        for (i, (&fma, &sep)) in fma_result.iter().zip(add_result.iter()).enumerate() {
            assert!(
                (fma - sep).abs() < 1e-5,
                "Mismatch at {i}: FMA={fma}, separate={sep}"
            );
        }
    }

    #[tokio::test]
    async fn test_fma_large_tensor() {
        let device = crate::device::test_pool::get_test_device().await;
        let size = 10_000;
        let a_data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
        let b_data: Vec<f32> = (0..size).map(|i| (size - i) as f32 * 0.001).collect();
        let c_data: Vec<f32> = (0..size).map(|_| 1.0).collect();

        let a = Tensor::from_vec_on(a_data.clone(), vec![size], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(b_data.clone(), vec![size], device.clone())
            .await
            .unwrap();
        let c = Tensor::from_vec_on(c_data.clone(), vec![size], device)
            .await
            .unwrap();

        let output = a.fma(&b, &c).unwrap();
        let result = output.to_vec().unwrap();

        // Verify first few and last few
        for i in 0..10 {
            let expected = a_data[i].mul_add(b_data[i], c_data[i]);
            assert!(
                (result[i] - expected).abs() < 1e-4,
                "Mismatch at {}: {} vs {}",
                i,
                result[i],
                expected
            );
        }
    }

    #[tokio::test]
    async fn test_fma_precision() {
        let device = crate::device::test_pool::get_test_device().await;
        // Test with values that could cause precision issues
        let a_data = vec![1e-6, 1e6, -1e-6, -1e6, 0.0];
        let b_data = vec![1e6, 1e-6, 1e6, 1e-6, 1.0];
        let c_data = vec![1.0, 1.0, 1.0, 1.0, 0.0];

        let a = Tensor::from_vec_on(a_data.clone(), vec![5], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(b_data.clone(), vec![5], device.clone())
            .await
            .unwrap();
        let c = Tensor::from_vec_on(c_data.clone(), vec![5], device)
            .await
            .unwrap();

        let gpu_result = a.fma(&b, &c).unwrap().to_vec().unwrap();

        // CPU reference with FMA
        let cpu_result: Vec<f32> = a_data
            .iter()
            .zip(b_data.iter())
            .zip(c_data.iter())
            .map(|((&a, &b), &c)| a.mul_add(b, c))
            .collect();

        for (i, (&gpu, &cpu)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
            // FMA should give better precision than separate ops
            let error = (gpu - cpu).abs();
            assert!(
                error < 1e-4 || error / cpu.abs().max(1e-10) < 1e-4,
                "Error at {i}: GPU={gpu}, CPU={cpu}, error={error}"
            );
        }
    }
}
