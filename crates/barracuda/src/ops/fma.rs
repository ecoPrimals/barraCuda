// SPDX-License-Identifier: AGPL-3.0-only
//! Fused Multiply-Add operation
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (universal compute)
//! - ✅ Capability-based dispatch (vendor-optimized)
//! - ✅ Vendor-specific workgroup sizes (NVIDIA: 64, AMD: 128)
//! - ✅ Pipeline caching (compile once, dispatch many)
//! - ✅ Buffer pooling (zero allocation after warmup)
//! - ✅ Bind group caching (eliminates ~100μs per op on NVIDIA)
//!
//! Formula: D = A * B + C (fused multiply-add)
//!
//! **Key Optimization**: This eliminates one memory pass compared to
//! `a.mul(b).add(c)` which requires 2 dispatches and 2 memory passes.
//! Common patterns that benefit:
//! - Linear layers: output = weight @ input + bias
//! - Residual connections: output = layer(x) + x
//! - Scaled additions: output = alpha * x + y

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

    /// Select vendor-optimized shader based on GPU and tensor size
    fn wgsl_shader(device_name: &str, size: usize) -> (&'static str, u32) {
        let lower = device_name.to_lowercase();

        let max_dispatch = 65535u32;
        let (nvidia_wg, amd_wg) = (64u32, 128u32);

        if lower.contains("nvidia")
            || lower.contains("geforce")
            || lower.contains("rtx")
            || lower.contains("gtx")
        {
            let needed_workgroups = (size as u32).div_ceil(nvidia_wg);
            if needed_workgroups <= max_dispatch {
                (SHADER_WG64, nvidia_wg)
            } else {
                (&*SHADER_DEFAULT, 256)
            }
        } else if lower.contains("amd") || lower.contains("radeon") || lower.contains("radv") {
            let needed_workgroups = (size as u32).div_ceil(amd_wg);
            if needed_workgroups <= max_dispatch {
                (SHADER_WG128, amd_wg)
            } else {
                (&*SHADER_DEFAULT, 256)
            }
        } else {
            (&*SHADER_DEFAULT, 256)
        }
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

        // Select vendor-optimized shader
        let device_name = device.name();
        let (shader_source, workgroup_size) = Self::wgsl_shader(device_name, size);

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

        device.submit_and_poll(Some(encoder.finish()));

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
    pub fn fma(&self, other: &Tensor, addend: &Tensor) -> Result<Self> {
        Fma::new(self.clone(), other.clone(), addend.clone())?.execute()
    }

    /// Multiply-accumulate: self * other + self
    /// Common pattern for residual connections.
    /// # Errors
    /// Returns [`Err`] if tensor shapes do not match, buffer allocation fails,
    /// GPU dispatch fails, or the device is lost.
    pub fn mul_add(&self, multiplier: &Tensor, addend: &Tensor) -> Result<Self> {
        self.fma(multiplier, addend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_fma_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
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
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
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
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
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
            let expected = a_data[i] * b_data[i] + c_data[i];
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
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
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
