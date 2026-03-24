// SPDX-License-Identifier: AGPL-3.0-or-later
//! Element-wise multiplication
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (universal compute)
//! - ✅ Capability-based dispatch (hardware limits, not vendor IDs)
//! - ✅ Workgroup sizing from wgpu device limits
//! - ✅ Pipeline caching (compile once, dispatch many)
//! - ✅ Buffer pooling (zero allocation after warmup)
//!
//! Formula: C = A * B (element-wise, Hadamard product)

use crate::device::capabilities::DeviceCapabilities;
use crate::device::pipeline_cache::{BindGroupLayoutSignature, GLOBAL_CACHE};
use crate::device::tensor_context::get_device_context;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Shader source optimized for NVIDIA GPUs (WG=64)
const SHADER_WG64: &str = include_str!("../shaders/math/elementwise_mul_wg64.wgsl");

/// Shader source optimized for AMD GPUs (WG=128)  
const SHADER_WG128: &str = include_str!("../shaders/math/elementwise_mul_wg128.wgsl");

/// f64 is the canonical source — math is universal, precision is silicon.
pub const WGSL_MUL_F64: &str = include_str!("../shaders/math/elementwise_mul_f64.wgsl");

/// f32 variant derived from f64 via precision downcast.
static SHADER_DEFAULT: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| WGSL_MUL_F64.to_string());

/// Optimized element-wise mul variant.
pub const WGSL_MUL_OPTIMIZED: &str = include_str!("../shaders/math/elementwise_mul_optimized.wgsl");

/// Element-wise multiplication operation
pub struct Mul {
    lhs: Tensor,
    rhs: Tensor,
}

impl Mul {
    /// Create Mul operation
    /// # Errors
    /// Returns [`Err`] if lhs and rhs shapes do not match.
    pub fn new(lhs: Tensor, rhs: Tensor) -> Result<Self> {
        if lhs.shape() != rhs.shape() {
            return Err(BarracudaError::shape_mismatch(
                lhs.shape().to_vec(),
                rhs.shape().to_vec(),
            ));
        }
        Ok(Self { lhs, rhs })
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

    /// Execute multiplication on tensors
    ///
    /// Uses cached shader and pipeline for fast repeated calls.
    /// Output buffer is acquired from pool for zero-allocation steady-state.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.lhs.device();
        let size = self.lhs.len();

        // Get device context for buffer pooling
        let ctx = get_device_context(device);

        // Get device capabilities for vendor-based shader selection
        let caps = DeviceCapabilities::from_device(device);
        let (shader_source, workgroup_size) = Self::wgsl_shader(&caps, size);

        // Acquire pooled output buffer (returns to pool when tensor dropped!)
        let output_buffer = ctx.acquire_pooled_output(size);

        // Get cached bind group layout
        let layout_sig = BindGroupLayoutSignature::elementwise_binary();
        let adapter_info = device.adapter_info();

        // Create bind group using TensorContext's cache
        // This is a key optimization: bind group creation is ~100μs on NVIDIA
        // Caching by (layout, buffer_ids) allows reuse when same tensors are used repeatedly
        let bind_group = ctx.get_or_create_bind_group(
            layout_sig,
            &[self.lhs.buffer(), self.rhs.buffer(), &output_buffer],
            Some("Mul Bind Group"),
        );

        // Get cached pipeline (compiles shader on first call, reuses after)
        let pipeline = GLOBAL_CACHE.get_or_create_pipeline(
            device.device(),
            adapter_info,
            shader_source,
            layout_sig,
            "main",
            Some("Mul Pipeline"),
        );

        // Encode and execute
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Mul Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Mul Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&*bind_group), &[]);

            // Use vendor-optimized workgroup size
            let workgroups = (size as u32).div_ceil(workgroup_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        // Create output tensor with pooled buffer (auto-returns to pool on drop!)
        Ok(Tensor::from_pooled_buffer(
            output_buffer,
            self.lhs.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Element-wise multiplication
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn mul(&self, other: &Self) -> Result<Self> {
        Mul::new(self.clone(), other.clone())?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_mul_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let lhs = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5], device.clone())
            .await
            .unwrap();
        let rhs = Tensor::from_vec_on(vec![2.0, 3.0, 4.0, 5.0, 6.0], vec![5], device)
            .await
            .unwrap();

        let output = lhs.mul(&rhs).unwrap();
        let result = output.to_vec().unwrap();

        let expected = [2.0, 6.0, 12.0, 20.0, 30.0];
        for (&r, &e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-5);
        }
    }

    #[tokio::test]
    async fn test_mul_edge_cases() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Zero multiplication, small values
        let lhs = Tensor::from_vec_on(vec![0.0, 1e-6, -1e-6, 1.0, -1.0], vec![5], device.clone())
            .await
            .unwrap();
        let rhs = Tensor::from_vec_on(vec![5.0, 1e6, 1e6, 0.0, -0.0], vec![5], device)
            .await
            .unwrap();

        let output = lhs.mul(&rhs).unwrap();
        let result = output.to_vec().unwrap();

        assert_eq!(result[0], 0.0); // 0 * 5 = 0
        assert!((result[1] - 1.0).abs() < 1e-4); // 1e-6 * 1e6 = 1
        assert!((result[2] + 1.0).abs() < 1e-4); // -1e-6 * 1e6 = -1
        assert_eq!(result[3], 0.0); // 1 * 0 = 0
    }

    #[tokio::test]
    async fn test_mul_boundary() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let lhs = Tensor::from_vec_on(
            vec![f32::NEG_INFINITY, -1e10, 0.0, 1e10, f32::INFINITY],
            vec![5],
            device.clone(),
        )
        .await
        .unwrap();

        let rhs = Tensor::from_vec_on(vec![2.0, 2.0, 2.0, 2.0, 2.0], vec![5], device)
            .await
            .unwrap();

        let output = lhs.mul(&rhs).unwrap();
        let result = output.to_vec().unwrap();

        assert!(result[0].is_infinite() && result[0].is_sign_negative());
        assert_eq!(result[2], 0.0);
        assert!(result[4].is_infinite() && result[4].is_sign_positive());
    }

    #[tokio::test]
    async fn test_mul_large_tensor() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let size = 1000;
        let lhs_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let rhs_data = vec![2.0; size];

        let lhs = Tensor::from_vec_on(lhs_data.clone(), vec![size], device.clone())
            .await
            .unwrap();
        let rhs = Tensor::from_vec_on(rhs_data, vec![size], device)
            .await
            .unwrap();

        let output = lhs.mul(&rhs).unwrap();
        let result = output.to_vec().unwrap();

        for (i, &val) in result.iter().enumerate() {
            assert!((i as f32).mul_add(-2.0, val).abs() < 1e-4);
        }
    }

    #[tokio::test]
    async fn test_mul_precision() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let lhs_data = vec![-5.0, -2.5, -1.0, 0.0, 1.0, 2.5, 5.0];
        let rhs_data = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let lhs = Tensor::from_vec_on(lhs_data.clone(), vec![7], device.clone())
            .await
            .unwrap();
        let rhs = Tensor::from_vec_on(rhs_data.clone(), vec![7], device)
            .await
            .unwrap();

        let output = lhs.mul(&rhs).unwrap();
        let gpu_result = output.to_vec().unwrap();

        let cpu_result: Vec<f32> = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(&a, &b)| a * b)
            .collect();

        for (i, (&gpu, &cpu)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
            assert!(
                (gpu - cpu).abs() < 1e-6,
                "Error at {i}: GPU={gpu}, CPU={cpu}"
            );
        }
    }
}
