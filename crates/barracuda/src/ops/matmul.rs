// SPDX-License-Identifier: AGPL-3.0-or-later
//! `MatMul` operation - Matrix multiplication
//! Pure WGSL implementation
//!
//! **3-tier kernel router** (S-14: Naive tier removed):
//!
//! | Condition                          | Shader                   | Tile | Notes                       |
//! |------------------------------------|--------------------------|------|-----------------------------|
//! | CPU device (any size)              | `matmul_cpu_tiled.wgsl`  | 32   | Double-buffered, fma(), BLAS-style |
//! | GPU, M < 256 or N < 256 (medium)  | `matmul_tiled.wgsl`      | 16   | High occupancy              |
//! | GPU, M ≥ 256 and N ≥ 256 (large)  | `matmul_gpu_evolved.wgsl`| 32   | Double-buffered, 2×2 kernel |

use crate::device::DeviceCapabilities;
use crate::device::pipeline_cache::{BindGroupLayoutSignature, GLOBAL_CACHE};
use crate::device::tensor_context::get_device_context;
use crate::error::Result;
use crate::tensor::Tensor;

/// Large-matrix threshold for activating the evolved GPU shader.
/// Below this, the 16×16 tiled shader maintains higher SM occupancy.
const GPU_EVOLVED_THRESHOLD: usize = 256;

/// f64-emulated matmul (hi/lo f32 pairs).
pub const WGSL_MATMUL_FP64: &str = include_str!("../shaders/math/matmul_fp64.wgsl");

/// Linear layer shader (matmul + bias).
#[must_use]
pub fn wgsl_linear() -> &'static str {
    static SHADER: std::sync::LazyLock<String> =
        std::sync::LazyLock::new(|| include_str!("../shaders/misc/linear_f64.wgsl").to_string());
    std::sync::LazyLock::force(&SHADER).as_str()
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MatMulParams {
    m: u32,
    k: u32,
    n: u32,
    _padding: u32,
}

/// Which matmul implementation to dispatch.
///
/// S-14: Naive tier removed — Tiled16 is the minimum for all GPU sizes.
#[derive(Debug, Clone, Copy, PartialEq)]
enum MatMulTier {
    Tiled16,
    CpuTiled32,
    GpuEvolved32,
}

/// Matrix multiplication with tiered kernel selection.
pub struct MatMul<'a> {
    lhs: &'a Tensor,
    rhs: &'a Tensor,
}

impl<'a> MatMul<'a> {
    /// Creates a new matmul for lhs × rhs.
    #[must_use]
    pub fn new(lhs: &'a Tensor, rhs: &'a Tensor) -> Self {
        Self { lhs, rhs }
    }

    /// Select the appropriate matmul kernel tier based on device and matrix size.
    /// S-14: Naive tier removed — Tiled16 is the minimum for all GPU sizes.
    /// The naive shader caused hangs on small square matrices. Tiled16 has
    /// proper bounds-checking and shared-memory prefetch even at small sizes.
    fn select_tier(caps: &DeviceCapabilities, m: usize, n: usize) -> MatMulTier {
        if caps.device_type == wgpu::DeviceType::Cpu {
            return MatMulTier::CpuTiled32;
        }
        if m >= GPU_EVOLVED_THRESHOLD && n >= GPU_EVOLVED_THRESHOLD {
            MatMulTier::GpuEvolved32
        } else {
            MatMulTier::Tiled16
        }
    }

    fn shader_for_tier(tier: MatMulTier) -> &'static str {
        match tier {
            MatMulTier::Tiled16 => {
                static S_TILED: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                    include_str!("../shaders/math/matmul_tiled_f64.wgsl").to_string()
                });
                &S_TILED
            }
            MatMulTier::CpuTiled32 => {
                static S_CPU: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                    include_str!("../shaders/math/matmul_cpu_tiled_f64.wgsl").to_string()
                });
                &S_CPU
            }
            MatMulTier::GpuEvolved32 => {
                static S_GPU: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                    include_str!("../shaders/math/matmul_gpu_evolved_f64.wgsl").to_string()
                });
                &S_GPU
            }
        }
    }

    /// Executes matrix multiplication and returns the result.
    /// # Errors
    /// Returns [`Err`] if tensors are not 2D+, inner dimensions do not match (k ≠ `k_rhs`), buffer allocation fails, or GPU dispatch fails (e.g. device lost).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.lhs.device();

        if self.lhs.shape().len() < 2 || self.rhs.shape().len() < 2 {
            return Err(crate::error::BarracudaError::invalid_op(
                "matmul",
                format!(
                    "requires 2D+ tensors, got shapes {:?} and {:?}",
                    self.lhs.shape(),
                    self.rhs.shape()
                ),
            ));
        }

        // lhs: [m, k], rhs: [k, n] → output: [m, n]
        let m = self.lhs.shape()[0];
        let k = self.lhs.shape()[1];
        let k_rhs = self.rhs.shape()[0];
        let n = self.rhs.shape()[1];

        if k != k_rhs {
            return Err(crate::error::BarracudaError::shape_mismatch(
                self.lhs.shape().to_vec(),
                self.rhs.shape().to_vec(),
            ));
        }

        let output_size = m * n;

        let caps = DeviceCapabilities::from_device(device);
        let tier = Self::select_tier(&caps, m, n);
        tracing::debug!(
            "MatMul [{m}×{k}]×[{k}×{n}] → tier {:?} (device: {:?})",
            tier,
            caps.device_type
        );

        let ctx = get_device_context(device);
        // Pooled output — zero allocation in steady state.
        let output_buffer = ctx.acquire_pooled_output(output_size);

        let params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MatMul Params"),
                contents: bytemuck::bytes_of(&MatMulParams {
                    m: m as u32,
                    k: k as u32,
                    n: n as u32,
                    _padding: 0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // matmul() = (2 read-only, 1 read-write, 1 uniform)
        let layout_sig = BindGroupLayoutSignature::matmul();
        let adapter_info = device.adapter_info();
        let bgl = GLOBAL_CACHE.get_or_create_layout(
            device.device(),
            adapter_info,
            layout_sig,
            Some("MatMul BGL"),
        );

        let bind_group =
            std::sync::Arc::new(device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("MatMul BG"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.lhs.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.rhs.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            }));

        let pipeline = GLOBAL_CACHE.get_or_create_pipeline(
            device.device(),
            adapter_info,
            Self::shader_for_tier(tier),
            layout_sig,
            "main",
            Some(match tier {
                MatMulTier::Tiled16 => "MatMul Tiled16",
                MatMulTier::CpuTiled32 => "MatMul CpuTiled32",
                MatMulTier::GpuEvolved32 => "MatMul GpuEvolved32",
            }),
        );

        // Dispatch parameters (computed before the closure so they are Copy).
        let (wg_x, wg_y) = match tier {
            MatMulTier::Tiled16 => ((n as u32).div_ceil(16), (m as u32).div_ceil(16)),
            MatMulTier::CpuTiled32 | MatMulTier::GpuEvolved32 => {
                ((n as u32).div_ceil(32), (m as u32).div_ceil(32))
            }
        };

        ctx.record_operation(move |encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MatMul Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&*bind_group), &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
            drop(params_buf);
        })?;

        Ok(Tensor::from_pooled_buffer(
            output_buffer,
            vec![m, n],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Matrix multiplication
    /// **Phase 3**: Now supports NPU routing!
    /// Automatically routes to best device:
    /// - NPU if sparse data or energy priority
    /// - GPU/CPU via WGSL otherwise
    /// # Example
    /// ```ignore
    /// let a = Tensor::randn(vec![128, 64]).await?;
    /// let b = Tensor::randn(vec![64, 32]).await?;
    /// let c = a.matmul(&b)?;  // Routes to best device!
    /// ```
    /// # Errors
    /// Returns [`Err`] if tensors are not 2D+, inner dimensions do not match, buffer allocation fails, or GPU dispatch fails (e.g. device lost).
    pub fn matmul(self, other: &Self) -> Result<Self> {
        self.matmul_ref(other)
    }

    /// Non-consuming matrix multiply: `C[m×n] = self[m×k] × other[k×n]`.
    /// Unlike [`matmul`](Self::matmul), this borrows `self` so it can be
    /// reused in recurrent architectures (ESN, LSTM) without cloning.
    /// # Errors
    /// Returns [`Err`] if tensors are not 2D+, inner dimensions do not match, buffer allocation fails, or GPU dispatch fails (e.g. device lost).
    pub fn matmul_ref(&self, other: &Self) -> Result<Self> {
        tracing::debug!("Routing matmul_ref to WGSL (GPU/CPU)");
        MatMul::new(self, other).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    fn matmul_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut result = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                result[i * n + j] = sum;
            }
        }
        result
    }

    #[tokio::test]
    async fn test_matmul_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // 2x3 * 3x2 = 2x2
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let a = Tensor::from_vec_on(a_data.clone(), vec![2, 3], device.clone())
            .await
            .unwrap();

        let b = Tensor::from_vec_on(b_data.clone(), vec![3, 2], device.clone())
            .await
            .unwrap();

        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);

        let output = result.to_vec().unwrap();
        let expected = matmul_cpu(&a_data, &b_data, 2, 3, 2);

        for (r, e) in output.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-5);
        }
    }

    #[tokio::test]
    async fn test_matmul_edge_cases() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Identity matrix
        let a_data = vec![1.0, 0.0, 0.0, 1.0];
        let b_data = vec![5.0, 6.0, 7.0, 8.0];

        let a = Tensor::from_vec_on(a_data.clone(), vec![2, 2], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(b_data.clone(), vec![2, 2], device.clone())
            .await
            .unwrap();

        let result = a.matmul(&b).unwrap();
        let output = result.to_vec().unwrap();
        let expected = matmul_cpu(&a_data, &b_data, 2, 2, 2);

        for (r, e) in output.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-5);
        }

        // Zero matrix
        let a_data = vec![0.0, 0.0, 0.0, 0.0];
        let b_data = vec![1.0, 2.0, 3.0, 4.0];

        let a = Tensor::from_vec_on(a_data.clone(), vec![2, 2], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(b_data.clone(), vec![2, 2], device.clone())
            .await
            .unwrap();

        let result = a.matmul(&b).unwrap();
        let output = result.to_vec().unwrap();

        for val in &output {
            assert!(val.abs() < 1e-6);
        }
    }

    #[tokio::test]
    async fn test_matmul_boundary() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // 1x1 matrices
        let a_data = vec![5.0];
        let b_data = vec![3.0];

        let a = Tensor::from_vec_on(a_data.clone(), vec![1, 1], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(b_data.clone(), vec![1, 1], device.clone())
            .await
            .unwrap();

        let result = a.matmul(&b).unwrap();
        let output = result.to_vec().unwrap();
        assert!((output[0] - 15.0).abs() < 1e-5);

        // Tall matrix: 4x2 * 2x3 = 4x3
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b_data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

        let a = Tensor::from_vec_on(a_data.clone(), vec![4, 2], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(b_data.clone(), vec![2, 3], device.clone())
            .await
            .unwrap();

        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape(), &[4, 3]);

        let output = result.to_vec().unwrap();
        let expected = matmul_cpu(&a_data, &b_data, 4, 2, 3);

        for (r, e) in output.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-5);
        }
    }

    #[tokio::test]
    async fn test_matmul_large_tensor() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // 64x32 * 32x64 = 64x64
        let m = 64;
        let k = 32;
        let n = 64;

        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();

        let a = Tensor::from_vec_on(a_data.clone(), vec![m, k], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(b_data.clone(), vec![k, n], device.clone())
            .await
            .unwrap();

        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape(), &[m, n]);

        let output = result.to_vec().unwrap();
        let expected = matmul_cpu(&a_data, &b_data, m, k, n);

        for (r, e) in output.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-3); // Slightly relaxed for large accumulations
        }
    }

    #[tokio::test]
    async fn test_matmul_precision() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Test FP32 precision with typical values
        let a_data = vec![1.234, 2.345, 3.456, 4.567, 5.678, 6.789];
        let b_data = vec![0.111, 0.222, 0.333, 0.444, 0.555, 0.666];

        let a = Tensor::from_vec_on(a_data.clone(), vec![2, 3], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(b_data.clone(), vec![3, 2], device.clone())
            .await
            .unwrap();

        let result = a.matmul(&b).unwrap();
        let output = result.to_vec().unwrap();
        let expected = matmul_cpu(&a_data, &b_data, 2, 3, 2);

        // Verify FP32 precision
        let max_error = output
            .iter()
            .zip(expected.iter())
            .map(|(r, e)| (r - e).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_error < 1e-5,
            "Max error: {max_error} exceeds FP32 threshold"
        );
    }
}
