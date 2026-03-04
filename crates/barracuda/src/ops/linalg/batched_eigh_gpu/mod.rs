// SPDX-License-Identifier: AGPL-3.0-or-later
//! Batched Eigenvalue Decomposition (eigh) - GPU-Accelerated Implementation (f64)
//!
//! Processes multiple symmetric matrices simultaneously.
//! **Use case**: HFB Hamiltonian diagonalization (52 nuclei, 20-50 dim each)
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (GPU-optimized)
//! - ✅ Full f64 precision via SPIR-V/Vulkan (bypasses CUDA fp64 throttle)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Batched parallel processing
//! - ✅ Runtime-configured matrix size and batch size
//!
//! ## Algorithm
//!
//! Jacobi eigenvalue algorithm for symmetric matrices, batched.
//! See `standard` (multi-dispatch) and `single_dispatch` (one submit for n≤32) modules.

mod params;
mod pipelines;
mod single_dispatch;
mod standard;
mod sweep;

use crate::device::capabilities::{CompilerKind, GpuDriverProfile};
use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use std::sync::Arc;

/// GPU-accelerated batched eigenvalue decomposition
///
/// Computes eigenvalue decomposition for multiple symmetric matrices simultaneously:
/// A_i = V_i · D_i · V_i^T for all i in batch
pub struct BatchedEighGpu;

impl BatchedEighGpu {
    fn wgsl_shader() -> &'static str {
        include_str!("../../../shaders/linalg/batched_eigh_f64.wgsl")
    }

    fn single_dispatch_shader() -> &'static str {
        include_str!("../../../shaders/linalg/batched_eigh_single_dispatch_f64.wgsl")
    }

    /// Select optimal eigensolve shader based on GPU driver.
    /// NAK (NVK open-source NVIDIA) uses hand-optimized variant with manual
    /// unrolling, hoisted locals, load-before-compute, explicit fma, and
    /// branchless select — workarounds for 5 NAK compiler deficiencies.
    fn single_dispatch_shader_for_device(device: &WgpuDevice) -> &'static str {
        let profile = GpuDriverProfile::from_device(device);
        if profile.compiler == CompilerKind::Nak {
            tracing::info!("BatchedEighGpu: using NAK-optimized eigensolve shader");
            include_str!("../../../shaders/linalg/batched_eigh_nak_optimized_f64.wgsl")
        } else {
            Self::single_dispatch_shader()
        }
    }

    /// Convenience method for processing a batch of matrices
    pub fn execute_batch(
        device: Arc<WgpuDevice>,
        matrices: &[Vec<f64>],
        n: usize,
    ) -> Result<Vec<(Vec<f64>, Vec<f64>)>> {
        if matrices.is_empty() {
            return Ok(vec![]);
        }

        for (i, m) in matrices.iter().enumerate() {
            if m.len() != n * n {
                return Err(BarracudaError::InvalidInput {
                    message: format!(
                        "Matrix {} has {} elements, expected {} for {}x{} matrix",
                        i,
                        m.len(),
                        n * n,
                        n,
                        n
                    ),
                });
            }
        }

        let batch_size = matrices.len();
        let packed: Vec<f64> = matrices.iter().flat_map(|m| m.iter().copied()).collect();
        let (eigenvalues_flat, eigenvectors_flat) =
            Self::execute_f64(device, &packed, n, batch_size, 30)?;

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let eig_start = i * n;
            let eig_end = eig_start + n;
            let vec_start = i * n * n;
            let vec_end = vec_start + n * n;
            results.push((
                eigenvalues_flat[eig_start..eig_end].to_vec(),
                eigenvectors_flat[vec_start..vec_end].to_vec(),
            ));
        }
        Ok(results)
    }

    /// Create GPU buffers for GPU-resident eigenvalue decomposition
    pub fn create_buffers(
        device: &Arc<WgpuDevice>,
        n: usize,
        batch_size: usize,
    ) -> Result<(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer)> {
        let matrices_size = (batch_size * n * n * 8) as u64;
        let matrices_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BatchedEigh matrices f64"),
            size: matrices_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let eigenvalues_size = (batch_size * n * 8) as u64;
        let eigenvalues_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BatchedEigh eigenvalues f64"),
            size: eigenvalues_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let eigenvectors_size = (batch_size * n * n * 8) as u64;
        let eigenvectors_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BatchedEigh eigenvectors f64"),
            size: eigenvectors_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Ok((matrices_buffer, eigenvalues_buffer, eigenvectors_buffer))
    }

    /// Read eigenvalues from a GPU buffer to CPU (for convergence checks)
    pub fn read_eigenvalues(
        device: &Arc<WgpuDevice>,
        eigenvalues_buffer: &wgpu::Buffer,
        n: usize,
        batch_size: usize,
    ) -> Result<Vec<f64>> {
        device.read_f64_buffer(eigenvalues_buffer, batch_size * n)
    }

    /// Read eigenvectors from a GPU buffer to CPU (optional, only when needed)
    pub fn read_eigenvectors(
        device: &Arc<WgpuDevice>,
        eigenvectors_buffer: &wgpu::Buffer,
        n: usize,
        batch_size: usize,
    ) -> Result<Vec<f64>> {
        device.read_f64_buffer(eigenvectors_buffer, batch_size * n * n)
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    const BATCHED_EIGH_SHADER: &str = include_str!("../../../shaders/linalg/batched_eigh_f64.wgsl");
    const BATCHED_EIGH_SINGLE_SHADER: &str =
        include_str!("../../../shaders/linalg/batched_eigh_single_dispatch_f64.wgsl");

    #[test]
    fn batched_eigh_shader_source_valid() {
        assert!(!BATCHED_EIGH_SHADER.is_empty());
        assert!(BATCHED_EIGH_SHADER.contains("fn ") || BATCHED_EIGH_SHADER.contains("@compute"));
    }

    #[test]
    fn batched_eigh_single_dispatch_shader_source_valid() {
        assert!(!BATCHED_EIGH_SINGLE_SHADER.is_empty());
        assert!(
            BATCHED_EIGH_SINGLE_SHADER.contains("fn ")
                || BATCHED_EIGH_SINGLE_SHADER.contains("@compute")
        );
    }

    fn approx_eq_f64(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[tokio::test]
    async fn test_batched_eigh_single_2x2() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };
        let data = vec![4.0_f64, 2.0, 2.0, 3.0];
        let (eigenvalues, eigenvectors) =
            BatchedEighGpu::execute_f64(device.clone(), &data, 2, 1, 30).unwrap();
        assert_eq!(eigenvalues.len(), 2);
        assert_eq!(eigenvectors.len(), 4);
        let trace = eigenvalues[0] + eigenvalues[1];
        assert!(
            approx_eq_f64(trace, 7.0, 1e-6),
            "Trace should be 7, got {}",
            trace
        );
        let det = eigenvalues[0] * eigenvalues[1];
        assert!(
            approx_eq_f64(det, 8.0, 1e-4),
            "Determinant should be 8, got {}",
            det
        );
    }

    #[tokio::test]
    async fn test_batched_eigh_identity_batch() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };
        let data = vec![
            1.0_f64, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
        ];
        let (eigenvalues, eigenvectors) =
            BatchedEighGpu::execute_f64(device.clone(), &data, 2, 3, 10).unwrap();
        assert_eq!(eigenvalues.len(), 6);
        assert_eq!(eigenvectors.len(), 12);
        for (i, &val) in eigenvalues.iter().enumerate() {
            assert!(
                approx_eq_f64(val, 1.0, 1e-6),
                "Eigenvalue {} should be 1, got {}",
                i,
                val
            );
        }
    }

    #[tokio::test]
    async fn test_batched_eigh_hfb_scale() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };
        let n = 20;
        let batch_size = 52;
        let mut data = vec![0.0_f64; batch_size * n * n];
        for b in 0..batch_size {
            for i in 0..n {
                data[b * n * n + i * n + i] = (i + 1) as f64;
            }
        }
        let result = BatchedEighGpu::execute_f64(device.clone(), &data, n, batch_size, 30);
        let (eigenvalues, _) = match result {
            Ok(v) => v,
            Err(e) if e.is_device_lost() => return,
            Err(e) => panic!("unexpected error: {e}"),
        };
        assert_eq!(eigenvalues.len(), batch_size * n);
        let first_sum: f64 = eigenvalues[0..n].iter().sum();
        assert!(
            approx_eq_f64(first_sum, 210.0, 1e-3),
            "First matrix sum should be 210"
        );
        let last_start = (batch_size - 1) * n;
        let last_sum: f64 = eigenvalues[last_start..last_start + n].iter().sum();
        assert!(
            approx_eq_f64(last_sum, 210.0, 1e-3),
            "Last matrix sum should be 210"
        );
    }

    #[tokio::test]
    async fn test_batched_eigh_execute_batch() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };
        let matrices = vec![
            vec![4.0_f64, 2.0, 2.0, 3.0],
            vec![1.0_f64, 0.0, 0.0, 1.0],
            vec![2.0_f64, 1.0, 1.0, 2.0],
        ];
        let results = BatchedEighGpu::execute_batch(device, &matrices, 2).unwrap();
        assert_eq!(results.len(), 3);
        assert!(approx_eq_f64(results[0].0.iter().sum(), 7.0, 1e-4));
        assert!(approx_eq_f64(results[1].0.iter().sum(), 2.0, 1e-4));
        assert!(approx_eq_f64(results[2].0.iter().sum(), 4.0, 1e-4));
    }

    #[tokio::test]
    async fn test_batched_eigh_buffer_api() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };
        let (matrices_buf, eigenvalues_buf, eigenvectors_buf) =
            BatchedEighGpu::create_buffers(&device, 2, 2).unwrap();
        let input_data: Vec<f64> = vec![4.0, 2.0, 2.0, 3.0, 2.0, 1.0, 1.0, 2.0];
        device
            .queue
            .write_buffer(&matrices_buf, 0, bytemuck::cast_slice(&input_data));
        BatchedEighGpu::execute_f64_buffers(
            &device,
            &matrices_buf,
            &eigenvalues_buf,
            &eigenvectors_buf,
            2,
            2,
            30,
        )
        .unwrap();
        let eigenvalues =
            BatchedEighGpu::read_eigenvalues(&device, &eigenvalues_buf, 2, 2).unwrap();
        assert_eq!(eigenvalues.len(), 4);
        assert!(approx_eq_f64(eigenvalues[0] + eigenvalues[1], 7.0, 1e-4));
        assert!(approx_eq_f64(eigenvalues[2] + eigenvalues[3], 4.0, 1e-4));
        let eigenvectors =
            BatchedEighGpu::read_eigenvectors(&device, &eigenvectors_buf, 2, 2).unwrap();
        assert_eq!(eigenvectors.len(), 8);
    }

    #[tokio::test]
    async fn test_batched_eigh_buffer_reuse() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };
        let (matrices_buf, eigenvalues_buf, eigenvectors_buf) =
            BatchedEighGpu::create_buffers(&device, 2, 1).unwrap();
        let data_1: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0];
        device
            .queue
            .write_buffer(&matrices_buf, 0, bytemuck::cast_slice(&data_1));
        BatchedEighGpu::execute_f64_buffers(
            &device,
            &matrices_buf,
            &eigenvalues_buf,
            &eigenvectors_buf,
            2,
            1,
            10,
        )
        .unwrap();
        let eig_1 = BatchedEighGpu::read_eigenvalues(&device, &eigenvalues_buf, 2, 1).unwrap();
        assert!(approx_eq_f64(eig_1.iter().sum(), 2.0, 1e-6));
        let data_2: Vec<f64> = vec![3.0, 1.0, 1.0, 3.0];
        device
            .queue
            .write_buffer(&matrices_buf, 0, bytemuck::cast_slice(&data_2));
        BatchedEighGpu::execute_f64_buffers(
            &device,
            &matrices_buf,
            &eigenvalues_buf,
            &eigenvectors_buf,
            2,
            1,
            10,
        )
        .unwrap();
        let eig_2 = BatchedEighGpu::read_eigenvalues(&device, &eigenvalues_buf, 2, 1).unwrap();
        assert!(approx_eq_f64(eig_2.iter().sum(), 6.0, 1e-4));
    }

    #[tokio::test]
    async fn test_single_dispatch_basic() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };
        let data: Vec<f64> = vec![4.0, 2.0, 2.0, 3.0, 2.0, 1.0, 1.0, 2.0];
        let (eigenvalues, eigenvectors) =
            BatchedEighGpu::execute_single_dispatch(device.clone(), &data, 2, 2, 30, 1e-12)
                .unwrap();
        assert_eq!(eigenvalues.len(), 4);
        assert_eq!(eigenvectors.len(), 8);
        assert!(approx_eq_f64(eigenvalues[0] + eigenvalues[1], 7.0, 1e-4));
        assert!(approx_eq_f64(eigenvalues[2] + eigenvalues[3], 4.0, 1e-4));
    }

    #[tokio::test]
    async fn test_single_dispatch_hotspring_scale() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };
        let n = 12;
        let batch_size = 40;
        let mut data = vec![0.0_f64; batch_size * n * n];
        for b in 0..batch_size {
            for i in 0..n {
                data[b * n * n + i * n + i] = (i + 1) as f64;
            }
        }
        let (eigenvalues, _) = BatchedEighGpu::execute_single_dispatch(
            device.clone(),
            &data,
            n,
            batch_size,
            30,
            1e-12,
        )
        .unwrap();
        assert_eq!(eigenvalues.len(), batch_size * n);
        let first_sum: f64 = eigenvalues[0..n].iter().sum();
        assert!(approx_eq_f64(first_sum, 78.0, 1e-3));
        let last_start = (batch_size - 1) * n;
        let last_sum: f64 = eigenvalues[last_start..last_start + n].iter().sum();
        assert!(approx_eq_f64(last_sum, 78.0, 1e-3));
    }

    #[tokio::test]
    async fn test_single_dispatch_buffers() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };
        let (matrices_buf, eigenvalues_buf, eigenvectors_buf) =
            BatchedEighGpu::create_buffers(&device, 4, 3).unwrap();
        let mut data = vec![0.0_f64; 3 * 4 * 4];
        for b in 0..3 {
            for i in 0..4 {
                data[b * 16 + i * 4 + i] = ((b + 1) * (i + 1)) as f64;
            }
        }
        device
            .queue
            .write_buffer(&matrices_buf, 0, bytemuck::cast_slice(&data));
        BatchedEighGpu::execute_single_dispatch_buffers(
            &device,
            &matrices_buf,
            &eigenvalues_buf,
            &eigenvectors_buf,
            4,
            3,
            30,
            1e-12,
        )
        .unwrap();
        let eigenvalues =
            BatchedEighGpu::read_eigenvalues(&device, &eigenvalues_buf, 4, 3).unwrap();
        let sum_0: f64 = eigenvalues[0..4].iter().sum();
        assert!(approx_eq_f64(sum_0, 10.0, 1e-4));
        let sum_1: f64 = eigenvalues[4..8].iter().sum();
        assert!(approx_eq_f64(sum_1, 20.0, 1e-4));
    }
}
