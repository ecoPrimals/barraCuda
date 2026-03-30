// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cdist - Pairwise distance computation - Pure WGSL
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (GPU-optimized)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//!
//! ## Algorithm
//!
//! Computes pairwise distances between vectors:
//! ```text
//! Input A: [M, D] - M vectors of dimension D
//! Input B: [N, D] - N vectors of dimension D
//! Output:  [M, N] - Distance matrix
//!
//! Supports: Euclidean (L2), Manhattan (L1), Cosine
//! ```

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// Distance metric for pairwise distance computation.
#[derive(Clone, Copy)]
pub enum DistanceMetric {
    /// L2 (Euclidean) distance.
    Euclidean = 0,
    /// L1 (Manhattan) distance.
    Manhattan = 1,
    /// Cosine distance (1 - cosine similarity).
    Cosine = 2,
}

/// Pairwise distance computation between two sets of vectors.
pub struct Cdist {
    input_a: Tensor,
    input_b: Tensor,
    metric: DistanceMetric,
}

impl Cdist {
    /// Create a cdist op with two input tensors and distance metric.
    #[must_use]
    pub fn new(input_a: Tensor, input_b: Tensor, metric: DistanceMetric) -> Self {
        Self {
            input_a,
            input_b,
            metric,
        }
    }

    /// WGSL kernel for pairwise distance computation (f32 variant).
    pub const WGSL_CDIST_F32: &str = include_str!("../shaders/misc/cdist.wgsl");

    /// f64 version for universal math library portability.
    #[must_use]
    pub fn wgsl_shader_f64() -> &'static str {
        include_str!("../shaders/misc/cdist_f64.wgsl")
    }

    /// Execute pairwise distance computation on GPU.
    /// # Errors
    /// Returns [`Err`] if inputs are not 2D [M, D] and [N, D], dimension D differs between inputs,
    /// buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input_a.device();
        let shape_a = self.input_a.shape();
        let shape_b = self.input_b.shape();

        // Expect 2D tensors [M, D] and [N, D]
        if shape_a.len() != 2 || shape_b.len() != 2 {
            return Err(crate::error::BarracudaError::InvalidShape {
                expected: vec![0, 0],
                actual: shape_a.to_vec(),
            });
        }

        let m = shape_a[0]; // Number of vectors in A
        let d_a = shape_a[1]; // Dimension of A
        let n = shape_b[0]; // Number of vectors in B
        let d_b = shape_b[1]; // Dimension of B

        if d_a != d_b {
            return Err(crate::error::BarracudaError::InvalidShape {
                expected: vec![n, d_a],
                actual: vec![n, d_b],
            });
        }

        let d = d_a;
        let output_buffer = device.create_buffer_f32(m * n)?;

        // Create params buffer
        let params_data = [m as u32, n as u32, d as u32, self.metric as u32];
        let params_buffer = device.create_uniform_buffer("Params", &params_data);

        let caps = DeviceCapabilities::from_device(device);
        let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::MatMul);
        let workgroups_x = (m as u32).div_ceil(optimal_wg_size);
        let workgroups_y = (n as u32).div_ceil(optimal_wg_size);

        ComputeDispatch::new(device, "cdist")
            .shader(Self::WGSL_CDIST_F32, "main")
            .storage_read(0, self.input_a.buffer())
            .storage_read(1, self.input_b.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch(workgroups_x.max(1), workgroups_y.max(1), 1)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![m, n],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Compute pairwise distances to another tensor; returns [M, N] distance matrix.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn cdist_wgsl(self, other: Self, metric: DistanceMetric) -> Result<Self> {
        Cdist::new(self, other, metric).execute()
    }
}

/// Standalone f64 pairwise distance computation (no Tensor needed).
///
/// * `x1` — `[n1 * d]` f64 flattened row-major
/// * `x2` — `[n2 * d]` f64 flattened row-major
///
/// Returns `[n1 * n2]` f64 distance matrix.
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, buffer readback fails,
/// or the device is lost.
pub fn compute_distances_f64_gpu(
    device: &crate::device::WgpuDevice,
    x1: &[f64],
    n1: usize,
    x2: &[f64],
    n2: usize,
    n_dim: usize,
    metric: DistanceMetric,
) -> Result<Vec<f64>> {
    use bytemuck::{Pod, Zeroable};
    use wgpu::util::DeviceExt;

    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable)]
    struct CdistParams {
        m: u32,
        n: u32,
        d: u32,
        metric: u32,
    }

    let d = device.device();
    let a_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cdist:a"),
        contents: bytemuck::cast_slice(x1),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let b_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cdist:b"),
        contents: bytemuck::cast_slice(x2),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_size = (n1 * n2 * 8) as u64;
    let out_buf = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cdist:out"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = CdistParams {
        m: n1 as u32,
        n: n2 as u32,
        d: n_dim as u32,
        metric: metric as u32,
    };
    let params_buf = device.create_uniform_buffer("cdist:params", &params);

    ComputeDispatch::new(device, "cdist_f64")
        .f64()
        .shader(Cdist::wgsl_shader_f64(), "main")
        .storage_read(0, &a_buf)
        .storage_read(1, &b_buf)
        .storage_rw(2, &out_buf)
        .uniform(3, &params_buf)
        .dispatch((n1 as u32).div_ceil(16), (n2 as u32).div_ceil(16), 1)
        .submit()?;

    let readback = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cdist_f64:rb"),
        size: out_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
        label: Some("cdist_f64:copy"),
    });
    enc.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, out_size);
    device.submit_commands(Some(enc.finish()));

    let result: Vec<f64> = device.map_staging_buffer(&readback, n1 * n2)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::device::test_pool::test_prelude::test_f64_device;

    #[tokio::test]
    async fn test_cdist_euclidean() {
        let device = crate::device::test_pool::get_test_device().await;
        let a_data = vec![0.0, 0.0];
        let b_data = vec![3.0, 4.0];

        let a = Tensor::from_vec_on(a_data, vec![1, 2], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(b_data, vec![1, 2], device)
            .await
            .unwrap();

        let result = a.cdist_wgsl(b, DistanceMetric::Euclidean).unwrap();
        let output = result.to_vec().unwrap();

        assert!((output[0] - 5.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_cdist_manhattan() {
        let device = crate::device::test_pool::get_test_device().await;
        let a_data = vec![0.0, 0.0];
        let b_data = vec![3.0, 4.0];

        let a = Tensor::from_vec_on(a_data, vec![1, 2], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(b_data, vec![1, 2], device)
            .await
            .unwrap();

        let result = a.cdist_wgsl(b, DistanceMetric::Manhattan).unwrap();
        let output = result.to_vec().unwrap();

        assert!((output[0] - 7.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_cdist_f64_euclidean() {
        let Some(device) = test_f64_device().await else {
            return;
        };
        // 2 points in 3D: (1,2,3) and (4,6,3)
        let x1 = vec![1.0_f64, 2.0, 3.0, 0.0, 0.0, 0.0];
        let x2 = vec![4.0_f64, 6.0, 3.0];

        let result =
            compute_distances_f64_gpu(&device, &x1, 2, &x2, 1, 3, DistanceMetric::Euclidean)
                .unwrap();

        // d((1,2,3),(4,6,3)) = sqrt(9+16+0) = 5.0
        assert!(
            (result[0] - 5.0).abs() < 1e-10,
            "expected 5.0, got {}",
            result[0]
        );
        // d((0,0,0),(4,6,3)) = sqrt(16+36+9) = sqrt(61) ≈ 7.8102
        let expected = 61.0_f64.sqrt();
        assert!(
            (result[1] - expected).abs() < 1e-10,
            "expected {expected}, got {}",
            result[1]
        );
    }
}
