// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bray-Curtis Distance (f64) — GPU-Accelerated All-Pairs Distance Matrix
//!
//! ABSORBED from wetSpring (Feb 16, 2026) — general-purpose distance metric
//!
//! Computes the condensed Bray-Curtis distance matrix for N samples, each with
//! D features. Bray-Curtis dissimilarity is commonly used for non-negative
//! abundance/count data in ecology and metagenomics.
//!
//! # Formula
//!
//! ```text
//! BC(a, b) = Σ|aₖ - bₖ| / Σ(aₖ + bₖ)
//! ```
//!
//! # Output Format
//!
//! Condensed distance matrix: N*(N-1)/2 values in order:
//! (1,0), (2,0), (2,1), (3,0), (3,1), (3,2), ...
//!
//! # Performance
//!
//! - O(N² × D) work, embarrassingly parallel across N*(N-1)/2 pairs
//! - For N=1000, D=2000: 500K independent tasks × 2000 features each
//! - GPU threshold: N ≥ 32 (GPU overhead dominates for smaller inputs)
//!
//! # Example
//!
//! ```rust,ignore
//! use barracuda::ops::bray_curtis_f64::BrayCurtisF64;
//!
//! let bc = BrayCurtisF64::new(device.clone())?;
//!
//! // 4 samples, 3 features each
//! let samples = vec![
//!     1.0, 2.0, 3.0,  // sample 0
//!     4.0, 5.0, 6.0,  // sample 1
//!     1.0, 1.0, 1.0,  // sample 2
//!     7.0, 8.0, 9.0,  // sample 3
//! ];
//!
//! let distances = bc.condensed_distance_matrix(&samples, 4, 3)?;
//! // distances.len() = 4*(4-1)/2 = 6
//! // distances[0] = BC(sample1, sample0)
//! // distances[1] = BC(sample2, sample0)
//! // ...
//! ```

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::error::{BarracudaError, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// Parameters for Bray-Curtis shader
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BcParams {
    n_samples: u32,
    n_features: u32,
    n_pairs: u32,
    _pad: u32,
}

/// GPU-accelerated Bray-Curtis distance computation
pub struct BrayCurtisF64 {
    device: Arc<WgpuDevice>,
    pipeline: wgpu::ComputePipeline,
}

#[expect(
    dead_code,
    reason = "CPU reference path for GPU parity validation in tests"
)]
impl BrayCurtisF64 {
    /// Create a new Bray-Curtis GPU operator
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        let shader_source = include_str!("../shaders/math/bray_curtis_f64.wgsl");
        let shader_module = device.compile_shader_f64(shader_source, Some("BrayCurtisF64 Shader"));

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("BrayCurtisF64 Pipeline"),
                layout: None,
                module: &shader_module,
                entry_point: Some("bray_curtis_pairs"),
                cache: None,
                compilation_options: Default::default(),
            });

        Ok(Self { device, pipeline })
    }

    /// Compute condensed Bray-Curtis distance matrix
    ///
    /// # Arguments
    /// * `samples` - Flattened sample matrix [`n_samples` × `n_features`], row-major
    /// * `n_samples` - Number of samples (N)
    /// * `n_features` - Number of features per sample (D)
    ///
    /// # Returns
    /// Condensed distance matrix [N*(N-1)/2]
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn condensed_distance_matrix(
        &self,
        samples: &[f64],
        n_samples: usize,
        n_features: usize,
    ) -> Result<Vec<f64>> {
        let expected_len = n_samples * n_features;
        if samples.len() < expected_len {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Expected {} elements ({} samples × {} features), got {}",
                    expected_len,
                    n_samples,
                    n_features,
                    samples.len()
                ),
            });
        }

        if n_samples < 2 {
            return Ok(Vec::new());
        }

        let n_pairs = n_samples * (n_samples - 1) / 2;

        // Create input buffer
        let input_buffer =
            self.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("BC Input"),
                    contents: bytemuck::cast_slice(samples),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        // Create output buffer
        let output_buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BC Output"),
            size: (n_pairs * 8) as u64, // f64 = 8 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create params buffer
        let params = BcParams {
            n_samples: n_samples as u32,
            n_features: n_features as u32,
            n_pairs: n_pairs as u32,
            _pad: 0,
        };
        let params_buffer =
            self.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("BC Params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        // Create bind group
        let bind_group_layout = self.pipeline.get_bind_group_layout(0);
        let bind_group = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("BC Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        // Execute
        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("BC Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BC Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            // Dispatch: ceil(n_pairs / 256) workgroups
            let n_workgroups = n_pairs.div_ceil(WORKGROUP_SIZE_1D as usize);
            pass.dispatch_workgroups(n_workgroups as u32, 1, 1);
        }

        self.device.submit_commands(Some(encoder.finish()));

        // Read results
        self.read_results(&output_buffer, n_pairs)
    }

    /// Read results from GPU buffer
    fn read_results(&self, buffer: &wgpu::Buffer, count: usize) -> Result<Vec<f64>> {
        let staging = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BC Staging"),
            size: (count * 8) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("BC Copy Encoder"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, (count * 8) as u64);
        self.device.submit_commands(Some(encoder.finish()));

        let results: Vec<f64> = self.device.map_staging_buffer(&staging, count)?;
        Ok(results)
    }

    /// CPU fallback for small inputs
    fn condensed_distance_matrix_cpu(
        &self,
        samples: &[f64],
        n_samples: usize,
        n_features: usize,
    ) -> Result<Vec<f64>> {
        let n_pairs = n_samples * (n_samples - 1) / 2;
        let mut distances = Vec::with_capacity(n_pairs);

        for i in 1..n_samples {
            for j in 0..i {
                let bc = bray_curtis_cpu(samples, i, j, n_features);
                distances.push(bc);
            }
        }

        Ok(distances)
    }

    /// Convert condensed matrix index to (i, j) pair
    ///
    /// Useful for interpreting results.
    #[must_use]
    pub fn condensed_index_to_pair(idx: usize) -> (usize, usize) {
        // i = floor((1 + sqrt(1 + 8*idx)) / 2)
        let k = idx as f64;
        let i = f64::midpoint(1.0, 8.0f64.mul_add(k, 1.0).sqrt()).floor() as usize;
        let j = idx - i * (i - 1) / 2;
        (i, j)
    }

    /// Convert (i, j) pair to condensed matrix index
    ///
    /// Requires i > j.
    #[must_use]
    pub fn pair_to_condensed_index(i: usize, j: usize) -> usize {
        debug_assert!(i > j, "Requires i > j for condensed indexing");
        i * (i - 1) / 2 + j
    }
}

/// CPU reference: Bray-Curtis distance between samples i and j
fn bray_curtis_cpu(samples: &[f64], i: usize, j: usize, n_features: usize) -> f64 {
    let base_i = i * n_features;
    let base_j = j * n_features;

    let mut sum_diff = 0.0;
    let mut sum_sum = 0.0;

    for k in 0..n_features {
        let a = samples[base_i + k];
        let b = samples[base_j + k];

        sum_diff += (a - b).abs();
        sum_sum += a + b;
    }

    if sum_sum > 0.0 {
        sum_diff / sum_sum
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bray_curtis_cpu_identical() {
        // Identical samples → distance = 0
        let samples = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let bc = bray_curtis_cpu(&samples, 1, 0, 3);
        assert!((bc - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_bray_curtis_cpu_disjoint() {
        // Completely disjoint: [1,0,0] vs [0,1,0] → BC = 1.0
        let samples = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let bc = bray_curtis_cpu(&samples, 1, 0, 3);
        assert!((bc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bray_curtis_cpu_known_value() {
        // [1, 2, 3] vs [4, 5, 6]
        // sum_diff = |1-4| + |2-5| + |3-6| = 9
        // sum_sum = (1+4) + (2+5) + (3+6) = 21
        // BC = 9/21 = 3/7 ≈ 0.4286
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bc = bray_curtis_cpu(&samples, 1, 0, 3);
        let expected = 9.0 / 21.0;
        assert!(
            (bc - expected).abs() < 1e-10,
            "Got {bc}, expected {expected}"
        );
    }

    #[test]
    fn test_condensed_index_conversion() {
        // Test round-trip conversion
        for i in 1..10 {
            for j in 0..i {
                let idx = BrayCurtisF64::pair_to_condensed_index(i, j);
                let (i2, j2) = BrayCurtisF64::condensed_index_to_pair(idx);
                assert_eq!((i, j), (i2, j2), "Failed for ({i}, {j})");
            }
        }
    }

    #[test]
    fn test_condensed_matrix_order() {
        // Verify condensed order: (1,0), (2,0), (2,1), (3,0), ...
        assert_eq!(BrayCurtisF64::pair_to_condensed_index(1, 0), 0);
        assert_eq!(BrayCurtisF64::pair_to_condensed_index(2, 0), 1);
        assert_eq!(BrayCurtisF64::pair_to_condensed_index(2, 1), 2);
        assert_eq!(BrayCurtisF64::pair_to_condensed_index(3, 0), 3);
        assert_eq!(BrayCurtisF64::pair_to_condensed_index(3, 1), 4);
        assert_eq!(BrayCurtisF64::pair_to_condensed_index(3, 2), 5);
    }
}
