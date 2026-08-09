// SPDX-License-Identifier: AGPL-3.0-or-later

//! K-mer Histogram — GPU kernel.
//!
//! Computes a 4^k histogram from encoded k-mer sequences using atomic
//! increments. One thread per k-mer.
//!
//! Provenance: wetSpring metagenomics → toadStool absorption

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;

/// WGSL shader for k-mer histogram computation (atomic increments).
pub const WGSL_KMER_HISTOGRAM: &str = include_str!("../../shaders/bio/kmer_histogram.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct KmerConfig {
    n_kmers: u32,
    k: u32,
    _pad0: u32,
    _pad1: u32,
}

/// GPU kernel for k-mer histogram: counts occurrences into 4^k bins.
pub struct KmerHistogramGpu {
    device: Arc<WgpuDevice>,
}

impl KmerHistogramGpu {
    /// Create a k-mer histogram GPU kernel.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Count k-mer occurrences into a histogram.
    ///
    /// `kmers_buf`: `[n_kmers]` u32 — encoded k-mer hashes (each < 4^k)
    /// `histogram_buf`: `[4^k]` u32 — output histogram (must be zeroed before dispatch)
    /// # Errors
    ///
    /// Returns [`Err`] if shader compilation or GPU dispatch fails.
    pub fn dispatch(
        &self,
        kmers_buf: &wgpu::Buffer,
        histogram_buf: &wgpu::Buffer,
        n_kmers: u32,
        k: u32,
    ) -> Result<()> {
        let d = self.device.device();

        let config = KmerConfig {
            n_kmers,
            k,
            _pad0: 0,
            _pad1: 0,
        };
        let config_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("KmerHistogram Config"),
            contents: bytemuck::bytes_of(&config),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        ComputeDispatch::new(&self.device, "KmerHistogram")
            .shader(WGSL_KMER_HISTOGRAM, "kmer_histogram")
            .uniform(0, &config_buf)
            .storage_read(1, kmers_buf)
            .storage_rw(2, histogram_buf)
            .dispatch_1d(n_kmers)
            .submit()?;
        Ok(())
    }
}

/// Convert a `u32` histogram (e.g. from GPU readback) to `f64`.
///
/// Spectrum and visualization channels require `Vec<f64>`; GPU k-mer
/// histograms produce `Vec<u32>`. This avoids repeated manual casting
/// in downstream consumers.
#[must_use]
pub fn histogram_u32_to_f64(counts: &[u32]) -> Vec<f64> {
    counts.iter().map(|&c| f64::from(c)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_contains_entry_point() {
        assert!(WGSL_KMER_HISTOGRAM.contains("fn kmer_histogram"));
    }

    #[test]
    fn histogram_conversion_roundtrip() {
        let counts: Vec<u32> = vec![0, 5, 12, 0, 3];
        let f64s = histogram_u32_to_f64(&counts);
        assert_eq!(f64s, vec![0.0, 5.0, 12.0, 0.0, 3.0]);
    }

    #[test]
    fn histogram_conversion_empty() {
        assert!(histogram_u32_to_f64(&[]).is_empty());
    }
}
