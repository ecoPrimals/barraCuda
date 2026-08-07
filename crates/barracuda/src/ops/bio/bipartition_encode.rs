// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bipartition Encoding — GPU kernel for Robinson-Foulds distance.
//!
//! Converts tree bipartition membership arrays into packed u32 bit-vectors,
//! enabling fast bitwise RF distance computation.
//!
//! Provenance: wetSpring V105 → barraCuda absorption (Mar 2026)

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;

/// WGSL shader for bipartition → bit-vector encoding.
pub const WGSL_BIPARTITION_ENCODE: &str = include_str!("../../shaders/bio/bipartition_encode.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BipartConfig {
    n_bipartitions: u32,
    n_taxa: u32,
    words_per_bipart: u32,
    _pad: u32,
}

/// GPU kernel for bipartition → bit-vector encoding.
pub struct BipartitionEncodeGpu {
    device: Arc<WgpuDevice>,
}

impl BipartitionEncodeGpu {
    /// Create the bipartition encoding kernel.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Encode bipartitions into packed bit-vectors.
    ///
    /// `membership` is `[n_bipartitions × n_taxa]` with values 0 or 1.
    /// Returns `[n_bipartitions × words_per_bipart]` packed u32 bit-vectors.
    ///
    /// # Errors
    /// Returns [`Err`] if the device is lost or poll fails.
    pub fn encode(&self, membership: &[u32], n_bipartitions: u32, n_taxa: u32) -> Result<Vec<u32>> {
        let words_per = n_taxa.div_ceil(32);

        let config = BipartConfig {
            n_bipartitions,
            n_taxa,
            words_per_bipart: words_per,
            _pad: 0,
        };

        let d = self.device.device();

        let config_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bipart_config"),
            contents: bytemuck::bytes_of(&config),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let membership_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bipart_membership"),
            contents: bytemuck::cast_slice(membership),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let out_size = (n_bipartitions * words_per) as u64 * 4;
        let output_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bipart_output"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bipart_staging"),
            size: out_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        ComputeDispatch::new(&self.device, "BipartitionEncode")
            .shader(WGSL_BIPARTITION_ENCODE, "main")
            .uniform(0, &config_buf)
            .storage_read(1, &membership_buf)
            .storage_rw(2, &output_buf)
            .dispatch_1d(n_bipartitions)
            .submit()?;

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("bipart_encode_copy"),
            });
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, out_size);
        self.device
            .queue()
            .submit(std::iter::once(encoder.finish()));
        self.device.poll_safe()?;

        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device.poll_safe()?;
        rx.recv()
            .map_err(|_| {
                crate::error::BarracudaError::device_lost("readback channel closed")
            })?
            .map_err(|e| {
                crate::error::BarracudaError::DeviceLost(format!("buffer map failed: {e:?}"))
            })?;

        let data = slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buf.unmap();

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_layout() {
        assert_eq!(std::mem::size_of::<BipartConfig>(), 16);
    }
}
