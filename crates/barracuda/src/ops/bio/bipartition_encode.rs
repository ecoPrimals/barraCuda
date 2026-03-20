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
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::compute_pipeline::{storage_bgl_entry, uniform_bgl_entry};
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
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<WgpuDevice>,
}

impl BipartitionEncodeGpu {
    /// Create the bipartition encoding kernel.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        let d = device.device();

        let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BipartitionEncode BGL"),
            entries: &[
                uniform_bgl_entry(0),
                storage_bgl_entry(1, true),
                storage_bgl_entry(2, false),
            ],
        });

        let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("BipartitionEncode Layout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let module = device.compile_shader(WGSL_BIPARTITION_ENCODE, Some("bipartition_encode"));

        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BipartitionEncode Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: device.pipeline_cache(),
        });

        Self {
            pipeline,
            bgl,
            device,
        }
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

        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bipart_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: config_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: membership_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let wg_count = n_bipartitions.div_ceil(WORKGROUP_SIZE_1D);
        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("bipart_encode"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bipart_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(wg_count, 1, 1);
        }
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
                crate::error::BarracudaError::DeviceLost("readback channel closed".into())
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
