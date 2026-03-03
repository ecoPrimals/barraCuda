// SPDX-License-Identifier: AGPL-3.0-only

//! Per-read parallel quality trimming on GPU.
//!
//! One thread per FASTQ read. Applies leading trim, trailing trim,
//! sliding window quality check, and minimum length filter.
//! Packs result as `(start << 16) | end`, or 0 for failed reads.
//!
//! ## Absorbed from
//!
//! wetSpring handoff v6, `quality_filter.wgsl` — 88 pipeline checks PASS.

use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("../../shaders/bio/quality_filter.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct QualityFilterParams {
    n_reads: u32,
    leading_min_quality: u32,
    trailing_min_quality: u32,
    window_min_quality: u32,
    window_size: u32,
    min_length: u32,
    phred_offset: u32,
    _pad: u32,
}

/// GPU-parallel quality trimming for FASTQ reads.
pub struct QualityFilterGpu {
    device: Arc<WgpuDevice>,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

/// Configuration for quality filtering.
#[derive(Debug, Clone)]
pub struct QualityConfig {
    pub leading_min_quality: u32,
    pub trailing_min_quality: u32,
    pub window_min_quality: u32,
    pub window_size: u32,
    pub min_length: u32,
    pub phred_offset: u32,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            leading_min_quality: 3,
            trailing_min_quality: 3,
            window_min_quality: 15,
            window_size: 4,
            min_length: 36,
            phred_offset: 33,
        }
    }
}

impl QualityFilterGpu {
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        let module = device.compile_shader_f64(SHADER, Some("quality_filter"));
        let bgl = super::snp::make_bgl(&device, &[true, true, true, false]);
        let layout = super::snp::make_layout(&device, &bgl, "QualityFilter");
        let pipeline =
            super::snp::make_pipeline(&device, &layout, &module, "quality_filter", "QualityFilter");
        Ok(Self {
            device,
            pipeline,
            bgl,
        })
    }

    pub fn dispatch(
        &self,
        n_reads: u32,
        config: &QualityConfig,
        qual_data: &wgpu::Buffer,
        read_offsets: &wgpu::Buffer,
        read_lengths: &wgpu::Buffer,
        results: &wgpu::Buffer,
    ) -> Result<()> {
        let params = QualityFilterParams {
            n_reads,
            leading_min_quality: config.leading_min_quality,
            trailing_min_quality: config.trailing_min_quality,
            window_min_quality: config.window_min_quality,
            window_size: config.window_size,
            min_length: config.min_length,
            phred_offset: config.phred_offset,
            _pad: 0,
        };
        let pbuf = super::snp::upload_uniform(&self.device, &params);
        let bg = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bgl,
                entries: &[
                    super::snp::bg_entry(0, &pbuf),
                    super::snp::bg_entry(1, qual_data),
                    super::snp::bg_entry(2, read_offsets),
                    super::snp::bg_entry(3, read_lengths),
                    super::snp::bg_entry(4, results),
                ],
            });
        super::snp::submit(&self.device, &self.pipeline, &bg, n_reads.div_ceil(256));
        Ok(())
    }
}
