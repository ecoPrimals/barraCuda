// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU batch beat classification via normalized cross-correlation template matching.
//!
//! Each thread classifies one beat window against N templates.
//! Absorbed from healthSpring V19 (Exp082, Exp085).

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("../../shaders/health/beat_classify_batch_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BeatClassifyParams {
    n_beats: u32,
    n_templates: u32,
    window_size: u32,
    _pad: u32,
}

/// Result of GPU beat classification.
#[derive(Debug, Clone)]
pub struct GpuBeatResult {
    /// Index of the best-matching template.
    pub template_index: u32,
    /// Normalized cross-correlation with the best template.
    pub correlation: f64,
}

/// GPU-accelerated batch beat classification.
pub struct BeatClassifyGpu {
    device: Arc<WgpuDevice>,
}

impl BeatClassifyGpu {
    /// Create a new `BeatClassifyGpu` for the given device.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Classify beats against templates using normalized cross-correlation.
    ///
    /// `beats` — flattened beat windows (`n_beats` × `window_size`).
    /// `templates` — flattened template windows (`n_templates` × `window_size`).
    ///
    /// # Errors
    /// Returns [`Err`] on pipeline creation, dispatch, or readback failure.
    pub fn classify(
        &self,
        beats: &[f64],
        templates: &[f64],
        n_beats: u32,
        n_templates: u32,
        window_size: u32,
    ) -> Result<Vec<GpuBeatResult>> {
        let params = BeatClassifyParams {
            n_beats,
            n_templates,
            window_size,
            _pad: 0,
        };

        let beats_buf = self.device.create_buffer_f64_init("beats:input", beats);
        let tmpl_buf = self
            .device
            .create_buffer_f64_init("beats:templates", templates);
        let out_buf = self.device.create_buffer_f64((n_beats as usize) * 2)?;
        let params_buf = self.device.create_uniform_buffer("beats:params", &params);

        let wg_count = n_beats.div_ceil(WORKGROUP_SIZE_1D);

        crate::device::compute_pipeline::ComputeDispatch::new(&self.device, "beat_classify")
            .shader(SHADER, "main")
            .f64()
            .storage_read(0, &beats_buf)
            .storage_read(1, &tmpl_buf)
            .storage_rw(2, &out_buf)
            .uniform(3, &params_buf)
            .dispatch(wg_count, 1, 1)
            .submit()?;

        let raw = self
            .device
            .read_f64_buffer(&out_buf, (n_beats as usize) * 2)?;
        let results = raw
            .chunks_exact(2)
            .map(|chunk| GpuBeatResult {
                template_index: chunk[0] as u32,
                correlation: chunk[1],
            })
            .collect();
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_layout() {
        assert_eq!(std::mem::size_of::<BeatClassifyParams>(), 16);
    }

    #[test]
    fn shader_source_valid() {
        assert!(SHADER.contains("normalized"));
        assert!(SHADER.contains("templates"));
        assert!(SHADER.contains("Params"));
    }
}
