// SPDX-License-Identifier: AGPL-3.0-or-later

//! TMU lookup tables for Box-Muller PRNG and stencil operations.
//!
//! Pre-computes textures used by TMU-routed shaders (Tier 0 silicon routing).
//! Textures are allocated once per GPU and reused across all trajectories.
//!
//! Absorbed from hotSpring V0632 (March 2026).
//!
//! - `log_table`: `R32Float` 4096x1 — `entry[i] = -2 * ln(i / 4095)` for Box-Muller
//! - `trig_table`: `Rg32Float` 4096x1 — `(cos(2*pi*i/4095), sin(2*pi*i/4095))`

use crate::device::WgpuDevice;
use std::sync::Arc;

const TABLE_SIZE: u32 = 4096;

/// Persistent TMU lookup textures for Box-Muller PRNG.
pub struct TmuLookupTables {
    /// `-2 * ln(x/4095)` for x in 0..4096. `R32Float`, 4096x1.
    pub log_table: wgpu::TextureView,
    /// `(cos(2*pi*x/4095), sin(2*pi*x/4095))` for x in 0..4096. `Rg32Float`, 4096x1.
    pub trig_table: wgpu::TextureView,
}

impl TmuLookupTables {
    /// Create and upload the TMU lookup tables. Call once per GPU at init time.
    #[must_use]
    pub fn new(device: &Arc<WgpuDevice>) -> Self {
        let wgpu_device = device.device();
        let queue = device.queue();

        let log_data = Self::build_log_table();
        let trig_data = Self::build_trig_table();

        let log_tex = wgpu_device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tmu_log_table"),
            size: wgpu::Extent3d {
                width: TABLE_SIZE,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &log_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&log_data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(TABLE_SIZE * 4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: TABLE_SIZE,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        let trig_tex = wgpu_device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tmu_trig_table"),
            size: wgpu::Extent3d {
                width: TABLE_SIZE,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &trig_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&trig_data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(TABLE_SIZE * 8),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: TABLE_SIZE,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        Self {
            log_table: log_tex.create_view(&wgpu::TextureViewDescriptor::default()),
            trig_table: trig_tex.create_view(&wgpu::TextureViewDescriptor::default()),
        }
    }

    fn build_log_table() -> Vec<f32> {
        (0..TABLE_SIZE)
            .map(|i| {
                let x = (f64::from(i) / f64::from(TABLE_SIZE - 1)).max(1e-20);
                (-2.0 * x.ln()) as f32
            })
            .collect()
    }

    fn build_trig_table() -> Vec<[f32; 2]> {
        (0..TABLE_SIZE)
            .map(|i| {
                let theta = 2.0 * std::f64::consts::PI * f64::from(i) / f64::from(TABLE_SIZE - 1);
                [theta.cos() as f32, theta.sin() as f32]
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_table_values() {
        let table = TmuLookupTables::build_log_table();
        assert_eq!(table.len(), TABLE_SIZE as usize);
        assert!(table[0] > 0.0, "log(~0) should be large positive");
        assert!(
            (table[(TABLE_SIZE - 1) as usize]).abs() < 0.01,
            "log(1) should be ~0"
        );
    }

    #[test]
    fn trig_table_values() {
        let table = TmuLookupTables::build_trig_table();
        assert_eq!(table.len(), TABLE_SIZE as usize);
        let [cos0, sin0] = table[0];
        assert!((cos0 - 1.0).abs() < 1e-5, "cos(0) should be ~1");
        assert!(sin0.abs() < 1e-5, "sin(0) should be ~0");
    }

    #[test]
    fn trig_table_midpoint() {
        let table = TmuLookupTables::build_trig_table();
        let mid = (TABLE_SIZE / 2) as usize;
        let [cos_pi, sin_pi] = table[mid];
        assert!((cos_pi + 1.0).abs() < 0.01, "cos(pi) should be ~-1");
        assert!(sin_pi.abs() < 0.01, "sin(pi) should be ~0");
    }
}
