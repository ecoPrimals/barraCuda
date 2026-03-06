// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU `PeakDetectF64` — 1D peak detection with prominence and width.
//!
//! Parallel local-maxima detection via WGSL shader, with CPU-side filtering
//! for height, prominence, distance, and width thresholds.

use std::sync::Arc;

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::error::Result;

/// A detected peak with its properties.
#[derive(Debug, Clone)]
pub struct DetectedPeak {
    /// Index in the signal
    pub index: usize,
    /// Peak height (signal value at peak)
    pub height: f64,
    /// Prominence (vertical distance to higher peak)
    pub prominence: f64,
    /// Width at half-prominence
    pub width: f64,
}

/// GPU-accelerated 1D peak detection with configurable thresholds.
///
/// The GPU shader detects all local maxima within a `distance` window and
/// computes their prominence. The CPU orchestrator then filters by the
/// optional `height`, `prominence`, and `width` thresholds.
pub struct PeakDetectF64<'a> {
    signal: &'a [f64],
    distance: usize,
    height: Option<f64>,
    prominence: Option<f64>,
    width: Option<f64>,
}

impl<'a> PeakDetectF64<'a> {
    /// Create a peak detector with minimum peak distance.
    #[must_use]
    pub fn new(signal: &'a [f64], distance: usize) -> Self {
        Self {
            signal,
            distance: distance.max(1),
            height: None,
            prominence: None,
            width: None,
        }
    }

    /// Set minimum peak height threshold.
    #[must_use]
    pub fn height(mut self, min_height: f64) -> Self {
        self.height = Some(min_height);
        self
    }

    /// Set minimum prominence threshold.
    #[must_use]
    pub fn prominence(mut self, min_prominence: f64) -> Self {
        self.prominence = Some(min_prominence);
        self
    }

    /// Set minimum width threshold.
    #[must_use]
    pub fn width(mut self, min_width: f64) -> Self {
        self.width = Some(min_width);
        self
    }

    /// Run peak detection on the GPU.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(&self, device: &Arc<WgpuDevice>) -> Result<Vec<DetectedPeak>> {
        let n = self.signal.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let (is_peak_data, prominence_data) = self.dispatch_gpu(device, n)?;

        let peaks: Vec<_> = (0..n)
            .filter(|&i| is_peak_data[i] == 1)
            .filter_map(|i| {
                let h = self.signal[i];
                let p = prominence_data[i];
                if !self.height.is_none_or(|min_h| h >= min_h) {
                    return None;
                }
                if !self.prominence.is_none_or(|min_p| p >= min_p) {
                    return None;
                }
                let w = Self::compute_width(self.signal, i, p);
                if !self.width.is_none_or(|min_w| w >= min_w) {
                    return None;
                }
                Some(DetectedPeak {
                    index: i,
                    height: h,
                    prominence: p,
                    width: w,
                })
            })
            .collect();

        Ok(peaks)
    }

    /// Compute peak width at half-prominence level.
    fn compute_width(signal: &[f64], peak_idx: usize, prominence: f64) -> f64 {
        let threshold = signal[peak_idx] - prominence / 2.0;

        let mut left = peak_idx;
        while left > 0 && signal[left - 1] > threshold {
            left -= 1;
        }

        let mut right = peak_idx;
        while right + 1 < signal.len() && signal[right + 1] > threshold {
            right += 1;
        }

        (right - left + 1) as f64
    }

    fn dispatch_gpu(&self, device: &Arc<WgpuDevice>, n: usize) -> Result<(Vec<u32>, Vec<f64>)> {
        let shader_src = include_str!("../shaders/signal/peak_detect_f64.wgsl");

        let signal_buf = Self::f64_buf(device, self.signal);
        let is_peak_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PeakDetect is_peak"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let prominence_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PeakDetect prominence"),
            size: (n * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            n: u32,
            distance: u32,
            _pad0: u32,
            _pad1: u32,
        }

        let params = Params {
            n: n as u32,
            distance: self.distance as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PeakDetect params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("PeakDetect BGL"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    storage_entry(2, false),
                    uniform_entry(3),
                ],
            });

        let bg = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PeakDetect BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: signal_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: is_peak_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: prominence_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let shader = device.compile_shader(shader_src, Some("PeakDetect"));
        let pl = device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("PeakDetect PL"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });
        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("PeakDetect Pipeline"),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let workgroups = (n as u32).div_ceil(WORKGROUP_SIZE_1D);
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("PeakDetect Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PeakDetect Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Readback buffers
        let is_peak_staging = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PeakDetect is_peak staging"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let prominence_staging = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PeakDetect prominence staging"),
            size: (n * 8) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&is_peak_buf, 0, &is_peak_staging, 0, (n * 4) as u64);
        encoder.copy_buffer_to_buffer(&prominence_buf, 0, &prominence_staging, 0, (n * 8) as u64);

        device.submit_and_poll(Some(encoder.finish()));

        let is_peak_data: Vec<u32> = device.map_staging_buffer(&is_peak_staging, n)?;
        let prominence_data: Vec<f64> = device.map_staging_buffer(&prominence_staging, n)?;

        Ok((is_peak_data, prominence_data))
    }

    fn f64_buf(device: &Arc<WgpuDevice>, data: &[f64]) -> wgpu::Buffer {
        device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PeakDetect f64"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }
}

/// CPU reference implementation for testing.
#[must_use]
pub fn find_peaks_cpu(
    signal: &[f64],
    distance: usize,
    min_height: Option<f64>,
    min_prominence: Option<f64>,
) -> Vec<DetectedPeak> {
    let n = signal.len();
    let dist = distance.max(1);
    let mut peaks = Vec::new();

    for i in 0..n {
        let val = signal[i];

        // Local maximum check
        let start = i.saturating_sub(dist);
        let end = (i + dist + 1).min(n);
        let is_max = (start..end).all(|j| j == i || signal[j] < val);
        if !is_max {
            continue;
        }

        if let Some(h) = min_height {
            if val < h {
                continue;
            }
        }

        // Prominence: scan left/right for min valley before higher peak
        let mut left_min = val;
        for j in (0..i).rev() {
            left_min = left_min.min(signal[j]);
            if signal[j] > val {
                break;
            }
        }
        let mut right_min = val;
        for j in (i + 1)..n {
            right_min = right_min.min(signal[j]);
            if signal[j] > val {
                break;
            }
        }
        let prominence = val - left_min.max(right_min);

        if let Some(mp) = min_prominence {
            if prominence < mp {
                continue;
            }
        }

        let width = PeakDetectF64::compute_width(signal, i, prominence);

        peaks.push(DetectedPeak {
            index: i,
            height: val,
            prominence,
            width,
        });
    }

    peaks
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_peak_detect_known_signal() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };

        // Signal: two clear peaks at indices 3 and 8
        let signal = vec![0.0, 1.0, 2.0, 5.0, 2.0, 1.0, 0.5, 3.0, 7.0, 3.0, 0.5, 0.0];

        let peaks = PeakDetectF64::new(&signal, 1)
            .execute(&device)
            .expect("peak detect");

        let indices: Vec<usize> = peaks.iter().map(|p| p.index).collect();
        assert!(indices.contains(&3), "should detect peak at index 3");
        assert!(indices.contains(&8), "should detect peak at index 8");
    }

    #[tokio::test]
    async fn test_peak_detect_with_height_filter() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };

        let signal = vec![0.0, 1.0, 2.0, 5.0, 2.0, 1.0, 0.5, 3.0, 7.0, 3.0, 0.5, 0.0];

        let peaks = PeakDetectF64::new(&signal, 1)
            .height(6.0)
            .execute(&device)
            .expect("peak detect");

        assert_eq!(peaks.len(), 1, "only peak at 7.0 passes height >= 6.0");
        assert_eq!(peaks[0].index, 8);
    }

    #[tokio::test]
    async fn test_peak_detect_prominence() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };

        // Signal with one prominent peak and one minor bump
        let signal = vec![0.0, 0.5, 0.3, 0.0, 3.0, 8.0, 3.0, 0.0];

        let peaks = PeakDetectF64::new(&signal, 1)
            .execute(&device)
            .expect("peak detect");

        let main_peak = peaks.iter().find(|p| p.index == 5).expect("peak at 5");
        assert!(
            main_peak.prominence > 5.0,
            "prominence should be > 5.0, got {}",
            main_peak.prominence
        );
    }

    #[tokio::test]
    async fn test_peak_detect_matches_cpu() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };

        let signal: Vec<f64> = (0..100)
            .map(|i| {
                let x = i as f64 * 0.1;
                (x * 2.0).sin() + 0.5 * (x * 5.0).sin()
            })
            .collect();

        let gpu_peaks = PeakDetectF64::new(&signal, 2)
            .execute(&device)
            .expect("gpu peaks");
        let cpu_peaks = find_peaks_cpu(&signal, 2, None, None);

        let gpu_idx: Vec<usize> = gpu_peaks.iter().map(|p| p.index).collect();
        let cpu_idx: Vec<usize> = cpu_peaks.iter().map(|p| p.index).collect();

        assert_eq!(gpu_idx, cpu_idx, "GPU and CPU should detect same peaks");

        for (g, c) in gpu_peaks.iter().zip(cpu_peaks.iter()) {
            assert!(
                (g.prominence - c.prominence).abs() < 1e-10,
                "prominence mismatch at {}: gpu={}, cpu={}",
                g.index,
                g.prominence,
                c.prominence
            );
        }
    }

    #[tokio::test]
    async fn test_peak_detect_edge_endpoints() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };

        // Peak at start and end
        let signal = vec![5.0, 3.0, 1.0, 2.0, 4.0];

        let peaks = PeakDetectF64::new(&signal, 1)
            .execute(&device)
            .expect("peak detect");

        let indices: Vec<usize> = peaks.iter().map(|p| p.index).collect();
        assert!(
            indices.contains(&0),
            "should detect peak at endpoint index 0"
        );
        assert!(
            indices.contains(&4),
            "should detect peak at endpoint index 4"
        );
    }

    #[tokio::test]
    async fn test_peak_detect_plateau_rejected() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };

        // Plateau: adjacent equal values should not be detected as peaks
        let signal = vec![0.0, 1.0, 5.0, 5.0, 1.0, 0.0];

        let peaks = PeakDetectF64::new(&signal, 1)
            .execute(&device)
            .expect("peak detect");

        let indices: Vec<usize> = peaks.iter().map(|p| p.index).collect();
        assert!(
            !indices.contains(&2) && !indices.contains(&3),
            "plateau values should not be detected as peaks: {indices:?}"
        );
    }
}
