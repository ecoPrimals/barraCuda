// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU compute operations for NMS
//!
//! This module contains the 4-pass GPU execution:
//! 1. Pass 1: Compute IoU Matrix on GPU (parallel)
//! 2. Pass 2: Sort indices by score (CPU - acceptable for small sets)
//! 3. Pass 3: Mark suppressed boxes on GPU (parallel)
//! 4. Pass 4: Compact results on GPU (parallel with atomics)

use super::NMS;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::DeviceCapabilities;
use crate::device::WgpuDevice;
use crate::error::Result;
use crate::tensor::Tensor;
use std::sync::Arc;

impl NMS {
    /// Execute NMS on a given GPU device.
    pub fn execute_on(self, device: Arc<WgpuDevice>) -> Result<Vec<usize>> {
        if self.boxes().is_empty() {
            return Ok(Vec::new());
        }

        let num_boxes = self.boxes().len();

        if num_boxes == 1 {
            return Ok(vec![0]);
        }

        self.execute_inner(device, num_boxes)
    }

    /// Execute NMS with automatic device discovery.
    pub fn execute(self) -> Result<Vec<usize>> {
        if self.boxes().is_empty() {
            return Ok(Vec::new());
        }

        let num_boxes = self.boxes().len();

        if num_boxes == 1 {
            return Ok(vec![0]);
        }

        let device = crate::device::test_pool::get_test_device_if_gpu_available_sync()
            .ok_or_else(|| crate::error::BarracudaError::device("No GPU available for NMS"))?;

        self.execute_inner(device, num_boxes)
    }

    fn execute_inner(self, device: Arc<WgpuDevice>, num_boxes: usize) -> Result<Vec<usize>> {
        // Convert boxes to tensor format [num_boxes, 5] where each box is [x1, y1, x2, y2, score]
        let box_data: Vec<f32> = self
            .boxes()
            .iter()
            .flat_map(|b| [b.x1, b.y1, b.x2, b.y2, b.score])
            .collect();

        // Create box tensor on GPU
        let boxes_tensor = Tensor::from_vec_on_sync(box_data, vec![num_boxes, 5], device.clone())?;

        // ====================================================================
        // Pass 1: Compute IoU Matrix on GPU
        // ====================================================================
        let iou_matrix_size = num_boxes * num_boxes;
        let iou_matrix_buffer = device.create_buffer_f32(iou_matrix_size)?;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct IoUParams {
            num_boxes: u32,
            _padding: [u32; 3],
        }

        let iou_params = IoUParams {
            num_boxes: num_boxes as u32,
            _padding: [0; 3],
        };
        let iou_params_buffer = device.create_uniform_buffer("NMS IoU Params", &iou_params);

        let caps = DeviceCapabilities::from_device(&device);
        let (workgroups_x, workgroups_y) = caps.dispatch_2d(num_boxes as u32, num_boxes as u32);

        ComputeDispatch::new(&device, "nms_iou")
            .shader(Self::wgsl_shader(), "compute_iou_matrix")
            .storage_read(0, boxes_tensor.buffer())
            .storage_rw(1, &iou_matrix_buffer)
            .uniform(2, &iou_params_buffer)
            .dispatch(workgroups_x.max(1), workgroups_y.max(1), 1)
            .submit();

        // ====================================================================
        // Pass 2: Sort indices by score (CPU - acceptable for small sets)
        // ====================================================================
        let mut sorted_indices: Vec<u32> = (0..num_boxes as u32).collect();
        sorted_indices.sort_by(|&a, &b| {
            self.boxes()[b as usize]
                .score
                .partial_cmp(&self.boxes()[a as usize].score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Create sorted indices buffer
        let sorted_indices_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("NMS Sorted Indices"),
                    contents: bytemuck::cast_slice(&sorted_indices),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

        // ====================================================================
        // Pass 3: Mark Suppressed Boxes on GPU
        // ====================================================================
        let suppressed_buffer = device.create_buffer_u32_zeros(num_boxes)?;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct SuppressParams {
            num_boxes: u32,
            iou_threshold: f32,
            _padding: [u32; 2],
        }

        let suppress_params = SuppressParams {
            num_boxes: num_boxes as u32,
            iou_threshold: self.iou_threshold(),
            _padding: [0; 2],
        };
        let suppress_params_buffer =
            device.create_uniform_buffer("NMS Suppress Params", &suppress_params);

        let workgroups_suppress = caps.dispatch_1d(num_boxes as u32);

        ComputeDispatch::new(&device, "nms_suppress")
            .shader(Self::wgsl_shader(), "mark_suppressed")
            .storage_read(0, &sorted_indices_buffer)
            .storage_read(1, &iou_matrix_buffer)
            .storage_rw(2, &suppressed_buffer)
            .uniform(3, &suppress_params_buffer)
            .dispatch(workgroups_suppress, 1, 1)
            .submit();

        // ====================================================================
        // Pass 4: Compact Results on GPU
        // ====================================================================
        let keep_indices_buffer = device.create_buffer_u32(num_boxes)?;
        let keep_count_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("NMS Keep Count"),
                    contents: bytemuck::cast_slice(&[0u32]),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct CompactParams {
            num_boxes: u32,
            _padding: [u32; 3],
        }

        let compact_params = CompactParams {
            num_boxes: num_boxes as u32,
            _padding: [0; 3],
        };
        let compact_params_buffer =
            device.create_uniform_buffer("NMS Compact Params", &compact_params);

        let workgroups_compact = caps.dispatch_1d(num_boxes as u32);

        ComputeDispatch::new(&device, "nms_compact")
            .shader(Self::wgsl_shader(), "compact_results")
            .storage_read(0, &sorted_indices_buffer)
            .storage_read(1, &suppressed_buffer)
            .storage_rw(2, &keep_indices_buffer)
            .storage_rw(3, &keep_count_buffer)
            .uniform(4, &compact_params_buffer)
            .dispatch(workgroups_compact, 1, 1)
            .submit();

        // ====================================================================
        // Read Results (read_buffer_u32 will handle GPU synchronization)
        // ====================================================================
        let keep_count_data = device.read_buffer_u32(&keep_count_buffer, 1)?;
        let keep_count = keep_count_data[0] as usize;

        if keep_count == 0 {
            return Ok(Vec::new());
        }

        let keep_indices_data = device.read_buffer_u32(&keep_indices_buffer, keep_count)?;
        let keep: Vec<usize> = keep_indices_data.iter().map(|&idx| idx as usize).collect();

        Ok(keep)
    }
}
