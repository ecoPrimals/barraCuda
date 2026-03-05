// SPDX-License-Identifier: AGPL-3.0-or-later
//! Intersection over Union for bounding boxes
//!
//! **Pure WGSL**: Single implementation via WebGPU shader
//! Computes IoU between pairs of bounding boxes

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BoxIoUParams {
    num_boxes_a: u32,
    num_boxes_b: u32,
    box_format: u32, // 0 = xyxy, 1 = xywh, 2 = cxcywh
    _padding: u32,
}

/// Intersection over Union between pairs of bounding boxes.
pub struct BoxIoU {
    boxes_a: Tensor,
    boxes_b: Tensor,
    box_format: u32,
}

impl BoxIoU {
    /// Create BoxIoU operation
    pub fn new(boxes_a: Tensor, boxes_b: Tensor, box_format: u32) -> Result<Self> {
        if box_format > 2 {
            return Err(BarracudaError::invalid_op(
                "BoxIoU",
                format!("box_format must be 0 (xyxy), 1 (xywh), or 2 (cxcywh), got {box_format}"),
            ));
        }

        Ok(Self {
            boxes_a,
            boxes_b,
            box_format,
        })
    }

    /// WGSL shader source (embedded at compile time)
    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../shaders/detection/box_iou_f64.wgsl"
                ))
            });
            SHADER.as_str()
        }
    }

    /// Execute BoxIoU on tensor
    pub fn execute(self) -> Result<Tensor> {
        let device = self.boxes_a.device();
        let a_shape = self.boxes_a.shape();
        let b_shape = self.boxes_b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(BarracudaError::invalid_op(
                "BoxIoU",
                format!("boxes must be 2D [num_boxes, 4], got shapes {a_shape:?} and {b_shape:?}"),
            ));
        }

        if a_shape[1] != 4 || b_shape[1] != 4 {
            return Err(BarracudaError::invalid_op(
                "BoxIoU",
                format!(
                    "boxes must have 4 coordinates, got {} and {}",
                    a_shape[1], b_shape[1]
                ),
            ));
        }

        let num_boxes_a = a_shape[0];
        let num_boxes_b = b_shape[0];

        // Create output buffer: [num_boxes_a, num_boxes_b]
        let output_size = num_boxes_a * num_boxes_b;
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = BoxIoUParams {
            num_boxes_a: num_boxes_a as u32,
            num_boxes_b: num_boxes_b as u32,
            box_format: self.box_format,
            _padding: 0,
        };

        let params_buffer = device.create_uniform_buffer("BoxIoU Params", &params);

        let workgroups_x = (num_boxes_a as u32).div_ceil(16);
        let workgroups_y = (num_boxes_b as u32).div_ceil(16);

        ComputeDispatch::new(device, "BoxIoU")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.boxes_a.buffer())
            .storage_read(1, self.boxes_b.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch(workgroups_x, workgroups_y, 1)
            .submit();

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![num_boxes_a, num_boxes_b],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_box_iou_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let num_boxes_a = 3;
        let num_boxes_b = 4;

        let boxes_a = Tensor::from_vec_on(
            vec![
                0.0, 0.0, 10.0, 10.0, 5.0, 5.0, 15.0, 15.0, 10.0, 10.0, 20.0, 20.0,
            ],
            vec![num_boxes_a, 4],
            device.clone(),
        )
        .await
        .unwrap();

        let boxes_b = Tensor::from_vec_on(
            vec![
                1.0, 1.0, 11.0, 11.0, 6.0, 6.0, 16.0, 16.0, 11.0, 11.0, 21.0, 21.0, 2.0, 2.0, 12.0,
                12.0,
            ],
            vec![num_boxes_b, 4],
            device.clone(),
        )
        .await
        .unwrap();

        let result = BoxIoU::new(boxes_a, boxes_b, 0).unwrap().execute().unwrap();

        assert_eq!(result.shape(), &[num_boxes_a, num_boxes_b]);
    }
}
