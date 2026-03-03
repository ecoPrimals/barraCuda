//! Generalized IoU Loss for object detection
//!
//! **Pure WGSL**: Single implementation via WebGPU shader
//! Improves upon IoU by considering the smallest enclosing box

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GIoULossParams {
    num_boxes: u32,
    box_format: u32, // 0 = xyxy, 1 = xywh, 2 = cxcywh
    _padding: [u32; 2],
}

pub struct GIoULoss {
    pred_boxes: Tensor,
    target_boxes: Tensor,
    box_format: u32,
}

impl GIoULoss {
    /// Create GIoULoss operation
    pub fn new(pred_boxes: Tensor, target_boxes: Tensor, box_format: u32) -> Result<Self> {
        if box_format > 2 {
            return Err(BarracudaError::invalid_op(
                "GIoULoss",
                format!("box_format must be 0 (xyxy), 1 (xywh), or 2 (cxcywh), got {box_format}"),
            ));
        }

        Ok(Self {
            pred_boxes,
            target_boxes,
            box_format,
        })
    }

    /// WGSL shader source (embedded at compile time)
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                "../shaders/loss/giou_loss_f64.wgsl"
            ))
        });
        &SHADER
    }

    /// Execute GIoULoss on tensor
    pub fn execute(self) -> Result<Tensor> {
        let device = self.pred_boxes.device();
        let pred_shape = self.pred_boxes.shape();
        let target_shape = self.target_boxes.shape();

        if pred_shape.len() != 2 || target_shape.len() != 2 {
            return Err(BarracudaError::invalid_op(
                "GIoULoss",
                format!(
                    "boxes must be 2D [num_boxes, 4], got shapes {pred_shape:?} and {target_shape:?}"
                ),
            ));
        }

        if pred_shape[1] != 4 || target_shape[1] != 4 {
            return Err(BarracudaError::invalid_op(
                "GIoULoss",
                format!(
                    "boxes must have 4 coordinates, got {} and {}",
                    pred_shape[1], target_shape[1]
                ),
            ));
        }

        if pred_shape[0] != target_shape[0] {
            return Err(BarracudaError::invalid_op(
                "GIoULoss",
                format!(
                    "pred and target must have same number of boxes: {} != {}",
                    pred_shape[0], target_shape[0]
                ),
            ));
        }

        let num_boxes = pred_shape[0];

        let output_buffer = device.create_buffer_f32(num_boxes)?;

        let params = GIoULossParams {
            num_boxes: num_boxes as u32,
            box_format: self.box_format,
            _padding: [0; 2],
        };

        let params_buffer = device.create_uniform_buffer("GIoULoss Params", &params);

        ComputeDispatch::new(device, "GIoULoss")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.pred_boxes.buffer())
            .storage_read(1, self.target_boxes.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(num_boxes as u32)
            .submit();

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![num_boxes],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_giou_loss_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let num_boxes = 3;

        let pred_boxes = Tensor::from_vec_on(
            vec![
                0.0, 0.0, 10.0, 10.0, 5.0, 5.0, 15.0, 15.0, 10.0, 10.0, 20.0, 20.0,
            ],
            vec![num_boxes, 4],
            device.clone(),
        )
        .await
        .unwrap();

        let target_boxes = Tensor::from_vec_on(
            vec![
                1.0, 1.0, 11.0, 11.0, 6.0, 6.0, 16.0, 16.0, 11.0, 11.0, 21.0, 21.0,
            ],
            vec![num_boxes, 4],
            device.clone(),
        )
        .await
        .unwrap();

        let result = GIoULoss::new(pred_boxes, target_boxes, 0)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(result.shape(), &[num_boxes]);
    }
}
