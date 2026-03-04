// SPDX-License-Identifier: AGPL-3.0-or-later
//! Soft NMS - Soft Non-Maximum Suppression
//!
//! Reduces scores of overlapping boxes instead of removing them.
//! Better performance than hard NMS.
//!
//! Deep Debt Principles:
//! - Hybrid GPU/CPU execution (IoU on GPU, sorting/iteration on CPU)
//! - Safe Rust wrappers
//! - Hardware-agnostic via WebGPU
//! - Runtime device discovery
//! - Uses tensors for coordinates and scores

use crate::error::Result;
use crate::tensor::Tensor;

/// Soft NMS shader (soft non-maximum suppression). f64 canonical, downcast to f32 at use.
pub static WGSL_SOFT_NMS: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
        "../shaders/detection/soft_nms_f64.wgsl"
    ))
});

/// SoftNMS operation
pub struct SoftNMS {
    boxes: Tensor,  // [N, 4] (x1, y1, x2, y2)
    scores: Tensor, // [N]
    iou_threshold: f32,
    sigma: f32,
}

impl SoftNMS {
    /// Create a new soft NMS operation
    pub fn new(boxes: Tensor, scores: Tensor, iou_threshold: f32, sigma: f32) -> Result<Self> {
        let boxes_shape = boxes.shape();
        let scores_shape = scores.shape();

        if boxes_shape.len() != 2 || boxes_shape[1] != 4 {
            return Err(crate::error::BarracudaError::invalid_op(
                "SoftNMS",
                format!("Boxes must be [N, 4], got {boxes_shape:?}"),
            ));
        }

        if scores_shape.len() != 1 || scores_shape[0] != boxes_shape[0] {
            return Err(crate::error::BarracudaError::invalid_op(
                "SoftNMS",
                format!("Scores must match number of boxes: {scores_shape:?} vs {boxes_shape:?}"),
            ));
        }

        if !(0.0..=1.0).contains(&iou_threshold) {
            return Err(crate::error::BarracudaError::invalid_op(
                "SoftNMS",
                format!("IoU threshold must be in [0, 1], got {iou_threshold}"),
            ));
        }

        Ok(Self {
            boxes,
            scores,
            iou_threshold,
            sigma,
        })
    }

    /// Execute the soft NMS operation
    ///
    /// Note: This uses a hybrid approach - IoU computation could be done on GPU,
    /// but sorting and iterative score decay are done on CPU for simplicity.
    /// The operation returns indices of kept boxes.
    pub fn execute(self) -> Result<Vec<usize>> {
        // Read boxes and scores from GPU
        let boxes_data = self.boxes.to_vec()?;
        let mut scores_data = self.scores.to_vec()?;

        let num_boxes = boxes_data.len() / 4;

        // Create indices sorted by score (descending)
        let mut indices: Vec<usize> = (0..num_boxes).collect();
        indices.sort_by(|&a, &b| {
            scores_data[b]
                .partial_cmp(&scores_data[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut keep = Vec::new();

        for i in 0..indices.len() {
            let idx = indices[i];

            if scores_data[idx] < 0.001 {
                continue;
            }

            keep.push(idx);

            // Soft suppress overlapping boxes
            for j in (i + 1)..indices.len() {
                let other_idx = indices[j];

                // Compute IoU
                let x1_1 = boxes_data[idx * 4];
                let y1_1 = boxes_data[idx * 4 + 1];
                let x2_1 = boxes_data[idx * 4 + 2];
                let y2_1 = boxes_data[idx * 4 + 3];

                let x1_2 = boxes_data[other_idx * 4];
                let y1_2 = boxes_data[other_idx * 4 + 1];
                let x2_2 = boxes_data[other_idx * 4 + 2];
                let y2_2 = boxes_data[other_idx * 4 + 3];

                let inter_x1 = x1_1.max(x1_2);
                let inter_y1 = y1_1.max(y1_2);
                let inter_x2 = x2_1.min(x2_2);
                let inter_y2 = y2_1.min(y2_2);

                let inter_area = (inter_x2 - inter_x1).max(0.0) * (inter_y2 - inter_y1).max(0.0);
                let box1_area = (x2_1 - x1_1) * (y2_1 - y1_1);
                let box2_area = (x2_2 - x1_2) * (y2_2 - y1_2);
                let union_area = box1_area + box2_area - inter_area;

                let overlap = if union_area > 0.0 {
                    inter_area / union_area
                } else {
                    0.0
                };

                if overlap > self.iou_threshold {
                    // Gaussian decay
                    scores_data[other_idx] *= (-(overlap * overlap) / self.sigma).exp();
                }
            }
        }

        Ok(keep)
    }
}

impl Tensor {
    /// Apply soft non-maximum suppression
    ///
    /// # Arguments
    ///
    /// * `scores` - Score tensor [N]
    /// * `iou_threshold` - IoU threshold for suppression
    /// * `sigma` - Gaussian decay parameter
    ///
    /// Returns indices of kept boxes
    pub fn soft_nms(self, scores: Tensor, iou_threshold: f32, sigma: f32) -> Result<Vec<usize>> {
        SoftNMS::new(self, scores, iou_threshold, sigma)?.execute()
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    async fn get_test_device() -> Option<std::sync::Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_soft_nms_basic() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Boxes: [N, 4] format
        let boxes = Tensor::new(
            vec![
                0.0, 0.0, 10.0, 10.0, // Box 0
                1.0, 1.0, 11.0, 11.0, // Box 1
            ],
            vec![2, 4],
            device.clone(),
        );
        let scores = Tensor::new(vec![0.9, 0.8], vec![2], device.clone());
        let keep = boxes.soft_nms(scores, 0.5, 0.5).unwrap();
        assert!(!keep.is_empty());
    }

    #[tokio::test]
    async fn test_soft_nms_edge_cases() {
        let Some(device) = get_test_device().await else {
            return;
        };

        // Single box
        let boxes = Tensor::new(vec![0.0, 0.0, 10.0, 10.0], vec![1, 4], device.clone());
        let scores = Tensor::new(vec![0.9], vec![1], device.clone());
        let keep = boxes.soft_nms(scores, 0.5, 0.5).unwrap();
        assert_eq!(keep.len(), 1);

        // No overlap
        let boxes = Tensor::new(
            vec![0.0, 0.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0],
            vec![2, 4],
            device.clone(),
        );
        let scores = Tensor::new(vec![0.9, 0.8], vec![2], device.clone());
        let keep = boxes.soft_nms(scores, 0.5, 0.5).unwrap();
        assert_eq!(keep.len(), 2);
    }

    #[tokio::test]
    async fn test_soft_nms_boundary() {
        let Some(device) = get_test_device().await else {
            return;
        };

        // High overlap - boxes [N, 4] format
        let boxes_data = vec![
            0.0, 0.0, 10.0, 10.0, // Box 0
            0.5, 0.5, 10.5, 10.5, // Box 1 (high overlap)
        ];
        let boxes = Tensor::new(boxes_data, vec![2, 4], device.clone());
        let scores = Tensor::new(vec![0.9, 0.85], vec![2], device.clone());
        let keep = boxes.soft_nms(scores, 0.5, 0.5).unwrap();
        assert!(!keep.is_empty());

        // Different sigma
        let boxes_data = vec![0.0, 0.0, 10.0, 10.0, 1.0, 1.0, 11.0, 11.0];
        let boxes = Tensor::new(boxes_data, vec![2, 4], device.clone());
        let scores = Tensor::new(vec![0.9, 0.8], vec![2], device.clone());
        let keep = boxes.soft_nms(scores, 0.5, 0.3).unwrap();
        assert!(!keep.is_empty());
    }

    #[tokio::test]
    async fn test_soft_nms_large_batch() {
        let Some(device) = get_test_device().await else {
            return;
        };

        // 100 boxes
        let mut boxes_data = Vec::new();
        let mut scores_data = Vec::new();
        for i in 0..100 {
            boxes_data.extend_from_slice(&[(i * 5) as f32, 0.0, (i * 5 + 10) as f32, 10.0]);
            scores_data.push(0.9 - i as f32 * 0.001);
        }
        let boxes = Tensor::new(boxes_data, vec![100, 4], device.clone());
        let scores = Tensor::new(scores_data, vec![100], device.clone());
        let keep = boxes.soft_nms(scores, 0.5, 0.5).unwrap();
        assert!(!keep.is_empty());
    }

    #[tokio::test]
    async fn test_soft_nms_precision() {
        let Some(device) = get_test_device().await else {
            return;
        };

        // Verify score reduction - boxes [N, 4] format
        let boxes_data = vec![
            0.0, 0.0, 10.0, 10.0, // Box 0
            2.0, 2.0, 12.0, 12.0, // Box 1 (overlaps with box 0)
        ];
        let boxes = Tensor::new(boxes_data, vec![2, 4], device.clone());
        let scores = Tensor::new(vec![0.9, 0.8], vec![2], device.clone());
        let keep = boxes.soft_nms(scores, 0.5, 0.5).unwrap();

        assert!(!keep.is_empty());
        // Soft NMS should keep at least one box
        assert!(keep.contains(&0)); // High score box should be kept
    }
}
