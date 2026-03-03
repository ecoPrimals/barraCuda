//! NMS - Non-Maximum Suppression (Pure GPU)
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute
//!
//! Filters overlapping bounding boxes in object detection.
//! Used in YOLO, Faster R-CNN, etc.
//!
//! Algorithm (Pure GPU):
//! 1. Compute IoU matrix on GPU (parallel)
//! 2. Sort indices by score (CPU - acceptable for small sets)
//! 3. Mark suppressed boxes on GPU (parallel)
//! 4. Compact results on GPU (parallel with atomics)

use crate::error::{BarracudaError, Result};

/// WGSL kernel for intersection-over-union computation (f64).
pub const WGSL_IOU_F64: &str = include_str!("../../shaders/misc/iou_f64.wgsl");

mod compute;

#[cfg(test)]
mod tests;

/// Bounding box representation
#[derive(Clone, Debug)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

/// NMS operation
pub struct NMS {
    boxes: Vec<BoundingBox>,
    iou_threshold: f32,
}

impl NMS {
    /// Create NMS operation
    pub fn new(boxes: Vec<BoundingBox>, iou_threshold: f32) -> Result<Self> {
        if !(0.0..=1.0).contains(&iou_threshold) {
            return Err(BarracudaError::invalid_op(
                "NMS",
                format!("iou_threshold must be in [0, 1], got {iou_threshold}"),
            ));
        }

        Ok(Self {
            boxes,
            iou_threshold,
        })
    }

    /// Get boxes
    pub(super) fn boxes(&self) -> &[BoundingBox] {
        &self.boxes
    }

    /// Get IoU threshold
    pub(super) fn iou_threshold(&self) -> f32 {
        self.iou_threshold
    }

    /// WGSL shader source (f64 canonical, downcast to f32 at compile)
    pub(super) fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                "../../shaders/detection/nms_f64.wgsl"
            ))
        });
        SHADER.as_str()
    }
}

/// Compute IoU between two boxes (public for use by soft_nms)
pub fn compute_iou(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    let x1 = box1.x1.max(box2.x1);
    let y1 = box1.y1.max(box2.y1);
    let x2 = box1.x2.min(box2.x2);
    let y2 = box1.y2.min(box2.y2);

    let intersection = ((x2 - x1).max(0.0)) * ((y2 - y1).max(0.0));

    let area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    let area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    let union = area1 + area2 - intersection;

    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

/// Convenience function for NMS
pub fn nms(boxes: Vec<BoundingBox>, iou_threshold: f32) -> Result<Vec<usize>> {
    NMS::new(boxes, iou_threshold)?.execute()
}
