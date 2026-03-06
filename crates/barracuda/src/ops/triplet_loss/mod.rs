// SPDX-License-Identifier: AGPL-3.0-or-later
//! Triplet Loss - GPU-accelerated metric learning loss
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (new shader!)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready for metric learning)
//!
//! ## Algorithm
//!
//! ```text
//! For each triplet (anchor, positive, negative):
//!
//! d_pos = distance(anchor, positive)   // Should be small
//! d_neg = distance(anchor, negative)   // Should be large
//!
//! loss = max(0, d_pos - d_neg + margin)
//! ```
//!
//! **Goal**: Learn embeddings where similar items are close, dissimilar items are far
//!
//! **Implementation**: Single-pass GPU distance computation
//!
//! **Key Properties**:
//! - Pulls positives closer to anchors
//! - Pushes negatives farther from anchors
//! - Margin ensures minimum separation
//! - No explicit classification needed
//!
//! **Used By**: Face recognition, person re-ID, similarity search, metric learning
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::tensor::Tensor;
//!
//! let anchors = Tensor::randn(vec![32, 128]).await?;   // [batch, embedding_dim]
//! let positives = Tensor::randn(vec![32, 128]).await?; // Same class as anchors
//! let negatives = Tensor::randn(vec![32, 128]).await?; // Different class
//!
//! let loss = anchors.triplet_loss(&positives, &negatives, 0.2)?;
//! ```

mod compute;

#[cfg(test)]
mod tests;

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Triplet loss parameters for WGSL shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct TripletParams {
    pub batch_size: u32,
    pub embedding_dim: u32,
    pub margin: f32,
    pub distance_type: u32,
}

/// Distance metric for triplet loss
#[derive(Copy, Clone, Debug)]
pub enum DistanceMetric {
    /// L2 Euclidean distance (default)
    L2,
    /// Cosine distance (1 - cosine similarity)
    Cosine,
}

/// Triplet Loss operation
///
/// **Deep Debt**: Uses new WGSL shader for metric learning
pub struct TripletLoss {
    anchors: Tensor,
    positives: Tensor,
    negatives: Tensor,
    margin: f32,
    distance_metric: DistanceMetric,
}

impl TripletLoss {
    /// Create new Triplet loss operation
    /// **Deep Debt**: Validates all inputs for shape compatibility
    /// # Errors
    /// Returns [`Err`] if anchor, positive, and negative shapes do not match, tensors are not 2D,
    /// or margin is negative.
    pub fn new(
        anchors: Tensor,
        positives: Tensor,
        negatives: Tensor,
        margin: f32,
        distance_metric: DistanceMetric,
    ) -> Result<Self> {
        // Validate shapes match
        if anchors.shape() != positives.shape() {
            return Err(BarracudaError::shape_mismatch(
                anchors.shape().to_vec(),
                positives.shape().to_vec(),
            ));
        }
        if anchors.shape() != negatives.shape() {
            return Err(BarracudaError::shape_mismatch(
                anchors.shape().to_vec(),
                negatives.shape().to_vec(),
            ));
        }

        // Validate shape is 2D [batch, embedding_dim]
        if anchors.shape().len() != 2 {
            return Err(BarracudaError::invalid_op(
                "TripletLoss",
                format!(
                    "Expected 2D tensors [batch, embedding_dim], got shape {:?}",
                    anchors.shape()
                ),
            ));
        }

        // Validate margin
        if margin < 0.0 {
            return Err(BarracudaError::invalid_op(
                "TripletLoss",
                format!("margin must be non-negative, got {margin}"),
            ));
        }

        Ok(Self {
            anchors,
            positives,
            negatives,
            margin,
            distance_metric,
        })
    }

    /// WGSL shader source
    pub(super) fn shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                "../../shaders/loss/triplet_loss_f64.wgsl"
            ))
        });
        &SHADER
    }

    /// Get anchors tensor
    pub(super) fn anchors(&self) -> &Tensor {
        &self.anchors
    }

    /// Get positives tensor
    pub(super) fn positives(&self) -> &Tensor {
        &self.positives
    }

    /// Get negatives tensor
    pub(super) fn negatives(&self) -> &Tensor {
        &self.negatives
    }

    /// Get margin
    pub(super) fn margin(&self) -> f32 {
        self.margin
    }

    /// Get distance metric
    pub(super) fn distance_metric(&self) -> DistanceMetric {
        self.distance_metric
    }
}

// ═══════════════════════════════════════════════════════════════
// TENSOR API INTEGRATION
// ═══════════════════════════════════════════════════════════════

impl Tensor {
    /// Triplet loss for metric learning
    /// **Deep Debt**: Essential for similarity learning and face recognition
    /// # Arguments
    /// - `positives`: Similar embeddings [same shape as anchors]
    /// - `negatives`: Dissimilar embeddings [same shape as anchors]
    /// - `margin`: Minimum separation between positive and negative (typically 0.2-1.0)
    /// # Returns
    /// - Loss tensor [`batch_size`] (one value per triplet)
    /// # Example
    /// ```rust,ignore
    /// // L2 distance (default)
    /// let loss = anchors.triplet_loss(&positives, &negatives, 0.2)?;
    /// // Cosine distance
    /// let loss = anchors.triplet_loss_cosine(&positives, &negatives, 0.1)?;
    /// ```
    /// # Note
    /// - Embeddings should be [batch, `embedding_dim`]
    /// - Margin controls how far negatives should be from positives
    /// - Larger margin = stricter separation requirement
    /// # Errors
    /// Returns [`Err`] if validation fails (shape mismatch, invalid margin), buffer allocation fails,
    /// or GPU dispatch fails (e.g., device lost).
    pub fn triplet_loss(self, positives: &Self, negatives: &Self, margin: f32) -> Result<Self> {
        TripletLoss::new(
            self,
            positives.clone(),
            negatives.clone(),
            margin,
            DistanceMetric::L2,
        )?
        .execute()
    }

    /// Triplet loss with cosine distance metric
    /// **Deep Debt**: Useful when embeddings are normalized
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn triplet_loss_cosine(
        self,
        positives: &Self,
        negatives: &Self,
        margin: f32,
    ) -> Result<Self> {
        TripletLoss::new(
            self,
            positives.clone(),
            negatives.clone(),
            margin,
            DistanceMetric::Cosine,
        )?
        .execute()
    }
}
