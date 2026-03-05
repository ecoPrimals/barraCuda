// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU Decision Tree / Random Forest inference (f64).
//!
//! Batches N samples through M trees in a single GPU dispatch.
//! Each thread processes one (sample, tree) pair, traversing from root
//! to leaf via a tight loop (depth-capped at `max_depth`).
//!
//! The flat-array tree structure maps directly to wetSpring's
//! `bio::decision_tree::from_arrays()` format and to sklearn's exported
//! node arrays.
//!
//! ## Random Forest
//!
//! For a forest with M trees, all node arrays are concatenated;
//! `tree_offsets[t]` gives the first node index of tree t.
//! Post-inference majority-vote or mean aggregation is done on CPU.
//!
//! ## Absorbed from
//!
//! wetSpring handoff §Shader Design 2 (Feb 2026) — validated on sklearn
//! export: 65 nodes × 28 features, 744 samples, 100% prediction parity.

use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

// ─── GPU params (matches WGSL TreeParams) ────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TreeParamsGpu {
    n_samples: u32,
    n_features: u32,
    n_nodes_max: u32,
    n_trees: u32,
    max_depth: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// ─── Flat tree structure ──────────────────────────────────────────────────────

/// Flat-array representation of one or more decision trees.
///
/// Mirrors `bio::decision_tree::from_arrays()` in wetSpring.
/// For a single tree, `tree_offsets = [0]`.
pub struct FlatForest {
    /// Feature index for each internal node (unused for leaves).
    pub feature_idx: Vec<u32>,
    /// Split threshold for each internal node.
    pub thresholds: Vec<f64>,
    /// Left child node index; `< 0` indicates a leaf.
    pub left_child: Vec<i32>,
    /// Right child node index.
    pub right_child: Vec<i32>,
    /// Predicted class at each node (meaningful at leaves).
    pub predictions: Vec<u32>,
    /// Starting node index for each tree in the flat arrays.
    pub tree_offsets: Vec<u32>,
}

impl FlatForest {
    /// Wrap a single tree (root at node 0).
    pub fn single_tree(
        feature_idx: Vec<u32>,
        thresholds: Vec<f64>,
        left_child: Vec<i32>,
        right_child: Vec<i32>,
        predictions: Vec<u32>,
    ) -> Self {
        Self {
            feature_idx,
            thresholds,
            left_child,
            right_child,
            predictions,
            tree_offsets: vec![0],
        }
    }

    /// Number of trees in the forest.
    pub fn n_trees(&self) -> usize {
        self.tree_offsets.len()
    }
    /// Total number of nodes across all trees.
    pub fn n_nodes(&self) -> usize {
        self.left_child.len()
    }
}

// ─── Main operator ────────────────────────────────────────────────────────────

/// GPU-accelerated decision tree / random forest inference (f64).
///
/// # Example
///
/// ```rust,ignore
/// # use barracuda::prelude::WgpuDevice;
/// # use barracuda::ops::bio::tree_inference::{TreeInferenceGpu, FlatForest};
/// # crate::device::test_pool::tokio_block_on(async {
/// let device = WgpuDevice::new().await.unwrap();
/// let infer = TreeInferenceGpu::new(&device);
///
/// // Single stump: if feature[0] ≤ 0.5 → class 0, else class 1
/// let forest = FlatForest::single_tree(
///     vec![0],          // root: split on feature 0
///     vec![0.5],        // threshold
///     vec![1, -1, -1],  // left child: node 1 (leaf); -1 = leaf marker
///     vec![2, -1, -1],  // right child: node 2
///     vec![99, 0, 1],   // class at each node (99 = unused internal)
/// );
/// let samples = vec![0.3_f64, 0.7_f64]; // 2 samples × 1 feature
/// let output = infer.predict(&forest, &samples, 1).unwrap();
/// assert_eq!(output, vec![0, 1]); // left leaf, right leaf
/// # });
/// ```
pub struct TreeInferenceGpu {
    device: Arc<WgpuDevice>,
}

impl TreeInferenceGpu {
    /// Create tree inference operator for the given device.
    pub fn new(device: &WgpuDevice) -> Self {
        Self {
            device: Arc::new(device.clone()),
        }
    }

    /// Run batch inference for `n_samples` through all trees in `forest`.
    ///
    /// Returns `output[sample_id * n_trees + tree_id]` = predicted class.
    pub fn predict(
        &self,
        forest: &FlatForest,
        samples: &[f64],
        n_samples: usize,
    ) -> Result<Vec<u32>> {
        let dev = &self.device;
        let n_trees = forest.n_trees();
        let n_nodes = forest.n_nodes();
        let n_feat = samples.len() / n_samples;

        // ── Upload buffers ─────────────────────────────────────────────────────
        let upload = |data: &[u8], label: &str| {
            dev.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: data,
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };

        let samples_buf = upload(bytemuck::cast_slice(samples), "Tree samples");
        let feat_buf = upload(bytemuck::cast_slice(&forest.feature_idx), "Tree feat_idx");
        let thresh_buf = upload(bytemuck::cast_slice(&forest.thresholds), "Tree thresholds");
        let left_buf = upload(bytemuck::cast_slice(&forest.left_child), "Tree left");
        let right_buf = upload(bytemuck::cast_slice(&forest.right_child), "Tree right");
        let pred_buf = upload(bytemuck::cast_slice(&forest.predictions), "Tree preds");
        let offsets_buf = upload(bytemuck::cast_slice(&forest.tree_offsets), "Tree offsets");
        let output_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tree output"),
            size: (n_samples * n_trees * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = TreeParamsGpu {
            n_samples: n_samples as u32,
            n_features: n_feat as u32,
            n_nodes_max: n_nodes as u32,
            n_trees: n_trees as u32,
            max_depth: 32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tree params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // ── Dispatch (f64-aware for threshold comparisons) ───────────────────
        let total_threads = (n_samples * n_trees) as u32;
        ComputeDispatch::new(dev, "tree_inference")
            .shader(
                include_str!("../../shaders/bio/tree_inference_f64.wgsl"),
                "main",
            )
            .f64()
            .uniform(0, &params_buf)
            .storage_read(1, &samples_buf)
            .storage_read(2, &feat_buf)
            .storage_read(3, &thresh_buf)
            .storage_read(4, &left_buf)
            .storage_read(5, &right_buf)
            .storage_read(6, &pred_buf)
            .storage_read(7, &offsets_buf)
            .storage_rw(8, &output_buf)
            .dispatch(total_threads.div_ceil(WORKGROUP_SIZE_1D), 1, 1)
            .submit();

        dev.read_buffer_u32(&output_buf, n_samples * n_trees)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool;

    fn simple_stump() -> FlatForest {
        // if feature[0] <= 0.5 → class 0, else class 1
        // node 0: split; node 1: left leaf (class 0); node 2: right leaf (class 1)
        FlatForest::single_tree(
            vec![0, 0, 0],       // feature index (unused at leaves)
            vec![0.5, 0.0, 0.0], // threshold (unused at leaves)
            vec![1, -1, -1],     // left children
            vec![2, -1, -1],     // right children
            vec![99, 0, 1],      // predictions (99 = internal, ignored)
        )
    }

    #[tokio::test]
    async fn test_stump_two_samples() {
        let Some(device) = test_pool::get_test_device_if_f64_gpu_available().await else {
            return;
        };
        let infer = TreeInferenceGpu::new(&device);
        let forest = simple_stump();
        // sample 0: feature=0.3 → left leaf → class 0
        // sample 1: feature=0.8 → right leaf → class 1
        let samples = vec![0.3_f64, 0.8];
        let output = infer.predict(&forest, &samples, 2).unwrap();
        assert_eq!(output[0], 0, "sample 0 should be class 0");
        assert_eq!(output[1], 1, "sample 1 should be class 1");
    }

    #[tokio::test]
    async fn test_deeper_tree() {
        let Some(device) = test_pool::get_test_device_if_f64_gpu_available().await else {
            return;
        };
        let infer = TreeInferenceGpu::new(&device);
        // Two-level tree: 7 nodes (0..6)
        //        0 (feat0 ≤ 0.5)
        //       / \
        //      1          4
        // (feat1≤0.5)  (feat1≤0.5)
        //     / \          / \
        //    2   3        5   6
        //  cls0 cls1    cls2 cls3
        let forest = FlatForest::single_tree(
            vec![0, 1, 0, 0, 1, 0, 0], // node 0: feat0; node 1: feat1; node 4: feat1
            vec![0.5, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0],
            vec![1, 2, -1, -1, 5, -1, -1],
            vec![4, 3, -1, -1, 6, -1, -1],
            vec![99, 99, 0, 1, 99, 2, 3],
        );
        // 4 samples, 2 features each: (f0, f1)
        let samples = vec![0.3_f64, 0.3, 0.3, 0.8, 0.8, 0.3, 0.8, 0.8];
        let output = infer.predict(&forest, &samples, 4).unwrap();
        assert_eq!(output[0], 0, "left-left → class 0");
        assert_eq!(output[1], 1, "left-right → class 1");
        assert_eq!(output[2], 2, "right-left → class 2");
        assert_eq!(output[3], 3, "right-right → class 3");
    }
}
