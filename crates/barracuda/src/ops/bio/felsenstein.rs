// SPDX-License-Identifier: AGPL-3.0-or-later

//! Felsenstein Pruning Phylogenetic Likelihood (f64).
//!
//! Computes per-site conditional likelihoods for maximum-likelihood
//! phylogenetics using Felsenstein's (1981) pruning algorithm.
//!
//! ## Algorithm
//!
//! Bottom-up (postorder) traversal: for each internal node `n` with
//! children `l` (left) and `r` (right):
//!
//! ```text
//! L[n][site][s] = (Σ_j P_l[s][j] × L[l][site][j])
//!               × (Σ_k P_r[s][k] × L[r][site][k])
//! ```
//!
//! where `P_x` is the `n_states × n_states` transition probability matrix
//! for branch `x`, pre-computed via matrix exponentiation on CPU.
//!
//! ## GPU Strategy
//!
//! Level-order parallelism: all nodes at the same tree depth are independent
//! given their children.  One dispatch per tree level; threads cover
//! `(site, node_in_level)` pairs.
//!
//! ## Final log-likelihood
//!
//! After the root is computed, use [`LogsumexpWgsl`] (already in barracuda) or
//! the CPU to sum `π_s × L[root][site][s]` over states and then over sites.
//!
//! ## Absorbed from
//!
//! wetSpring handoff §Shader Design 3 (Feb 2026) — targets Exp019 PhyloNet-HMM
//! and Liu 2014 maximum-likelihood phylogenetics.

use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

// ─── GPU params (matches WGSL FelsensteinParams) ─────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FelsensteinParamsGpu {
    n_sites: u32,
    n_nodes_this_level: u32,
    n_states: u32,
    n_nodes_total: u32,
}

// ─── Tree topology ────────────────────────────────────────────────────────────

/// Topology of an unrooted/rooted binary tree with node-indexed arrays.
pub struct PhyloTree {
    /// Left child of each node; `-1` for leaves.
    pub left_child: Vec<i32>,
    /// Right child of each node; `-1` for leaves.
    pub right_child: Vec<i32>,
    /// Branch length for each node (its parent edge).
    pub branch_lengths: Vec<f64>,
    /// Level-order groups: `levels[k]` = node indices at depth k (leaves first).
    pub levels: Vec<Vec<u32>>,
}

impl PhyloTree {
    /// Number of nodes in the tree.
    pub fn n_nodes(&self) -> usize {
        self.left_child.len()
    }
}

// ─── Main operator ────────────────────────────────────────────────────────────

/// GPU-accelerated Felsenstein pruning likelihood (f64).
///
/// # Example
///
/// ```rust,ignore
/// # use barracuda::prelude::WgpuDevice;
/// # use barracuda::ops::bio::felsenstein::{FelsensteinGpu, PhyloTree};
/// # crate::device::test_pool::tokio_block_on(async {
/// let device = WgpuDevice::new().await.unwrap();
/// let pruner = FelsensteinGpu::new(&device);
///
/// // Tiny tree: root(0) → leaf1(1) + leaf2(2), 2 sites, 4 DNA states
/// let tree = PhyloTree {
///     left_child:     vec![1, -1, -1],
///     right_child:    vec![2, -1, -1],
///     branch_lengths: vec![0.1, 0.1, 0.1],
///     levels: vec![vec![1, 2], vec![0]],  // leaves first, then root
/// };
/// // Tip likelihoods: 1.0 at observed state, 0.0 elsewhere
/// let n_sites = 2usize; let n_states = 4usize;
/// let mut lik = vec![0.0_f64; 3 * n_sites * n_states];
/// // Leaf 1 observes A(0) at site 0, C(1) at site 1
/// // Index formula: node * n_sites * n_states + site * n_states + state
/// lik[n_sites * n_states] = 1.0; // leaf1, site0, A
/// lik[n_sites * n_states + n_states + 1] = 1.0; // leaf1, site1, C
/// // Leaf 2 observes A(0) at both sites
/// lik[2 * n_sites * n_states] = 1.0; // leaf2, site0, A
/// lik[2 * n_sites * n_states + n_states] = 1.0; // leaf2, site1, A
/// // Transition matrices (identity for illustration)
/// let eye4: Vec<f64> = vec![1.,0.,0.,0., 0.,1.,0.,0., 0.,0.,1.,0., 0.,0.,0.,1.];
/// let tp: Vec<f64> = eye4.iter().cloned().cycle().take(3 * 16).collect();
/// let result = pruner.prune(&tree, &lik, &tp, n_sites, n_states).unwrap();
/// // result.likelihoods[root_idx] contains site likelihoods at root
/// # });
/// ```
pub struct FelsensteinGpu {
    device: Arc<WgpuDevice>,
}

/// Result of a Felsenstein pruning pass.
pub struct FelsensteinResult {
    /// Full likelihood array [n_nodes × n_sites × n_states].
    pub likelihoods: Vec<f64>,
    /// Total nodes in the tree.
    pub n_nodes: usize,
    /// Number of alignment sites.
    pub n_sites: usize,
    /// Number of character states (4 for DNA).
    pub n_states: usize,
}

impl FelsensteinResult {
    /// Slice the root likelihoods: `[n_sites × n_states]`.
    pub fn root_likelihoods(&self, root_node: usize) -> &[f64] {
        let stride = self.n_sites * self.n_states;
        &self.likelihoods[root_node * stride..(root_node + 1) * stride]
    }

    /// Compute log-likelihood at the root given stationary distribution `pi`.
    pub fn log_likelihood(&self, root_node: usize, pi: &[f64]) -> f64 {
        let root = self.root_likelihoods(root_node);
        let mut log_lik = 0.0_f64;
        for site in 0..self.n_sites {
            let mut site_lik = 0.0_f64;
            for s in 0..self.n_states {
                site_lik += pi[s] * root[site * self.n_states + s];
            }
            log_lik += site_lik.max(f64::MIN_POSITIVE).ln();
        }
        log_lik
    }
}

impl FelsensteinGpu {
    /// Create Felsenstein pruner for given device.
    pub fn new(device: &WgpuDevice) -> Self {
        Self {
            device: Arc::new(device.clone()),
        }
    }

    /// Run Felsenstein pruning.
    ///
    /// # Arguments
    /// - `tree`             : topology with level-order groups
    /// - `tip_likelihoods`  : `[n_nodes × n_sites × n_states]` with leaf rows
    ///   pre-filled (1.0 at observed state, 0.0 elsewhere)
    /// - `transition_probs` : pre-computed `[n_nodes × n_states × n_states]`
    ///   transition matrices (one per node, indexed by child)
    /// - `n_sites`          : number of alignment columns
    /// - `n_states`         : 4 for DNA, 20 for protein
    pub fn prune(
        &self,
        tree: &PhyloTree,
        tip_likelihoods: &[f64],
        transition_probs: &[f64],
        n_sites: usize,
        n_states: usize,
    ) -> Result<FelsensteinResult> {
        let dev = &self.device;
        let n_nodes = tree.n_nodes();

        // ── Upload read-only buffers ───────────────────────────────────────────
        let left_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Fels left_child"),
                contents: bytemuck::cast_slice(&tree.left_child),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let right_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Fels right_child"),
                contents: bytemuck::cast_slice(&tree.right_child),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let tp_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Fels transition_probs"),
                contents: bytemuck::cast_slice(transition_probs),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // ── Mutable likelihood buffer (init from tips) ────────────────────────
        let lik_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Fels likelihoods"),
                contents: bytemuck::cast_slice(tip_likelihoods),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        const SHADER: &str = include_str!("../../shaders/bio/felsenstein_f64.wgsl");

        // ── One dispatch per tree level (bottom-up) ───────────────────────────
        for level_nodes in &tree.levels {
            if level_nodes.is_empty() {
                continue;
            }
            // Check if any node in this level is internal (has children)
            let has_internal = level_nodes
                .iter()
                .any(|&n| tree.left_child[n as usize] >= 0);
            if !has_internal {
                continue;
            } // all leaves — nothing to compute

            let node_ids_buf = dev
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(level_nodes),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let params = FelsensteinParamsGpu {
                n_sites: n_sites as u32,
                n_nodes_this_level: level_nodes.len() as u32,
                n_states: n_states as u32,
                n_nodes_total: n_nodes as u32,
            };
            let params_buf = dev
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let total = (level_nodes.len() * n_sites) as u32;
            ComputeDispatch::new(dev, "FelsensteinF64")
                .shader(SHADER, "main")
                .f64()
                .uniform(0, &params_buf)
                .storage_read(1, &node_ids_buf)
                .storage_read(2, &left_buf)
                .storage_read(3, &right_buf)
                .storage_read(4, &tp_buf)
                .storage_rw(5, &lik_buf)
                .dispatch(total.div_ceil(WORKGROUP_SIZE_1D), 1, 1)
                .submit();
        }

        let likelihoods = dev.read_buffer_f64(&lik_buf, n_nodes * n_sites * n_states)?;

        Ok(FelsensteinResult {
            likelihoods,
            n_nodes,
            n_sites,
            n_states,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool;

    /// Identity 4×4 transition matrix (any branch length, for deterministic testing).
    fn eye_tp(n_nodes: usize) -> Vec<f64> {
        let eye: [f64; 16] = [
            1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
        ];
        eye.iter().cloned().cycle().take(n_nodes * 16).collect()
    }

    #[tokio::test]
    async fn test_root_inherits_identical_tips() {
        // Root(0) → Leaf1(1) + Leaf2(2), 1 site, 4 states
        // Both leaves observe A (state 0) → root should be 1.0 at A, 0.0 elsewhere
        let Some(device) = test_pool::get_test_device_if_f64_gpu_available().await else {
            return;
        };
        let pruner = FelsensteinGpu::new(&device);

        let tree = PhyloTree {
            left_child: vec![1, -1, -1],
            right_child: vec![2, -1, -1],
            branch_lengths: vec![0.1, 0.1, 0.1],
            levels: vec![vec![1, 2], vec![0]],
        };

        let n_sites = 1;
        let n_states = 4;
        let mut lik = vec![0.0_f64; 3 * n_sites * n_states];
        // Leaf 1: observe A (state 0)
        lik[n_states] = 1.0;
        // Leaf 2: observe A (state 0)
        lik[2 * n_states] = 1.0;

        let tp = eye_tp(3);
        let result = pruner.prune(&tree, &lik, &tp, n_sites, n_states).unwrap();

        let root = result.root_likelihoods(0);
        assert!(
            (root[0] - 1.0).abs() < 1e-9,
            "root L[A] should be 1.0, got {}",
            root[0]
        );
        assert!(
            root[1].abs() < 1e-9,
            "root L[C] should be 0.0, got {}",
            root[1]
        );
    }

    #[tokio::test]
    async fn test_log_likelihood_two_sites() {
        // Both leaves observe A at site 0 and C at site 1
        // With equal stationary dist (pi = 0.25 each) and identity TP
        // L[root][0] = A: (1×1)×(1×1) = 1.0; log-lik contribution: log(0.25 × 1.0) = log(0.25)
        // L[root][1] = C: similar; log-lik = 2 × log(0.25)
        let Some(device) = test_pool::get_test_device_if_f64_gpu_available().await else {
            return;
        };
        let pruner = FelsensteinGpu::new(&device);

        let tree = PhyloTree {
            left_child: vec![1, -1, -1],
            right_child: vec![2, -1, -1],
            branch_lengths: vec![0.1, 0.1, 0.1],
            levels: vec![vec![1, 2], vec![0]],
        };

        let n_sites = 2;
        let n_states = 4;
        let mut lik = vec![0.0_f64; 3 * n_sites * n_states];
        lik[n_sites * n_states] = 1.0; // leaf1, site0, A
        lik[n_sites * n_states + n_states + 1] = 1.0; // leaf1, site1, C
        lik[2 * n_sites * n_states] = 1.0; // leaf2, site0, A
        lik[2 * n_sites * n_states + n_states + 1] = 1.0; // leaf2, site1, C

        let tp = eye_tp(3);
        let result = pruner.prune(&tree, &lik, &tp, n_sites, n_states).unwrap();

        let pi = vec![0.25_f64; 4];
        let log_lik = result.log_likelihood(0, &pi);
        let expected = 2.0 * (0.25_f64).ln();
        assert!(
            (log_lik - expected).abs() < 1e-9,
            "log_lik={log_lik:.6}, expected={expected:.6}"
        );
    }
}
