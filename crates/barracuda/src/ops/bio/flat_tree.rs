// SPDX-License-Identifier: AGPL-3.0-or-later

//! Generic CSR `FlatTree` for phylogenetic and classification tree dispatch.
//!
//! A `FlatTree` stores tree topology in Compressed Sparse Row (CSR) format:
//! parent indices, branch lengths, and per-node metadata. This is the common
//! layout consumed by GPU shaders (Felsenstein pruning, `UniFrac` propagation,
//! bootstrap resampling, NJ clustering, DTL reconciliation).
//!
//! Provenance: wetSpring metagenomics + neuralSpring metalForge → toadStool

use crate::error::{BarracudaError, Result as BarracudaResult};

/// Generic CSR tree with parent-indexed topology.
///
/// Nodes are numbered `0..n_nodes`. Leaves occupy indices `0..n_leaves`.
/// Internal nodes occupy `n_leaves..n_nodes`. The root has `parent[root] == -1`.
#[derive(Debug, Clone)]
pub struct FlatTree {
    /// Parent index for each node; `-1` for the root.
    pub parent: Vec<i32>,
    /// Branch length (edge weight) for each node.
    pub branch_length: Vec<f64>,
    /// Number of leaf nodes (always stored at indices `0..n_leaves`).
    pub n_leaves: usize,
}

impl FlatTree {
    /// Build a `FlatTree` from a Newick format string.
    /// Parses strings like `((A:0.1,B:0.2):0.3,C:0.4);` and constructs the tree
    /// with parent indices, branch lengths, and leaf ordering. Level ordering
    /// is computed automatically via [`bottom_up_levels`](Self::bottom_up_levels).
    /// # Errors
    /// Returns an error if the Newick string is malformed or invalid.
    pub fn from_newick(newick: &str) -> BarracudaResult<Self> {
        let newick = newick.trim();
        if newick.is_empty() {
            return Err(BarracudaError::InvalidInput {
                message: "Newick string cannot be empty".to_string(),
            });
        }
        let s = newick
            .strip_suffix(';')
            .ok_or_else(|| BarracudaError::InvalidInput {
                message: "Newick string must end with semicolon".to_string(),
            })?;
        let (root_idx, nodes) = parse_newick_subtree(s, 0)?;
        if root_idx.is_none() {
            return Err(BarracudaError::InvalidInput {
                message: "Newick parse produced no root".to_string(),
            });
        }
        let root = root_idx.ok_or_else(|| BarracudaError::InvalidInput {
            message: "Newick parse produced no root (unreachable)".to_string(),
        })?;
        build_flat_tree_from_parsed(nodes, root)
    }

    /// Build a `FlatTree` from an edge list.
    /// Takes `(parent, child, branch_length)` tuples and constructs the tree.
    /// Node indices are assigned automatically: leaves first (`0..n_leaves`),
    /// then internal nodes, with the root last. Level ordering is computed
    /// automatically via [`bottom_up_levels`](Self::bottom_up_levels).
    #[must_use]
    pub fn from_edges(edges: &[(usize, usize, f64)]) -> Self {
        if edges.is_empty() {
            return FlatTree {
                parent: vec![-1],
                branch_length: vec![0.0],
                n_leaves: 1,
            };
        }
        let (parent, branch_length, n_leaves) = build_from_edges(edges);
        FlatTree {
            parent,
            branch_length,
            n_leaves,
        }
    }

    /// Total number of nodes in the tree.
    #[must_use]
    pub fn n_nodes(&self) -> usize {
        self.parent.len()
    }

    /// Validate structural invariants.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if the tree structure violates invariants (branch
    /// length mismatch, invalid root, out-of-bounds parent, etc.).
    pub fn validate(&self) -> crate::error::Result<()> {
        use crate::error::BarracudaError;

        let n = self.n_nodes();
        if self.branch_length.len() != n {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "branch_length len {} != n_nodes {n}",
                    self.branch_length.len()
                ),
            });
        }
        if self.n_leaves > n {
            return Err(BarracudaError::InvalidInput {
                message: format!("n_leaves {} exceeds n_nodes {n}", self.n_leaves),
            });
        }
        let root_count = self.parent.iter().filter(|&&p| p < 0).count();
        if root_count != 1 {
            return Err(BarracudaError::InvalidInput {
                message: format!("tree must have exactly one root (found {root_count})"),
            });
        }
        for (i, &p) in self.parent.iter().enumerate() {
            if p >= 0 && p as usize >= n {
                return Err(BarracudaError::InvalidInput {
                    message: format!("parent[{i}]={p} out of bounds (n={n})"),
                });
            }
            if p >= 0 && p as usize == i {
                return Err(BarracudaError::InvalidInput {
                    message: format!("self-loop at node {i}"),
                });
            }
        }
        Ok(())
    }

    /// Convert to GPU-ready buffers (parent as `i32`, `branch_length` as `f64`).
    #[must_use]
    pub fn to_gpu_arrays(&self) -> (Vec<i32>, Vec<f64>) {
        (self.parent.clone(), self.branch_length.clone())
    }

    /// Build tree levels bottom-up for multi-pass GPU dispatch.
    /// Returns a vector of levels, each containing node indices for that level.
    /// Level 0 = leaves, last level = root.
    #[must_use]
    pub fn bottom_up_levels(&self) -> Vec<Vec<usize>> {
        let n = self.n_nodes();
        let mut depth = vec![0u32; n];
        let mut max_depth = 0u32;

        // Compute depths top-down
        let root = self.parent.iter().position(|&p| p < 0).unwrap_or(0);
        depth[root] = 0;

        // BFS from root
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(root);
        let mut children: Vec<Vec<usize>> = vec![vec![]; n];
        for (i, &p) in self.parent.iter().enumerate() {
            if p >= 0 {
                children[p as usize].push(i);
            }
        }
        while let Some(node) = queue.pop_front() {
            for &child in &children[node] {
                depth[child] = depth[node] + 1;
                max_depth = max_depth.max(depth[child]);
                queue.push_back(child);
            }
        }

        // Group by depth, reverse for bottom-up
        let mut levels: Vec<Vec<usize>> = vec![vec![]; (max_depth + 1) as usize];
        for (i, &d) in depth.iter().enumerate() {
            levels[d as usize].push(i);
        }
        levels.reverse();
        levels
    }
}

/// Parsed node: (name, `parent_placeholder`, `branch_length`, `children_indices`)
#[derive(Debug)]
struct ParsedNode {
    name: Option<String>,
    branch_length: f64,
    children: Vec<usize>,
}

fn parse_newick_subtree(
    s: &str,
    start: usize,
) -> BarracudaResult<(Option<usize>, Vec<ParsedNode>)> {
    let mut nodes = Vec::new();
    let mut i = start;
    let s = s.as_bytes();

    fn skip_ws(s: &[u8], i: &mut usize) {
        while *i < s.len() && (s[*i] == b' ' || s[*i] == b'\t') {
            *i += 1;
        }
    }

    fn parse_subtree(
        s: &[u8],
        i: &mut usize,
        nodes: &mut Vec<ParsedNode>,
    ) -> BarracudaResult<Option<usize>> {
        skip_ws(s, i);
        if *i >= s.len() {
            return Ok(None);
        }

        let node_idx = nodes.len();
        let mut branch_length = 0.0;
        let mut name = None;
        let mut children = Vec::new();

        if s[*i] == b'(' {
            *i += 1;
            loop {
                skip_ws(s, i);
                if *i >= s.len() {
                    return Err(BarracudaError::InvalidInput {
                        message: "Unclosed parenthesis in Newick".to_string(),
                    });
                }
                if s[*i] == b')' {
                    *i += 1;
                    break;
                }
                let child = parse_subtree(s, i, nodes)?;
                if let Some(c) = child {
                    children.push(c);
                }
                skip_ws(s, i);
                if *i < s.len() && s[*i] == b',' {
                    *i += 1;
                }
            }
        }

        skip_ws(s, i);
        if *i < s.len() && s[*i] != b',' && s[*i] != b')' && s[*i] != b';' {
            let name_start = *i;
            while *i < s.len() && s[*i] != b':' && s[*i] != b',' && s[*i] != b')' && s[*i] != b';' {
                *i += 1;
            }
            let n = std::str::from_utf8(&s[name_start..*i]).map_err(|_| {
                BarracudaError::InvalidInput {
                    message: "Invalid UTF-8 in Newick label".to_string(),
                }
            })?;
            let n = n.trim();
            if !n.is_empty() {
                name = Some(n.to_string());
            }
        }
        if *i < s.len() && s[*i] == b':' {
            *i += 1;
            let bl_start = *i;
            while *i < s.len()
                && (s[*i] == b'.' || s[*i] == b'-' || s[*i] == b'+' || s[*i].is_ascii_digit())
            {
                *i += 1;
            }
            let bl_str = std::str::from_utf8(&s[bl_start..*i]).map_err(|_| {
                BarracudaError::InvalidInput {
                    message: "Invalid branch length in Newick".to_string(),
                }
            })?;
            branch_length = bl_str
                .parse::<f64>()
                .map_err(|_| BarracudaError::InvalidInput {
                    message: format!("Invalid branch length: {bl_str}"),
                })?;
        }

        nodes.push(ParsedNode {
            name,
            branch_length,
            children,
        });
        Ok(Some(node_idx))
    }

    let root = parse_subtree(s, &mut i, &mut nodes)?;
    Ok((root, nodes))
}

fn build_flat_tree_from_parsed(
    nodes: Vec<ParsedNode>,
    root_idx: usize,
) -> BarracudaResult<FlatTree> {
    let n = nodes.len();
    let mut leaf_names: Vec<(String, usize)> = Vec::new();
    for (idx, node) in nodes.iter().enumerate() {
        if node.children.is_empty() {
            let name = node.name.clone().unwrap_or_else(|| format!("leaf_{idx}"));
            leaf_names.push((name, idx));
        }
    }
    leaf_names.sort_by(|a, b| a.0.cmp(&b.0));
    let n_leaves = leaf_names.len();

    let mut old_to_new: Vec<Option<usize>> = vec![None; n];
    for (new_idx, (_, old_idx)) in leaf_names.iter().enumerate() {
        old_to_new[*old_idx] = Some(new_idx);
    }
    let mut next_internal = n_leaves;
    for idx in (0..n).rev() {
        if old_to_new[idx].is_some() {
            continue;
        }
        let is_root = idx == root_idx;
        if is_root {
            old_to_new[idx] = Some(n - 1);
        } else {
            old_to_new[idx] = Some(next_internal);
            next_internal += 1;
        }
    }

    let mut parent = vec![0i32; n];
    let mut branch_length = vec![0.0; n];

    for (old_idx, node) in nodes.iter().enumerate() {
        let new_idx = old_to_new[old_idx].ok_or_else(|| BarracudaError::InvalidInput {
            message: "Node index mapping failed".to_string(),
        })?;
        branch_length[new_idx] = node.branch_length;

        if node.children.is_empty() {
            if old_idx == root_idx {
                parent[new_idx] = -1;
            } else {
                let child_old = old_idx;
                let parent_old = nodes
                    .iter()
                    .position(|n| n.children.contains(&child_old))
                    .ok_or_else(|| BarracudaError::InvalidInput {
                        message: "Leaf has no parent in Newick".to_string(),
                    })?;
                let parent_new =
                    old_to_new[parent_old].ok_or_else(|| BarracudaError::InvalidInput {
                        message: "Parent mapping failed".to_string(),
                    })?;
                parent[new_idx] = parent_new as i32;
            }
        } else {
            for &child_old in &node.children {
                let child_new =
                    old_to_new[child_old].ok_or_else(|| BarracudaError::InvalidInput {
                        message: "Child mapping failed".to_string(),
                    })?;
                parent[child_new] = new_idx as i32;
            }
        }
    }

    let root_new = old_to_new[root_idx].ok_or_else(|| BarracudaError::InvalidInput {
        message: "Root mapping failed".to_string(),
    })?;
    parent[root_new] = -1;

    let tree = FlatTree {
        parent,
        branch_length,
        n_leaves,
    };
    tree.validate()?;
    Ok(tree)
}

fn build_from_edges(edges: &[(usize, usize, f64)]) -> (Vec<i32>, Vec<f64>, usize) {
    use std::collections::{HashMap, HashSet};
    let mut children_of: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
    let mut all_nodes: HashSet<usize> = HashSet::new();
    for &(p, c, bl) in edges {
        all_nodes.insert(p);
        all_nodes.insert(c);
        children_of.entry(p).or_default().push((c, bl));
    }
    let parents: HashSet<usize> = edges.iter().map(|(p, _, _)| *p).collect();
    let children: HashSet<usize> = edges.iter().map(|(_, c, _)| *c).collect();
    let mut leaves: Vec<usize> = all_nodes
        .iter()
        .filter(|n| !parents.contains(n))
        .copied()
        .collect();
    leaves.sort_unstable();
    let root = all_nodes.iter().find(|n| !children.contains(n)).copied();
    let Some(root) = root else {
        return (vec![-1], vec![0.0], 1);
    };

    let mut node_list: Vec<usize> = leaves.clone();
    let mut internal: Vec<usize> = all_nodes
        .iter()
        .filter(|n| !leaves.contains(n) && **n != root)
        .copied()
        .collect();
    internal.sort_unstable();
    node_list.extend(internal);
    node_list.push(root);

    let old_to_new: HashMap<usize, usize> = node_list
        .iter()
        .enumerate()
        .map(|(new, &old)| (old, new))
        .collect();
    let n_leaves = leaves.len();
    let n_nodes = node_list.len();

    let mut parent = vec![-1i32; n_nodes];
    let mut branch_length = vec![0.0; n_nodes];

    for (old_p, old_c, bl) in edges.iter().copied() {
        let new_p = old_to_new[&old_p];
        let new_c = old_to_new[&old_c];
        parent[new_c] = new_p as i32;
        branch_length[new_c] = bl;
    }
    parent[old_to_new[&root]] = -1;

    (parent, branch_length, n_leaves)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tree() -> FlatTree {
        // Simple tree: 3 leaves (0,1,2), 2 internal (3,4=root)
        //       4
        //      / \
        //     3   2
        //    / \
        //   0   1
        FlatTree {
            parent: vec![3, 3, 4, 4, -1],
            branch_length: vec![0.1, 0.2, 0.3, 0.15, 0.0],
            n_leaves: 3,
        }
    }

    #[test]
    fn validate_ok() {
        assert!(sample_tree().validate().is_ok());
    }

    #[test]
    fn validate_bad_root_count() {
        let mut t = sample_tree();
        t.parent[3] = -1; // two roots
        assert!(t.validate().is_err());
    }

    #[test]
    fn bottom_up_levels_correct() {
        let t = sample_tree();
        let levels = t.bottom_up_levels();
        // Deepest level first (leaves 0,1), then (leaf 2, internal 3), then root 4
        assert_eq!(levels.len(), 3);
        assert!(levels[0].contains(&0));
        assert!(levels[0].contains(&1));
        assert!(levels.last().unwrap().contains(&4));
    }

    #[test]
    fn from_newick_two_leaves() {
        let newick = "(A:0.1,B:0.2);";
        let t = FlatTree::from_newick(newick).expect("parse should succeed");
        assert_eq!(t.n_leaves, 2);
        assert_eq!(t.n_nodes(), 3);
        assert!(t.validate().is_ok());
    }

    #[test]
    fn from_newick_simple() {
        let newick = "((A:0.1,B:0.2):0.3,C:0.4);";
        let t = FlatTree::from_newick(newick).expect("parse should succeed");
        assert_eq!(t.n_leaves, 3);
        assert_eq!(t.n_nodes(), 5);
        assert!(t.validate().is_ok());
        let levels = t.bottom_up_levels();
        assert!(!levels.is_empty());
        let root = t.parent.iter().position(|&p| p < 0).unwrap();
        assert!(levels.last().unwrap().contains(&root));
    }

    #[test]
    fn from_newick_matches_sample() {
        let newick = "((a:0.1,b:0.2):0.15,c:0.3);";
        let t = FlatTree::from_newick(newick).expect("parse should succeed");
        assert_eq!(t.n_leaves, 3);
        assert_eq!(t.n_nodes(), 5);
        assert!(t.validate().is_ok());
        let root = t.parent.iter().position(|&p| p < 0).unwrap();
        assert_eq!(t.parent[root], -1);
    }

    #[test]
    fn from_newick_empty_err() {
        assert!(FlatTree::from_newick("").is_err());
        assert!(FlatTree::from_newick("   ").is_err());
    }

    #[test]
    fn from_newick_no_semicolon_err() {
        assert!(FlatTree::from_newick("(A:1,B:2)").is_err());
    }

    #[test]
    fn from_edges_simple() {
        let edges = vec![(2, 0, 0.1), (2, 1, 0.2), (3, 2, 0.15), (3, 4, 0.3)];
        let t = FlatTree::from_edges(&edges);
        assert_eq!(t.n_leaves, 3);
        assert_eq!(t.n_nodes(), 5);
        assert!(t.validate().is_ok());
        assert_eq!(t.branch_length[0], 0.1);
        assert_eq!(t.branch_length[1], 0.2);
    }

    #[test]
    fn from_edges_empty() {
        let t = FlatTree::from_edges(&[]);
        assert_eq!(t.n_leaves, 1);
        assert_eq!(t.n_nodes(), 1);
        assert_eq!(t.parent[0], -1);
    }

    #[test]
    fn from_edges_single_edge() {
        let edges = vec![(1, 0, 0.5)];
        let t = FlatTree::from_edges(&edges);
        assert_eq!(t.n_leaves, 1);
        assert_eq!(t.n_nodes(), 2);
        assert!(t.validate().is_ok());
    }
}
