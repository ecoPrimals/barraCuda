// SPDX-License-Identifier: AGPL-3.0-only
//! Neighbor Search for Molecular Dynamics
//!
//! O(N) neighbor search algorithms for efficient force computation.
//!
//! **Cell List**:
//! - Divides simulation box into cells of size ≥ cutoff
//! - Each particle only interacts with 27 neighboring cells
//! - Reduces O(N²) to O(N) for large N
//!
//! **Architecture**:
//! - CPU builds cell list ([`CellList`]): assigns particles → sorts → computes offsets
//! - GPU-resident cell list ([`CellListGpu`]): 3-pass atomic pipeline, no CPU round-trip
//! - Verlet neighbor list ([`VerletList`]): explicit per-particle lists with skin radius
//! - Force kernel uses cell metadata: iterates only over 27 neighbors per particle
//!
//! **Usage Pattern**:
//! ```text
//! let cell_list = CellList::new(rc, box_side);
//!
//! for step in 0..n_steps {
//!     // Rebuild if needed (every step for correctness, or use Verlet skin)
//!     cell_list.rebuild(&positions, n);
//!     
//!     // Sort particle data for coalesced GPU access
//!     let sorted_pos = cell_list.sort_array(&positions, 3);
//!     let sorted_vel = cell_list.sort_array(&velocities, 3);
//!     
//!     // Upload sorted data + cell metadata to GPU
//!     // Force kernel uses cell_start/cell_count for 27-neighbor iteration
//! }
//! ```
//!
//! **hotSpring Integration** (Feb 2026):
//! - CPU-managed cell list with GPU 27-neighbor force kernel
//! - Rebuild every step (`rebuild_interval=1`) for guaranteed correctness
//! - <5% CPU overhead at N=10k, GPU speedup dominates
//!
//! **Deep Debt Compliance**:
//! - ✅ Zero unsafe code
//! - ✅ Pure Rust (no external dependencies)
//! - ✅ Modular design (works with any force kernel)

mod cell_list;
mod cell_list_gpu;
mod verlet_list;

pub use cell_list::CellList;
pub use cell_list_gpu::CellListGpu;
pub use verlet_list::VerletList;
