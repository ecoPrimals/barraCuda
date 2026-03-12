// SPDX-License-Identifier: AGPL-3.0-only
// GPU-Resident Cell-List: Particle Scatter (Pass 3 of 3)
//
// Writes each particle's sorted index into the cell list using the prefix-sum
// offsets computed by pass 2.  Uses atomicAdd on per-cell write cursors so all
// N threads scatter concurrently without conflicts.
//
// Pass order (hotSpring feedback, Feb 19 2026):
//   1. atomic_cell_bin   — particle → cell_id + cell_counts[]
//   2. prefix_sum        — cell_counts[] → cell_start[] (exclusive prefix sum)
//   3. cell_list_scatter (this shader) — fill sorted_indices[]
//
// After this pass:
//   sorted_indices[cell_start[c] .. cell_start[c] + cell_count[c]]
//     = indices of particles in cell c (in arbitrary order within cell)
//
// This is sufficient for the force kernel: iterate neighbour cells, walk their
// particle ranges, compute pairwise interactions.  The resulting cell list is
// 100% GPU-resident — no CPU readback or re-upload between rebuilds.
//
// Memory layout:
//   cell_ids:       [N]  u32 — output of pass 1 (cell for particle i)
//   cell_start:     [Nc] u32 — output of pass 2 (exclusive prefix sum of counts)
//   write_cursors:  [Nc] u32 — zero-initialised scratch; holds per-cell write position
//   sorted_indices: [N]  u32 — output of this pass (particle indices sorted by cell)
//
// Dispatch: (ceil(N / 64), 1, 1)

struct CellScatterParams {
    n_particles: u32,
    n_cells:     u32,
    _pad0:       u32,
    _pad1:       u32,
}

@group(0) @binding(0) var<uniform>             params:         CellScatterParams;
@group(0) @binding(1) var<storage, read>       cell_ids:       array<u32>; // [N] — from pass 1
@group(0) @binding(2) var<storage, read>       cell_start:     array<u32>; // [Nc] — from pass 2
@group(0) @binding(3) var<storage, read_write>  write_cursors:  array<atomic<u32>>; // [Nc] zero-init
@group(0) @binding(4) var<storage, read_write>  sorted_indices: array<u32>; // [N] — output

@compute @workgroup_size(64)
fn cell_list_scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let i = gid.x;
    if (i >= params.n_particles) { return; }

    let c   = cell_ids[i];                     // which cell does particle i belong to?
    let pos = atomicAdd(&write_cursors[c], 1u); // claim next slot in this cell
    sorted_indices[cell_start[c] + pos] = i;   // write particle index into sorted array
}
