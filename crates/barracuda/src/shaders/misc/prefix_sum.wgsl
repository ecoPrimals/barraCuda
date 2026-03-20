// SPDX-License-Identifier: AGPL-3.0-or-later
// prefix_sum.wgsl — GPU exclusive prefix sum (Blelloch scan)
//
// Hierarchical 2-pass scan for arrays of arbitrary length:
//
//   Pass A — local_scan:
//     Each workgroup of 256 threads scans its own 256-element segment using
//     shared memory.  At the end of the scan the exclusive prefix sum for
//     that segment is written to `scan_out[]`.  The workgroup total (sum of
//     all elements in that segment) is written to `wg_sums[workgroup_id.x]`.
//
//   Pass B — add_wg_offsets:
//     A second, single-workgroup pass computes the exclusive prefix sum of
//     `wg_sums[]` and then adds the appropriate cumulative offset back to
//     every element of `scan_out[]`.
//
// After both passes, `scan_out[i]` contains the exclusive prefix sum of the
// original input, which is the scatter index for stream compaction:
//
//   output[scan_out[i]] = input[i]   iff   flags[i] == 1
//
// `total[0]` is set by the caller as `scan_out[N-1] + flags[N-1]`
// (i.e. the total number of selected elements), or can be read from the
// last wg_sums entry after Pass B.

const WG: u32 = 256u;

var<workgroup> shmem: array<u32, 256>;

struct ScanConfig {
    n: u32,          // Total number of elements
    n_groups: u32,   // ceil(n / WG) — number of workgroups for Pass A
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform>             config:   ScanConfig;
@group(0) @binding(1) var<storage, read>       flags_in: array<u32>;  // predicate flags [N]
@group(0) @binding(2) var<storage, read_write> scan_out: array<u32>;  // exclusive scan  [N]
@group(0) @binding(3) var<storage, read_write> wg_sums:  array<u32>;  // workgroup totals [n_groups]

// ── Pass A: intra-workgroup exclusive scan ────────────────────────────────────
@compute @workgroup_size(256)
fn local_scan(
    @builtin(global_invocation_id) global_id:    vec3<u32>,
    @builtin(local_invocation_id)  local_id:     vec3<u32>,
    @builtin(workgroup_id)         workgroup_id: vec3<u32>,
) {
    let lid  = local_id.x;
    let gid  = global_id.x;
    let wgid = workgroup_id.x;

    // Load into shared memory (zero-pad if out of bounds)
    shmem[lid] = select(0u, flags_in[gid], gid < config.n);
    workgroupBarrier();

    // Up-sweep (reduce) phase
    var stride = 1u;
    while (stride < WG) {
        if (lid >= stride && (lid + 1u) % (stride * 2u) == 0u) {
            shmem[lid] = shmem[lid] + shmem[lid - stride];
        }
        stride = stride * 2u;
        workgroupBarrier();
    }

    // Write total and clear root (convert inclusive to exclusive)
    if (lid == WG - 1u) {
        wg_sums[wgid] = shmem[lid];
        shmem[lid] = 0u;
    }
    workgroupBarrier();

    // Down-sweep phase
    stride = WG / 2u;
    while (stride >= 1u) {
        if (lid >= stride && (lid + 1u) % (stride * 2u) == 0u) {
            let left  = shmem[lid - stride];
            let right = shmem[lid];
            shmem[lid - stride] = right;
            shmem[lid]          = left + right;
        }
        stride = stride / 2u;
        workgroupBarrier();
    }

    // Write exclusive scan result
    if (gid < config.n) {
        scan_out[gid] = shmem[lid];
    }
}

// ── Pass B: add workgroup offsets ─────────────────────────────────────────────
//
// A single workgroup computes the exclusive prefix sum of `wg_sums[]` and adds
// the appropriate cumulative offset to every `scan_out` element.
//
// Constraint: works for `n_groups ≤ WG` (i.e. n ≤ WG² = 65,536 elements).
// For n > 65,536 and n ≤ WG³ = 16,777,216, use the two-level path:
//   1. Pass A on input        → scan_out,       wg_sums1
//   2. Pass A on wg_sums1     → wg_sums1_scan,  wg_sums2  (using separate bind group)
//   3. Pass B on wg_sums2/1   → corrects wg_sums1_scan    (single workgroup)
//   4. Pass C (apply_l1_offsets) using wg_sums1_scan → corrects scan_out
//   5. scatter
@compute @workgroup_size(256)
fn add_wg_offsets(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id)  local_id:  vec3<u32>,
) {
    let lid = local_id.x;

    // Load workgroup sums
    shmem[lid] = select(0u, wg_sums[lid], lid < config.n_groups);
    workgroupBarrier();

    // In-place exclusive prefix sum of wg_sums (same Blelloch algorithm)
    var stride = 1u;
    while (stride < WG) {
        if (lid >= stride && (lid + 1u) % (stride * 2u) == 0u) {
            shmem[lid] = shmem[lid] + shmem[lid - stride];
        }
        stride = stride * 2u;
        workgroupBarrier();
    }
    if (lid == WG - 1u) { shmem[lid] = 0u; }
    workgroupBarrier();
    stride = WG / 2u;
    while (stride >= 1u) {
        if (lid >= stride && (lid + 1u) % (stride * 2u) == 0u) {
            let left  = shmem[lid - stride];
            let right = shmem[lid];
            shmem[lid - stride] = right;
            shmem[lid]          = left + right;
        }
        stride = stride / 2u;
        workgroupBarrier();
    }
    workgroupBarrier();

    // Write the workgroup offset back so Pass A's scan_out values become global.
    // Each of the original `n_groups` workgroups owns WG consecutive scan_out entries.
    // We add the cumulative offset for that workgroup to each element.
    let wg_offset = shmem[lid];
    for (var i = 0u; i < WG && (lid * WG + i) < config.n; i = i + 1u) {
        scan_out[lid * WG + i] = scan_out[lid * WG + i] + wg_offset;
    }
}

// ── Pass C: apply level-1 offsets (two-level hierarchy only) ─────────────────
//
// Used when n > WG² (>65,536 elements).  After steps 2 + 3 have produced a
// globally-correct `wg_sums1_scan[]`, this pass adds each workgroup's offset
// to its 256-element segment of `scan_out`.
//
// Binding repurposing (same BGL as Pass A):
//   flags_in  → wg_sums1_scan[] (globally-correct L1 prefix sums, read-only)
//   scan_out  → the L0 exclusive scan to correct (read_write)
//   wg_sums   → unused (bound but not read)
//   config.n  → total element count N (same as Pass A)
//
// Dispatched with n_groups1 workgroups so each workgroup handles its own WG
// elements.  Thread `lid` in workgroup `wgid` processes `scan_out[wgid*WG+lid]`.
@compute @workgroup_size(256)
fn apply_l1_offsets(
    @builtin(global_invocation_id) global_id:   vec3<u32>,
    @builtin(workgroup_id)         workgroup_id: vec3<u32>,
) {
    let offset = flags_in[workgroup_id.x];
    if (global_id.x < config.n) {
        scan_out[global_id.x] = scan_out[global_id.x] + offset;
    }
}
