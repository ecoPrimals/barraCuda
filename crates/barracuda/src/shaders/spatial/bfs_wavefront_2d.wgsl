// SPDX-License-Identifier: AGPL-3.0-or-later
// BFS wavefront — single-step breadth-first expansion on a 2D grid.
//
// Universal spatial primitive: pathfinding, influence maps, flood-fill,
// reachability analysis, Voronoi diagrams.
//
// Each dispatch expands the frontier by one ring. Host loops until
// frontier_count == 0 (full exploration) or a target is found.
//
// Absorbed from ludoSpring (Sprint 72) and generalized for cross-spring use.
//
// Bindings:
//   0: params       [grid_w, grid_h, current_dist, _pad]
//   1: passability  [f32 per tile — values >= threshold are impassable]
//   2: dist_map     [u32 per tile — 0xFFFFFFFF = unvisited, else distance]
//   3: frontier     [atomic u32 — tiles expanded this step]

struct Params {
    grid_w: u32,
    grid_h: u32,
    current_dist: u32,
    impassable_threshold: u32,  // reinterpret as f32 bits for comparison
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> passability: array<f32>;
@group(0) @binding(2) var<storage, read_write> dist_map: array<u32>;
@group(0) @binding(3) var<storage, read_write> frontier: atomic<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.grid_w * params.grid_h;
    if idx >= total {
        return;
    }

    if dist_map[idx] != params.current_dist {
        return;
    }

    let x = idx % params.grid_w;
    let y = idx / params.grid_w;
    let next_dist = params.current_dist + 1u;

    let offsets = array<vec2<i32>, 4>(
        vec2<i32>(0, -1),
        vec2<i32>(0, 1),
        vec2<i32>(1, 0),
        vec2<i32>(-1, 0)
    );

    for (var i = 0u; i < 4u; i = i + 1u) {
        let nx = i32(x) + offsets[i].x;
        let ny = i32(y) + offsets[i].y;

        if nx < 0 || ny < 0 || nx >= i32(params.grid_w) || ny >= i32(params.grid_h) {
            continue;
        }

        let nidx = u32(ny) * params.grid_w + u32(nx);

        if passability[nidx] >= 0.9 {
            continue;
        }

        if dist_map[nidx] == 0xFFFFFFFFu {
            dist_map[nidx] = next_dist;
            atomicAdd(&frontier, 1u);
        }
    }
}
