// SPDX-License-Identifier: AGPL-3.0-or-later
// Fog of war — GPU-parallel visibility state computation on a 2D grid.
//
// Universal spatial primitive: exploration tracking, progressive revelation,
// sensor coverage simulation, mapping/cartography.
//
// Per-tile visibility: 0 = unknown, 1 = explored (previously seen), 2 = visible.
// Uses radial distance² check from viewer position.
//
// Absorbed from ludoSpring (Sprint 72) and generalized for cross-spring use.
//
// Bindings:
//   0: params    (uniform) [viewer_x, viewer_y, grid_w, grid_h, sight_radius², _pad]
//   1: terrain   [f32 per tile — >= 0.9 = occluder/wall]
//   2: prev_vis  [u32 per tile — previous visibility state]
//   3: out_vis   [u32 per tile — new visibility state]

struct Params {
    viewer_x: f32,
    viewer_y: f32,
    grid_w: u32,
    grid_h: u32,
    sight_radius_sq: f32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> terrain: array<f32>;
@group(0) @binding(2) var<storage, read> prev_vis: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_vis: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.grid_w * params.grid_h;
    if idx >= total {
        return;
    }

    let tile_x = f32(idx % params.grid_w) + 0.5;
    let tile_y = f32(idx / params.grid_w) + 0.5;
    let dx = tile_x - params.viewer_x;
    let dy = tile_y - params.viewer_y;
    let dist_sq = dx * dx + dy * dy;

    if dist_sq <= params.sight_radius_sq {
        out_vis[idx] = 2u;
    } else if prev_vis[idx] >= 1u {
        out_vis[idx] = 1u;
    } else {
        out_vis[idx] = 0u;
    }
}
