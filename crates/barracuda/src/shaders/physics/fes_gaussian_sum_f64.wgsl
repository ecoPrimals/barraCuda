// SPDX-License-Identifier: AGPL-3.0-or-later
// FES Gaussian Summation (f64) — GPU-Parallel
//
// Reconstructs a 2D Free Energy Surface from metadynamics HILLS data.
// Each workgroup thread computes one grid point by summing all Gaussians:
//   bias(x,y) = Σ_g h_g · exp(-(x-cx_g)²/(2σx²)) · exp(-(y-cy_g)²/(2σy²))
//   FES(x,y) = -bias(x,y) + min_shift
//
// Layout:
//   hills_data: [cx_0, cy_0, sx_0, sy_0, h_0, cx_1, cy_1, ...]  (5 f64 per Gaussian)
//   grid_out:   [fes_00, fes_10, fes_20, ...]  (nbins_x * nbins_y output)

struct FesParams {
    n_gaussians: u32,
    nbins_x: u32,
    nbins_y: u32,
    _pad: u32,
    grid_min_x: f64,
    grid_max_x: f64,
    grid_min_y: f64,
    grid_max_y: f64,
}

@group(0) @binding(0) var<storage, read> hills_data: array<f64>;
@group(0) @binding(1) var<storage, read_write> grid_out: array<f64>;
@group(0) @binding(2) var<uniform> config: FesParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total_bins = config.nbins_x * config.nbins_y;
    if (idx >= total_bins) {
        return;
    }

    let ix = idx % config.nbins_x;
    let iy = idx / config.nbins_x;

    let x = config.grid_min_x + (config.grid_max_x - config.grid_min_x) * f64(ix) / f64(config.nbins_x - 1u);
    let y = config.grid_min_y + (config.grid_max_y - config.grid_min_y) * f64(iy) / f64(config.nbins_y - 1u);

    var bias: f64 = 0.0;
    for (var g: u32 = 0u; g < config.n_gaussians; g = g + 1u) {
        let base = g * 5u;
        let cx = hills_data[base + 0u];
        let cy = hills_data[base + 1u];
        let sx = hills_data[base + 2u];
        let sy = hills_data[base + 3u];
        let h  = hills_data[base + 4u];

        let dx = x - cx;
        let dy = y - cy;
        let inv_2sx2 = 1.0 / (2.0 * sx * sx);
        let inv_2sy2 = 1.0 / (2.0 * sy * sy);

        bias = bias + h * exp(-dx * dx * inv_2sx2) * exp(-dy * dy * inv_2sy2);
    }

    grid_out[idx] = -bias;
}
