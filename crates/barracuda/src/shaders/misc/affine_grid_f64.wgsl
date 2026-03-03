// affine_grid_f64.wgsl - Affine Grid Generator (f64 canonical)
//
// Generates sampling grid for spatial transformer networks
// Takes affine transformation matrix and produces coordinate grid
//
// Reference: "Spatial Transformer Networks" by Jaderberg et al. (2015)

struct Params {
    batch_size: u32,
    height: u32,
    width: u32,
    align_corners: u32, // 1 = true, 0 = false
}

@group(0) @binding(0) var<storage, read> theta: array<f64>;        // [B, 2, 3] affine matrices
@group(0) @binding(1) var<storage, read_write> grid: array<f64>;   // [B, H, W, 2] output grid
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.z;
    let h = global_id.y;
    let w = global_id.x;

    if (b >= params.batch_size || h >= params.height || w >= params.width) {
        return;
    }

    // Normalize coordinates to [-1, 1]
    var x: f64;
    var y: f64;

    if (params.align_corners == 1u) {
        // AlignCorners=true: corners at -1 and 1
        x = -1.0 + 2.0 * f64(w) / f64(params.width - 1u);
        y = -1.0 + 2.0 * f64(h) / f64(params.height - 1u);
    } else {
        // AlignCorners=false: pixel centers
        x = -1.0 + 2.0 * (f64(w) + 0.5) / f64(params.width);
        y = -1.0 + 2.0 * (f64(h) + 0.5) / f64(params.height);
    }

    // Load affine transformation matrix for this batch
    let theta_offset = b * 6u;
    let a = theta[theta_offset + 0u];
    let b_coef = theta[theta_offset + 1u];
    let tx = theta[theta_offset + 2u];
    let c = theta[theta_offset + 3u];
    let d = theta[theta_offset + 4u];
    let ty = theta[theta_offset + 5u];

    // Apply affine transformation
    let out_x = a * x + b_coef * y + tx;
    let out_y = c * x + d * y + ty;

    // Write to output grid [B, H, W, 2]
    let grid_offset = b * params.height * params.width * 2u +
                      h * params.width * 2u +
                      w * 2u;

    grid[grid_offset + 0u] = out_x;
    grid[grid_offset + 1u] = out_y;
}
