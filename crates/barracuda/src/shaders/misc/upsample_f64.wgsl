// SPDX-License-Identifier: AGPL-3.0-only
// Upsample - Upsampling (nearest neighbor or bilinear) (f64 canonical)
// Increases spatial resolution of input tensor
//
// Supports:
// - Nearest neighbor: mode=0
// - Bilinear interpolation: mode=1
//
// Algorithm:
// For each output pixel, compute corresponding input coordinates and interpolate

struct Params {
    batch_size: u32,
    channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
    mode: u32,         // 0 = nearest, 1 = bilinear
    align_corners: u32, // 0 or 1
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;  // [B, C, H_in, W_in]
@group(0) @binding(2) var<storage, read_write> output: array<f64>; // [B, C, H_out, W_out]

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.z / params.channels;
    let c = global_id.z % params.channels;
    let oh = global_id.y;
    let ow = global_id.x;
    
    if (b >= params.batch_size || c >= params.channels || 
        oh >= params.out_height || ow >= params.out_width) {
        return;
    }

    // Compute input coordinates
    var scale_h: f64;
    var scale_w: f64;
    
    if (params.align_corners != 0u) {
        scale_h = f64(params.in_height - 1u) / f64(params.out_height - 1u);
        scale_w = f64(params.in_width - 1u) / f64(params.out_width - 1u);
    } else {
        scale_h = f64(params.in_height) / f64(params.out_height);
        scale_w = f64(params.in_width) / f64(params.out_width);
    }
    
    let fh = f64(oh) * scale_h;
    let fw = f64(ow) * scale_w;
    
    var value: f64;
    
    if (params.mode == 0u) {
        // Nearest neighbor
        let ih = u32(floor(fh + 0.5));
        let iw = u32(floor(fw + 0.5));
        let ih_clamped = clamp(ih, 0u, params.in_height - 1u);
        let iw_clamped = clamp(iw, 0u, params.in_width - 1u);
        
        let in_idx = ((b * params.channels + c) * params.in_height + ih_clamped) * params.in_width + iw_clamped;
        value = input[in_idx];
        
    } else {
        // Bilinear interpolation
        let ih0 = u32(floor(fh));
        let iw0 = u32(floor(fw));
        let ih1 = min(ih0 + 1u, params.in_height - 1u);
        let iw1 = min(iw0 + 1u, params.in_width - 1u);
        
        let fh_frac = fh - f64(ih0);
        let fw_frac = fw - f64(iw0);
        
        // Fetch 4 neighbors
        let idx00 = ((b * params.channels + c) * params.in_height + ih0) * params.in_width + iw0;
        let idx01 = ((b * params.channels + c) * params.in_height + ih0) * params.in_width + iw1;
        let idx10 = ((b * params.channels + c) * params.in_height + ih1) * params.in_width + iw0;
        let idx11 = ((b * params.channels + c) * params.in_height + ih1) * params.in_width + iw1;
        
        let v00 = input[idx00];
        let v01 = input[idx01];
        let v10 = input[idx10];
        let v11 = input[idx11];
        
        // Bilinear interpolation
        let v0 = v00 * (1.0 - fw_frac) + v01 * fw_frac;
        let v1 = v10 * (1.0 - fw_frac) + v11 * fw_frac;
        value = v0 * (1.0 - fh_frac) + v1 * fh_frac;
    }
    
    let out_idx = ((b * params.channels + c) * params.out_height + oh) * params.out_width + ow;
    output[out_idx] = value;
}
