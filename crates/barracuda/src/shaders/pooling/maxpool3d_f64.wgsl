// SPDX-License-Identifier: AGPL-3.0-only
// maxpool3d.wgsl - 3D Max Pooling (f64 canonical)
//
// Max pooling for 3D data (video, volumetric medical imaging)
// Commonly used in 3D CNNs for action recognition and medical imaging

struct Params {
    batch_size: u32,
    channels: u32,
    in_depth: u32,
    in_height: u32,
    in_width: u32,
    out_depth: u32,
    out_height: u32,
    out_width: u32,
    kernel_d: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_d: u32,
    stride_h: u32,
    stride_w: u32,
    pad_d: u32,
    pad_h: u32,
    pad_w: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let od = global_id.z;
    let oh = global_id.y;
    let ow = global_id.x;

    for (var b: u32 = 0u; b < params.batch_size; b = b + 1u) {
        for (var c: u32 = 0u; c < params.channels; c = c + 1u) {
            if (od >= params.out_depth || oh >= params.out_height || ow >= params.out_width) {
                continue;
            }

            var max_val: f64 = -1e308;

            for (var kd: u32 = 0u; kd < params.kernel_d; kd = kd + 1u) {
                for (var kh: u32 = 0u; kh < params.kernel_h; kh = kh + 1u) {
                    for (var kw: u32 = 0u; kw < params.kernel_w; kw = kw + 1u) {
                        let id_raw = i32(od * params.stride_d) - i32(params.pad_d) + i32(kd);
                        let ih_raw = i32(oh * params.stride_h) - i32(params.pad_h) + i32(kh);
                        let iw_raw = i32(ow * params.stride_w) - i32(params.pad_w) + i32(kw);

                        if (id_raw >= 0 && id_raw < i32(params.in_depth) &&
                            ih_raw >= 0 && ih_raw < i32(params.in_height) &&
                            iw_raw >= 0 && iw_raw < i32(params.in_width)) {

                            let id = u32(id_raw);
                            let ih = u32(ih_raw);
                            let iw = u32(iw_raw);

                            let in_idx = b * params.channels * params.in_depth * params.in_height * params.in_width +
                                        c * params.in_depth * params.in_height * params.in_width +
                                        id * params.in_height * params.in_width +
                                        ih * params.in_width +
                                        iw;

                            max_val = max(max_val, input[in_idx]);
                        }
                    }
                }
            }

            let out_idx = b * params.channels * params.out_depth * params.out_height * params.out_width +
                          c * params.out_depth * params.out_height * params.out_width +
                          od * params.out_height * params.out_width +
                          oh * params.out_width +
                          ow;

            output[out_idx] = max_val;
        }
    }
}
