// BFGS Inverse Hessian Update - f64 canonical
//
// Updates the inverse Hessian approximation H⁻¹ using the BFGS formula:
//   H⁻¹_new = (I - ρsy^T)H⁻¹(I - ρys^T) + ρss^T
//
// Expanded form:
//   H⁻¹_new[i,j] = H⁻¹[i,j] - ρ(s[i]·Hy[j] + Hy[i]·s[j]) + factor·s[i]·s[j]

struct Params {
    n: u32,
    rho: f64,
    yHy: f64,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> s: array<f64>;
@group(0) @binding(2) var<storage, read> Hy: array<f64>;
@group(0) @binding(3) var<storage, read_write> H_inv: array<f64>;

@compute @workgroup_size(16, 16, 1)
fn bfgs_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.y;
    let j = global_id.x;
    let n = params.n;

    if (i >= n || j >= n) {
        return;
    }

    let idx = i * n + j;
    let rho = params.rho;

    let factor = rho * (1.0 + rho * params.yHy);

    H_inv[idx] = H_inv[idx]
                 - rho * (s[i] * Hy[j] + Hy[i] * s[j])
                 + factor * s[i] * s[j];
}

@group(0) @binding(0) var<uniform> dot_params: Params;
@group(0) @binding(1) var<storage, read> dot_a: array<f64>;
@group(0) @binding(2) var<storage, read> dot_b: array<f64>;
@group(0) @binding(3) var<storage, read_write> dot_result: array<f64>;

var<workgroup> partial_sums: array<f64, 256>;

@compute @workgroup_size(256, 1, 1)
fn dot_product(@builtin(local_invocation_id) local_id: vec3<u32>,
               @builtin(global_invocation_id) global_id: vec3<u32>,
               @builtin(workgroup_id) wg_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;
    let n = dot_params.n;

    var sum: f64 = 0.0;
    var i = gid;
    while (i < n) {
        sum = sum + dot_a[i] * dot_b[i];
        i = i + 256u * 256u;
    }

    partial_sums[tid] = sum;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            partial_sums[tid] = partial_sums[tid] + partial_sums[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        dot_result[wg_id.x] = partial_sums[0];
    }
}

@compute @workgroup_size(256, 1, 1)
fn mat_vec_mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = params.n;

    if (i >= n) {
        return;
    }

    var sum: f64 = 0.0;
    for (var j = 0u; j < n; j = j + 1u) {
        sum = sum + H_inv[i * n + j] * s[j];
    }

    Hy[i] = sum;
}

struct CombinedParams {
    n: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> combined_params: CombinedParams;
@group(0) @binding(1) var<storage, read> y_vec: array<f64>;
@group(0) @binding(2) var<storage, read> H_mat: array<f64>;
@group(0) @binding(3) var<storage, read_write> Hy_out: array<f64>;
@group(0) @binding(4) var<storage, read_write> yHy_out: array<f64>;

var<workgroup> Hy_shared: array<f64, 256>;

@compute @workgroup_size(256, 1, 1)
fn compute_Hy_and_yHy(@builtin(local_invocation_id) local_id: vec3<u32>,
                       @builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = local_id.x;
    let i = global_id.x;
    let n = combined_params.n;

    var sum: f64 = 0.0;
    if (i < n) {
        for (var j = 0u; j < n; j = j + 1u) {
            sum = sum + H_mat[i * n + j] * y_vec[j];
        }
        Hy_out[i] = sum;
        Hy_shared[tid] = y_vec[i] * sum;
    } else {
        Hy_shared[tid] = 0.0;
    }
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride && tid + stride < n) {
            Hy_shared[tid] = Hy_shared[tid] + Hy_shared[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        yHy_out[0] = Hy_shared[0];
    }
}
