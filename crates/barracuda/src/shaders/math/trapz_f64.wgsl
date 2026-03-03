// trapz_f64.wgsl — Parallel trapezoidal integration for f64
//
// **Math**: Trapezoidal rule approximates ∫_a^b f(x) dx ≈ Σ_i 0.5*(y[i]+y[i+1])*(x[i+1]-x[i])
// where y[i] = f(x[i]). This kernel computes partial sums for parallel reduction.
//
// **Algorithm**: Each thread i computes one trapezoidal contribution:
//   - Thread i < n-1: partial_sums[i] = 0.5 * (y[i] + y[i+1]) * (x[i+1] - x[i])
//   - Thread i = n-1: partial_sums[i] = 0.0
// A subsequent reduction kernel sums partial_sums to get the integral.
//
// **Precision**: f64 via bitcast<f64>(vec2<u32>)
// **Workgroup**: @compute @workgroup_size(256)
//
// Bindings:
//   0: y values    array<vec2<u32>>  read
//   1: x values    array<vec2<u32>>  read
//   2: partial_sums array<vec2<u32>> read_write
//
// Params: { n: u32 } — number of data points
//
// Applications: Numerical integration, cumulative distribution, area under curves.
// Reference: Press et al. "Numerical Recipes", trapezoidal rule (Eq. 4.1.11)

@group(0) @binding(0) var<storage, read> y_vals: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> x_vals: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> partial_sums: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    n: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = params.n;
    if (i >= n) {
        return;
    }

    if (i == n - 1u) {
        // Last thread outputs 0.0 (f64 zero bitcast)
        partial_sums[i] = vec2<u32>(0u, 0u);
        return;
    }

    let yi = bitcast<f64>(y_vals[i]);
    let yi1 = bitcast<f64>(y_vals[i + 1u]);
    let xi = bitcast<f64>(x_vals[i]);
    let xi1 = bitcast<f64>(x_vals[i + 1u]);

    // 0.5 * (y[i] + y[i+1]) * (x[i+1] - x[i])
    let half = xi - xi + 0.5;
    let trapz = half * (yi + yi1) * (xi1 - xi);
    partial_sums[i] = bitcast<vec2<u32>>(trapz);
}
