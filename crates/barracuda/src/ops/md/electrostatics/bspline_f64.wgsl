// bspline_f64.wgsl — Cardinal B-spline M_n(u) evaluation for PPPM charge spreading
//
// **Math**: Cardinal B-spline of order n: M_n(u) defined on [0,n], zero elsewhere.
// Explicit formulas for order 4, 5, 6 (most common in PPPM):
//
//   M_4(u): piecewise cubic on [0,4]
//   M_5(u): piecewise quartic on [0,5]
//   M_6(u): piecewise quintic on [0,6]
//
// Recursive: M_n(u) = (u/(n-1))*M_{n-1}(u) + ((n-u)/(n-1))*M_{n-1}(u-1)
// Direct formulas used for order 4,5,6 for efficiency.
//
// **Precision**: f64 via bitcast<f64>(vec2<u32>)
// **Workgroup**: @compute @workgroup_size(256)
//
// Bindings:
//   0: u_vals   array<vec2<u32>>  read       — fractional coordinates u (f64)
//   1: output   array<vec2<u32>>  read_write — B-spline values M_order(u)
//
// Params: { n: u32, order: u32 } — order typically 4, 5, or 6
//
// Reference: Hockney & Eastwood "Computer Simulation Using Particles"; Schoenberg

@group(0) @binding(0) var<storage, read> u_vals: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    n: u32,
    order: u32,
}

// M_2(u) = u for u in [0,1], 2-u for u in [1,2], else 0
fn M2(u: f64) -> f64 {
    let z = 0.0;
    if (u <= z || u >= 2.0) { return z; }
    if (u < 1.0) { return u; }
    return 2.0 - u;
}

// M_3(u) — quadratic
fn M3(u: f64) -> f64 {
    let z = 0.0;
    if (u <= z || u >= 3.0) { return z; }
    if (u < 1.0) { return 0.5 * u * u; }
    if (u < 2.0) { return 0.5 * (-2.0*u*u + 6.0*u - 3.0); }
    return 0.5 * (u - 3.0) * (u - 3.0);
}

// M_4(u) — cubic cardinal B-spline (order 4)
fn M4(u: f64) -> f64 {
    let z = 0.0;
    if (u <= z || u >= 4.0) { return z; }
    let u2 = u * u;
    let u3 = u2 * u;
    if (u < 1.0) { return u3 / 6.0; }
    if (u < 2.0) { return ( -3.0*u3 + 12.0*u2 - 12.0*u + 4.0) / 6.0; }
    if (u < 3.0) { return ( 3.0*u3 - 24.0*u2 + 60.0*u - 44.0) / 6.0; }
    return (4.0 - u) * (4.0 - u) * (4.0 - u) / 6.0;
}

// M_5(u) — quartic
fn M5(u: f64) -> f64 {
    let z = 0.0;
    if (u <= z || u >= 5.0) { return z; }
    let u2 = u * u;
    let u3 = u2 * u;
    let u4 = u3 * u;
    if (u < 1.0) { return u4 / 24.0; }
    if (u < 2.0) { return (u4 - 5.0*u3 + 10.0*u2 - 10.0*u + 4.0) / 24.0; }
    if (u < 3.0) { return (-4.0*u4 + 40.0*u3 - 140.0*u2 + 220.0*u - 124.0) / 24.0; }
    if (u < 4.0) { return (6.0*u4 - 90.0*u3 + 510.0*u2 - 1290.0*u + 1204.0) / 24.0; }
    let v = 5.0 - u;
    return v * v * v * v / 24.0;
}

// M_6(u) — quintic
fn M6(u: f64) -> f64 {
    let z = 0.0;
    if (u <= z || u >= 6.0) { return z; }
    let u2 = u * u;
    let u3 = u2 * u;
    let u4 = u3 * u;
    let u5 = u4 * u;
    if (u < 1.0) { return u5 / 120.0; }
    if (u < 2.0) { return (u5 - 6.0*u4 + 15.0*u3 - 20.0*u2 + 15.0*u - 4.0) / 120.0; }
    if (u < 3.0) { return (-5.0*u5 + 60.0*u4 - 315.0*u3 + 900.0*u2 - 1245.0*u + 624.0) / 120.0; }
    if (u < 4.0) { return (10.0*u5 - 180.0*u4 + 1395.0*u3 - 5940.0*u2 + 12915.0*u - 10704.0) / 120.0; }
    if (u < 5.0) { return (-10.0*u5 + 240.0*u4 - 2385.0*u3 + 12600.0*u2 - 35235.0*u + 38364.0) / 120.0; }
    let v = 6.0 - u;
    return v * v * v * v * v / 120.0;
}

fn bspline_value(order: u32, u: f64) -> f64 {
    if (order == 2u) { return M2(u); }
    if (order == 3u) { return M3(u); }
    if (order == 4u) { return M4(u); }
    if (order == 5u) { return M5(u); }
    if (order == 6u) { return M6(u); }
    // Default: order 4 (most common)
    return M4(u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = params.n;
    let order = params.order;
    if (i >= n) {
        return;
    }

    let u = bitcast<f64>(u_vals[i]);
    let val = bspline_value(order, u);
    output[i] = bitcast<vec2<u32>>(val);
}
