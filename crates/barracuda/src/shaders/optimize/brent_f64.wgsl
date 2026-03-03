// Batched Brent root-finding at f64 precision
// Finds x such that f(x) = 0 for each batch element independently.
// Each workgroup processes one root-finding problem.
//
// Applications: VG inverse (theta -> h), Green-Ampt ponding time,
//               dose-response IC50, spectral eigenvalue bracketing.
//
// REQUIRES: SHADER_F64
// Provenance: airSpring V035 handoff -> toadStool absorption

struct BrentParams {
    batch_size: u32,
    max_iter: u32,
    operation: u32,     // Which function to solve: 0=VG_inverse, 1=GreenAmpt, 2=custom
    _pad: u32,
    tol: f64,
    aux_a: f64,         // Function-specific parameters
    aux_b: f64,
    aux_c: f64,
    aux_d: f64,
}

@group(0) @binding(0) var<storage, read> bracket_a: array<f64>;
@group(0) @binding(1) var<storage, read> bracket_b: array<f64>;
@group(0) @binding(2) var<storage, read> targets: array<f64>;
@group(0) @binding(3) var<storage, read_write> roots: array<f64>;
@group(0) @binding(4) var<storage, read_write> iterations: array<u32>;
@group(0) @binding(5) var<uniform> params: BrentParams;

// Van Genuchten theta(h) - target = 0
// aux_a = theta_r, aux_b = theta_s, aux_c = alpha, aux_d = n
fn vg_theta_residual(h: f64, tgt: f64) -> f64 {
    let theta_r = params.aux_a;
    let theta_s = params.aux_b;
    let alpha = params.aux_c;
    let n = params.aux_d;
    let one = h - h + f64(1.0);
    let m = one - one / n;
    let ah = alpha * abs(h);
    let denom = pow_f64(one + pow_f64(ah, n), m);
    let theta = theta_r + (theta_s - theta_r) / denom;
    return theta - tgt;
}

// Green-Ampt cumulative infiltration F(t) - target = 0
// aux_a = Ks, aux_b = psi_f * delta_theta
fn green_ampt_residual(f_val: f64, tgt: f64) -> f64 {
    let ks = params.aux_a;
    let psi_dt = params.aux_b;
    let one = f_val - f_val + f64(1.0);
    let t_from_f = f_val / ks - psi_dt / ks * log_f64(one + f_val / psi_dt);
    return t_from_f - tgt;
}

fn eval_function(x: f64, tgt: f64) -> f64 {
    switch (params.operation) {
        case 0u: { return vg_theta_residual(x, tgt); }
        case 1u: { return green_ampt_residual(x, tgt); }
        default: {
            // Custom: simple polynomial x^2 - target (placeholder)
            return x * x - tgt;
        }
    }
}

@compute @workgroup_size(1)
fn brent_solve(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let idx = wg_id.x;
    if (idx >= params.batch_size) { return; }

    var a = bracket_a[idx];
    var b = bracket_b[idx];
    let tgt = targets[idx];
    let tol = params.tol;
    let max_it = params.max_iter;

    var fa = eval_function(a, tgt);
    var fb = eval_function(b, tgt);

    var c = a;
    var fc = fa;
    var d = b - a;
    var e = d;
    let zero = a - a;

    var iter = 0u;
    for (var i = 0u; i < max_it; i = i + 1u) {
        if (abs(fc) < abs(fb)) {
            a = b; b = c; c = a;
            fa = fb; fb = fc; fc = fa;
        }

        let tol1 = (zero + 2.0) * (zero + 1.1920929e-7) * abs(b) + (zero + 0.5) * tol;
        let m = (zero + 0.5) * (c - b);

        if (abs(m) <= tol1 || fb == zero) {
            iter = i;
            break;
        }

        if (abs(e) >= tol1 && abs(fa) > abs(fb)) {
            let s_val = fb / fa;
            var p: f64;
            var q: f64;
            if (a == c) {
                p = (zero + 2.0) * m * s_val;
                q = (zero + 1.0) - s_val;
            } else {
                let r = fb / fc;
                p = s_val * ((zero + 2.0) * m * (r - (zero + 1.0)) - (b - a) * (r - (zero + 1.0)));
                q = (fa / fc - (zero + 1.0)) * (fa / fb - (zero + 1.0)) * (fb / fc - (zero + 1.0));
            }
            if (p > zero) { q = -q; } else { p = -p; }

            if ((zero + 2.0) * p < (zero + 3.0) * m * q - abs(tol1 * q) && (zero + 2.0) * p < abs(e * q)) {
                e = d;
                d = p / q;
            } else {
                d = m;
                e = m;
            }
        } else {
            d = m;
            e = m;
        }

        a = b;
        fa = fb;

        if (abs(d) > tol1) {
            b = b + d;
        } else {
            if (m > zero) { b = b + tol1; } else { b = b - tol1; }
        }

        fb = eval_function(b, tgt);

        if ((fb > zero && fc > zero) || (fb < zero && fc < zero)) {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }

        iter = i + 1u;
    }

    roots[idx] = b;
    iterations[idx] = iter;
}
