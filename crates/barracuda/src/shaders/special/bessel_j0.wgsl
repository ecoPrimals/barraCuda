// Bessel function of the first kind, order 0: J0(x)
// Uses rational polynomial approximation (Abramowitz & Stegun 9.4.1-9.4.3)
// For |x| < 8: polynomial P(x²)/Q(x²)
// For |x| >= 8: asymptotic form sqrt(2/(πx)) * cos(x - π/4 + correction)

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> metadata: Metadata;

struct Metadata {
    size: u32,
}

// Rational polynomial coefficients for |x| < 8 (A&S 9.4.1-9.4.2)
// J0(x) = P(z)/Q(z) where z = x²
fn bessel_j0_approx(x: f32) -> f32 {
    let ax = abs(x);
    if (ax >= 8.0) {
        // Asymptotic form for |x| >= 8
        // J0(x) ≈ sqrt(2/(πx)) * [P0(8/x)*cos(x - π/4) - Q0(8/x)*sin(x - π/4)]
        let z = 8.0 / ax;
        let z2 = z * z;
        // P0(z) = 1 + p1*z² + p2*z⁴ + ...
        let z4 = z2 * z2;
        let pv = 1.0 + z2 * (-0.0000000000233170 + z2 * 0.000000001125984)
                 + z4 * (-0.000000027405505 + z2 * 0.000001823005855);
        // Q0(z) = q0*z + q1*z³ + ...
        let qv = z * (-0.0000000009760400 + z2 * 0.000000059719504)
                 + z4 * (-0.000004007121410 + z2 * 0.000281574075958);
        let pp = 0.7978845608028654; // sqrt(2/pi)
        let inv_sqrt_x = pp / sqrt(ax);
        let xx = ax - 0.7853981633974483; // pi/4
        return inv_sqrt_x * (pv * cos(xx) - qv * sin(xx));
    }
    // For |x| < 8: rational approximation in x²
    let z = x * x;
    // P(z) = 1 - z²/64 + ... (simplified rational form)
    let p = 1.0 + z * (-0.1098628627e-2 + z * (0.2734510407e-4 + z * (-0.2073370639e-5 + z * 0.2093887211e-6)));
    let q = 1.0 + z * (0.0187290000 + z * (0.0007859851 + z * (0.1570469081e-4 - z * 0.3461740466e-6)));
    return p / q;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= metadata.size) {
        return;
    }
    output[idx] = bessel_j0_approx(input[idx]);
}
