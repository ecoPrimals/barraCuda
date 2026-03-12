// SPDX-License-Identifier: AGPL-3.0-only
// Legendre polynomials Pₙ(x) and associated Legendre functions Pₙᵐ(x)
// Uses three-term recurrence relations:
//   P₀(x) = 1
//   P₁(x) = x
//   (n+1)Pₙ₊₁(x) = (2n+1)x·Pₙ(x) - n·Pₙ₋₁(x)
//
// Associated Legendre Pₙᵐ(x) uses Condon-Shortley phase convention.
//
// Applications: angular momentum, spherical harmonics, multipole expansion
// Reference: Abramowitz & Stegun §8.5, DLMF §14

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    size: u32,
    n: u32,      // Degree (0, 1, 2, ...)
    m: u32,      // Order for associated Legendre (0 for regular Pₙ)
    is_assoc: u32, // 0 = regular Legendre Pₙ(x), 1 = associated Pₙᵐ(x)
}

// Double factorial (2m-1)!! = 1*3*5*...*(2m-1)
fn double_factorial(m: u32) -> f32 {
    if (m == 0u) {
        return 1.0;
    }
    var r = 1.0;
    for (var k = 1u; k <= m; k = k + 1u) {
        r = r * f32(2u * k - 1u);
    }
    return r;
}

// Regular Legendre polynomial Pₙ(x)
fn legendre(n: u32, x: f32) -> f32 {
    if (n == 0u) {
        return 1.0;
    }
    if (n == 1u) {
        return x;
    }

    var p_prev: f32 = 1.0;  // P₀
    var p_curr: f32 = x;    // P₁

    for (var k = 1u; k < n; k = k + 1u) {
        let k_f32 = f32(k);
        let p_next = ((2.0 * k_f32 + 1.0) * x * p_curr - k_f32 * p_prev) / (k_f32 + 1.0);
        p_prev = p_curr;
        p_curr = p_next;
    }

    return p_curr;
}

// Associated Legendre function Pₙᵐ(x) with Condon-Shortley phase
fn assoc_legendre(n: u32, m: u32, x: f32) -> f32 {
    if (m > n) {
        return 0.0;
    }
    if (m == 0u) {
        return legendre(n, x);
    }

    // Compute (1 - x²)^(m/2)
    let sin_sq = 1.0 - x * x;
    if (sin_sq <= 0.0) {
        return 0.0;  // x = ±1, Pₙᵐ = 0 for m > 0
    }
    let sin_theta_m = pow(sqrt(sin_sq), f32(m));

    // P_m^m = (-1)^m (2m-1)!! (1-x²)^(m/2) [Condon-Shortley]
    var pmm = double_factorial(m) * sin_theta_m;
    if (m % 2u == 1u) {
        pmm = -pmm;
    }

    if (n == m) {
        return pmm;
    }

    // P_{m+1}^m = x(2m+1) P_m^m
    var pm1m = x * f32(2u * m + 1u) * pmm;

    if (n == m + 1u) {
        return pm1m;
    }

    // Upward recurrence for l = m+2, ..., n
    var pl_minus2 = pmm;
    var pl_minus1 = pm1m;

    for (var l = m + 2u; l <= n; l = l + 1u) {
        let l_f32 = f32(l);
        let m_f32 = f32(m);
        let pl = ((2.0 * l_f32 - 1.0) * x * pl_minus1 - (l_f32 + m_f32 - 1.0) * pl_minus2) / (l_f32 - m_f32);
        pl_minus2 = pl_minus1;
        pl_minus1 = pl;
    }

    return pl_minus1;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    let x = input[idx];

    if (params.is_assoc == 0u) {
        output[idx] = legendre(params.n, x);
    } else {
        output[idx] = assoc_legendre(params.n, params.m, x);
    }
}
