// deformed_bcs_f64.wgsl — BCS pairing for axially-deformed nuclei
//
// hotSpring absorption: same BCS physics as spherical but with deformed
// single-particle energies and level degeneracies.
//
// Three entry points:
//   bcs_occupations: v²_i, u²_i from (ε_i, λ, Δ)
//   particle_number: N = Σ (2j_i+1) · v²_i  (for bisection solver)
//   gap_equation:    Δ = G/2 · Σ (2j_i+1) · u_i · v_i
//
// Bindings:
//   0: params       uniform { n_states, _pad, lambda, delta, G, target_N: f64 }
//   1: sp_energies  [n_states] f64
//   2: degeneracy   [n_states] f64 — (2j_i + 1) per deformed level
//   3: v_sq         [n_states] f64 — output: v²_i
//   4: u_sq         [n_states] f64 — output: u²_i
//   5: result       [2] f64 — result[0] = particle count, result[1] = gap

// f64 enabled by compile_shader_f64() preamble injection

struct BcsParams {
    n_states: u32,
    _pad:     u32,
    lambda:   f64,
    delta:    f64,
    G:        f64,
    target_N: f64,
}

@group(0) @binding(0) var<uniform>             params: BcsParams;
@group(0) @binding(1) var<storage, read>       sp_energies: array<f64>;
@group(0) @binding(2) var<storage, read>       degeneracy: array<f64>;
@group(0) @binding(3) var<storage, read_write> v_sq: array<f64>;
@group(0) @binding(4) var<storage, read_write> u_sq: array<f64>;
@group(0) @binding(5) var<storage, read_write> result: array<f64>;

@compute @workgroup_size(256)
fn bcs_occupations(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n_states) { return; }

    let zero = params.lambda - params.lambda;
    let half = zero + 0.5;
    let eps = sp_energies[gid.x];
    let xi = eps - params.lambda;
    let e_qp = sqrt(xi * xi + params.delta * params.delta);

    let v2 = half * ((zero + 1.0) - xi / e_qp);
    let u2 = half * ((zero + 1.0) + xi / e_qp);

    v_sq[gid.x] = v2;
    u_sq[gid.x] = u2;
}

var<workgroup> shared_n: array<f64, 256>;
var<workgroup> shared_gap: array<f64, 256>;

@compute @workgroup_size(256)
fn particle_number_and_gap(@builtin(global_invocation_id) gid: vec3<u32>,
                           @builtin(local_invocation_id) lid: vec3<u32>) {
    let zero = params.lambda - params.lambda;
    var n_val = zero;
    var gap_val = zero;

    if (gid.x < params.n_states) {
        let degen = degeneracy[gid.x];
        n_val = degen * v_sq[gid.x];
        gap_val = degen * sqrt(v_sq[gid.x] * u_sq[gid.x]);
    }

    shared_n[lid.x] = n_val;
    shared_gap[lid.x] = gap_val;
    workgroupBarrier();

    var stride = 128u;
    while (stride > 0u) {
        if (lid.x < stride) {
            shared_n[lid.x] = shared_n[lid.x] + shared_n[lid.x + stride];
            shared_gap[lid.x] = shared_gap[lid.x] + shared_gap[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (lid.x == 0u) {
        result[0] = shared_n[0];
        result[1] = params.G * (zero + 0.5) * shared_gap[0];
    }
}
