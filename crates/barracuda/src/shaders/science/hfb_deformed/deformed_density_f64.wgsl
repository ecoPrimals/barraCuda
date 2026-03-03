// deformed_density_f64.wgsl — Axially-deformed nuclear density on (ρ,z) grid
//
// hotSpring absorption: constructs ρ(ρ,z) from BCS occupation numbers and
// pre-evaluated wavefunctions. Supports density mixing for SCF convergence.
//
// ρ(ρ,z) = Σ_i v²_i · (2j_i + 1) · |ψ_i(ρ,z)|²
//
// Thread assignment: one thread per grid point.
//
// Bindings:
//   0: params         uniform { n_grid_rho, n_grid_z, n_states, _pad, mix_alpha: f64 }
//   1: v_squared      [n_states] f64 — BCS occupation probabilities
//   2: degeneracy     [n_states] f64 — (2j+1) factors
//   3: wavefunctions  [n_states × n_grid_rho × n_grid_z] f64 — from deformed_wavefunction
//   4: density_new    [n_grid_rho × n_grid_z] f64 — output: newly computed density
//   5: density_old    [n_grid_rho × n_grid_z] f64 — previous SCF density (for mixing)
//   6: density_mixed  [n_grid_rho × n_grid_z] f64 — output: mixed density

// f64 enabled by compile_shader_f64() preamble injection

struct DensityParams {
    n_grid_rho: u32,
    n_grid_z:   u32,
    n_states:   u32,
    _pad:       u32,
    mix_alpha:  f64,
}

@group(0) @binding(0) var<uniform>             params: DensityParams;
@group(0) @binding(1) var<storage, read>       v_squared: array<f64>;
@group(0) @binding(2) var<storage, read>       degeneracy: array<f64>;
@group(0) @binding(3) var<storage, read>       wavefunctions: array<f64>;
@group(0) @binding(4) var<storage, read_write> density_new: array<f64>;
@group(0) @binding(5) var<storage, read>       density_old: array<f64>;
@group(0) @binding(6) var<storage, read_write> density_mixed: array<f64>;

@compute @workgroup_size(256)
fn compute_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid_total = params.n_grid_rho * params.n_grid_z;
    if (gid.x >= grid_total) { return; }

    let zero = params.mix_alpha - params.mix_alpha;
    var rho = zero;
    let ns = params.n_states;

    for (var s = 0u; s < ns; s = s + 1u) {
        let psi = wavefunctions[s * grid_total + gid.x];
        rho = rho + v_squared[s] * degeneracy[s] * psi * psi;
    }

    density_new[gid.x] = rho;
}

@compute @workgroup_size(256)
fn mix_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid_total = params.n_grid_rho * params.n_grid_z;
    if (gid.x >= grid_total) { return; }

    let alpha = params.mix_alpha;
    let one = alpha / alpha;
    density_mixed[gid.x] = alpha * density_new[gid.x] + (one - alpha) * density_old[gid.x];
}
