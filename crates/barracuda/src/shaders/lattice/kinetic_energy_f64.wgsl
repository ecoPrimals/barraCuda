// kinetic_energy_f64.wgsl — Per-link kinetic energy from HMC momenta
//
// Prepend: complex_f64.wgsl + su3.wgsl
//
// Computes T_link = -0.5 × Re Tr(π² ) for each link.
// Total kinetic energy = Σ T_link  (reduction on host).
//
// Buffer layout:
//   momenta[V × 4 × 18]:  conjugate momenta (algebra elements)
//   energy[V × 4]:         per-link kinetic energy (output)

struct KineticParams {
    n_links: u32,
    _pad0:   u32,
    _pad1:   u32,
    _pad2:   u32,
}

@group(0) @binding(0) var<uniform>             params:  KineticParams;
@group(0) @binding(1) var<storage, read>       momenta: array<f64>;
@group(0) @binding(2) var<storage, read_write> energy:  array<f64>;

fn load_su3(base: u32) -> array<vec2<f64>, 9> {
    var m: array<vec2<f64>, 9>;
    for (var i = 0u; i < 9u; i = i + 1u) {
        m[i] = c64_new(momenta[base + i * 2u], momenta[base + i * 2u + 1u]);
    }
    return m;
}

@compute @workgroup_size(64)
fn kinetic_energy_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let link_id = gid.x;
    if (link_id >= params.n_links) { return; }

    let base = link_id * 18u;
    let p = load_su3(base);
    let p2 = su3_mul(p, p);
    energy[link_id] = -0.5 * su3_re_trace(p2);
}
