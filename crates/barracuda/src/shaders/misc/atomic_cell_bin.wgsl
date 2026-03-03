// GPU-Resident Cell-List: Atomic Particle Binning (Pass 1 of 3)
//
// Assigns each particle to its cell and atomically increments the cell count.
// After this pass, cell_counts[c] = number of particles in cell c.
//
// Pass order:
//   1. atomic_cell_bin   (this shader) — particle → cell assignment + count
//   2. prefix_sum        (existing)    — cell_counts → cell_start offsets
//   3. cell_list_scatter (next shader) — write sorted particle indices
//
// Design (hotSpring feedback, Feb 19 2026):
//   The CPU cell-list rebuild reads back all N positions (N×24 bytes), sorts
//   on CPU, then re-uploads.  At N=10,000 that is 240 KB readback + 240 KB
//   upload every 20 steps.  Three embarrassingly parallel GPU passes eliminate
//   all CPU involvement between rebuilds:
//
//     1. Atomic bin  — one thread per particle, atomicAdd to count
//     2. Prefix sum  — parallel scan → cell_start[] offsets
//     3. Scatter     — each particle writes to its cell slot
//
//   barracuda already has prefix_sum.wgsl (Blelloch scan).
//   This shader implements pass 1; cell_list_scatter.wgsl implements pass 3.
//
// Memory layout:
//   positions: [N × 3] f64 interleaved (x0 y0 z0 x1 y1 z1 ...)
//   cell_counts: [Nc] u32, zero-initialised before dispatch
//   cell_ids: [N] u32, output — cell index for particle i
//
// Nc = Mx × My × Mz (total cells)
// cell_id = cx + Mx * cy + Mx * My * cz
//
// Dispatch: (ceil(N / 64), 1, 1)

struct CellBinParams {
    n_particles: u32,
    mx:          u32,    // cells in x
    my:          u32,    // cells in y
    mz:          u32,    // cells in z
    // Box dimensions and cell size (as u32-encoded f32 to avoid f64 uniform
    // complications; precision is sufficient for cell assignment)
    box_lx:   f32,
    box_ly:   f32,
    box_lz:   f32,
    cell_size: f32,      // same in all three dimensions
}

@group(0) @binding(0) var<uniform>            params:      CellBinParams;
@group(0) @binding(1) var<storage, read>      positions:   array<f64>;  // [N × 3]
@group(0) @binding(2) var<storage, read_write> cell_counts: array<atomic<u32>>; // [Nc]
@group(0) @binding(3) var<storage, read_write> cell_ids:    array<u32>;  // [N]

@compute @workgroup_size(64)
fn atomic_cell_bin(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let i = gid.x;
    if (i >= params.n_particles) { return; }

    // Read position (f64 stored as [x, y, z] at offset i*3)
    let px = f32(positions[i * 3u]);
    let py = f32(positions[i * 3u + 1u]);
    let pz = f32(positions[i * 3u + 2u]);

    // Map position into [0, L) with periodic wrap
    let wx = ((px % params.box_lx) + params.box_lx) % params.box_lx;
    let wy = ((py % params.box_ly) + params.box_ly) % params.box_ly;
    let wz = ((pz % params.box_lz) + params.box_lz) % params.box_lz;

    // Cell index (clamped to avoid out-of-bounds on boundary particles)
    let cx = min(u32(wx / params.cell_size), params.mx - 1u);
    let cy = min(u32(wy / params.cell_size), params.my - 1u);
    let cz = min(u32(wz / params.cell_size), params.mz - 1u);

    let cell_id = cx + params.mx * cy + params.mx * params.my * cz;

    // Store cell assignment for this particle
    cell_ids[i] = cell_id;

    // Atomically increment the count for this cell
    atomicAdd(&cell_counts[cell_id], 1u);
}
