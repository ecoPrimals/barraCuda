// pairwise_jaccard_f64.wgsl — Pairwise Jaccard Distance (f64 canonical)
//
// Computes the upper-triangle Jaccard distance matrix for a pangenome
// presence/absence (PA) matrix. Each thread handles one genome pair.
//
// Jaccard(i,j) = 1 - |intersection(i,j)| / |union(i,j)|
//
// PA matrix stored column-major: pa[gene * n_genomes + genome].
//
// Provenance: neuralSpring metalForge (Feb 21, 2026) → ToadStool absorption

@group(0) @binding(0) var<storage, read> pa: array<f64>;
@group(0) @binding(1) var<storage, read_write> distances: array<f64>;

struct JaccardParams {
    n_genomes: u32,
    n_genes: u32,
}
@group(0) @binding(2) var<uniform> params: JaccardParams;

@compute @workgroup_size(256)
fn pairwise_jaccard(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = gid.x;
    let n = params.n_genomes;
    let n_pairs = n * (n - 1u) / 2u;
    if pair_idx >= n_pairs {
        return;
    }

    // Decode pair index to (i, j) where i < j
    var i: u32 = 0u;
    var remaining = pair_idx;
    loop {
        let row_len = n - 1u - i;
        if remaining < row_len {
            break;
        }
        remaining = remaining - row_len;
        i = i + 1u;
    }
    let j = i + 1u + remaining;

    var intersection: f64 = 0.0;
    var union_count: f64 = 0.0;
    for (var g: u32 = 0u; g < params.n_genes; g = g + 1u) {
        let a = pa[g * n + i];
        let b = pa[g * n + j];
        intersection = intersection + a * b;
        union_count = union_count + max(a, b);
    }

    var dist: f64 = 1.0;
    if union_count > 0.0 {
        dist = 1.0 - intersection / union_count;
    }
    distances[pair_idx] = dist;
}
