// pangenome_classify.wgsl — Pangenome Gene Classification
//
// One thread per gene cluster. Each thread reads its presence row
// from the flat boolean matrix [n_genes × n_genomes], counts how
// many genomes the gene is present in, and classifies:
//   3 = core (all genomes), 2 = accessory (2+ but not all),
//   1 = unique (exactly 1), 0 = absent (none)
//
// GPU dispatch: ceil(n_genes / 256) workgroups, 256 threads each.
//
// Provenance: wetSpring handoff v5 (Feb 20, 2026) → ToadStool absorption

struct PangenomeParams {
    n_genes:   u32,
    n_genomes: u32,
}

@group(0) @binding(0) var<uniform>             params:     PangenomeParams;
@group(0) @binding(1) var<storage, read>       presence:   array<u32>;  // [n_genes * n_genomes], 0 or 1
@group(0) @binding(2) var<storage, read_write> class_out:  array<u32>;  // [n_genes]: 0/1/2/3
@group(0) @binding(3) var<storage, read_write> count_out:  array<u32>;  // [n_genes]: genome count

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gene_idx = gid.x;
    if gene_idx >= params.n_genes { return; }

    let row_base = gene_idx * params.n_genomes;
    var count: u32 = 0u;

    for (var g: u32 = 0u; g < params.n_genomes; g = g + 1u) {
        if presence[row_base + g] > 0u {
            count = count + 1u;
        }
    }

    count_out[gene_idx] = count;

    if count == params.n_genomes {
        class_out[gene_idx] = 3u;  // core
    } else if count > 1u {
        class_out[gene_idx] = 2u;  // accessory
    } else if count == 1u {
        class_out[gene_idx] = 1u;  // unique
    } else {
        class_out[gene_idx] = 0u;  // absent
    }
}
