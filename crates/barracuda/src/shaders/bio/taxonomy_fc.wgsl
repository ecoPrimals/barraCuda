// SPDX-License-Identifier: AGPL-3.0-or-later
// taxonomy_fc.wgsl — GPU taxonomy Naive Bayes scoring (f64)
//
// wetSpring absorption: metagenomic taxonomy classification.
//
// Computes log-posterior = log_prior + sum(log_prob[feature]) for each taxon.
// One thread per (query, taxon) pair. GEMM-like but with log-space accumulation.
//
// CPU reference: wetspring_barracuda::bio::taxonomy::NaiveBayesClassifier::classify
//
// Bindings:
//   0: config     uniform { n_queries, n_taxa, n_features, _pad }
//   1: log_probs  [n_taxa × n_features] f64 — log emission probabilities
//   2: log_priors [n_taxa] f64 — log prior probabilities
//   3: features   [n_queries × n_features] u32 — binary feature presence
//   4: scores     [n_queries × n_taxa] f64 — output log-posteriors

struct TaxConfig {
    n_queries:  u32,
    n_taxa:     u32,
    n_features: u32,
    _pad:       u32,
}

@group(0) @binding(0) var<uniform>             config:     TaxConfig;
@group(0) @binding(1) var<storage, read>       log_probs:  array<f64>;
@group(0) @binding(2) var<storage, read>       log_priors: array<f64>;
@group(0) @binding(3) var<storage, read>       features:   array<u32>;
@group(0) @binding(4) var<storage, read_write> scores:     array<f64>;

@compute @workgroup_size(16, 16)
fn taxonomy_fc(@builtin(global_invocation_id) gid: vec3<u32>) {
    let query = gid.x;
    let taxon = gid.y;

    if query >= config.n_queries || taxon >= config.n_taxa {
        return;
    }

    var score = log_priors[taxon];

    let feat_base = query * config.n_features;
    let prob_base = taxon * config.n_features;

    for (var f: u32 = 0u; f < config.n_features; f = f + 1u) {
        let feat_present = features[feat_base + f];
        if feat_present != 0u {
            score = score + log_probs[prob_base + f];
        }
    }

    scores[query * config.n_taxa + taxon] = score;
}
