// SPDX-License-Identifier: AGPL-3.0-only

//! Life-science, evolutionary computation, and analytical-chemistry GPU primitives.
//!
//! Absorbed from wetSpring handoff v4–v8 and neuralSpring metalForge (Feb 2026).
//!
//! | Module | Shader | Primitive |
//! |--------|--------|-----------|
//! | `smith_waterman`    | `smith_waterman_banded_f64.wgsl` | Banded SW local alignment |
//! | `gillespie`         | `gillespie_ssa_f64.wgsl`         | Parallel Gillespie SSA |
//! | `tree_inference`    | `tree_inference_f64.wgsl`        | Decision tree / RF inference |
//! | `felsenstein`       | `felsenstein_f64.wgsl`           | Felsenstein pruning likelihood |
//! | `hmm`               | `hmm_forward_f64.wgsl`           | Batch HMM forward (f64) |
//! | `ani`               | `ani_batch_f64.wgsl`             | Pairwise ANI |
//! | `snp`               | `snp_calling_f64.wgsl`           | Position-parallel SNP calling |
//! | `dnds`              | `dnds_batch_f64.wgsl`            | Batch Nei-Gojobori dN/dS |
//! | `pangenome`         | `pangenome_classify.wgsl`        | Gene family classification |
//! | `quality_filter`    | `quality_filter.wgsl`            | Per-read quality trimming |
//! | `batched_multinomial`| `batched_multinomial_f64.wgsl`  | Batched multinomial rarefaction |
//! | `dada2`             | `dada2_e_step.wgsl`              | DADA2 E-step (batch log_p_error) |
//! | `diversity_fusion`  | `diversity_fusion_f64.wgsl`      | Fused Shannon + Simpson + evenness |
//! | `locus_variance`    | `locus_variance.wgsl`            | Per-locus AF variance (FST) |
//! | `multi_obj_fitness` | `multi_obj_fitness.wgsl`         | Multi-objective fitness eval |
//! | `pairwise_hamming`  | `pairwise_hamming.wgsl`          | Pairwise Hamming distance |
//! | `pairwise_l2`      | `pairwise_l2.wgsl`               | Pairwise L2 (Euclidean) distance |
//! | `pairwise_jaccard`  | `pairwise_jaccard.wgsl`          | Pairwise Jaccard distance |
//! | `hill_gate`        | `hill_gate.wgsl`                | Two-input Hill AND gate |
//! | `spatial_payoff`    | `spatial_payoff.wgsl`            | Spatial PD payoff stencil |
//! | `batch_fitness`     | `batch_fitness_eval.wgsl`        | EA batch fitness evaluation |
//! | `rf_inference`      | `rf_batch_inference.wgsl`        | Batch RF inference (SoA f64) |
//! | `swarm_nn`          | `swarm_nn_forward.wgsl`         | Swarm NN forward pass |
//! | `kmer_histogram`    | `kmer_histogram.wgsl`            | K-mer counting (atomic histogram) |
//! | `taxonomy_fc`       | `taxonomy_fc.wgsl`               | Naive Bayes taxonomy FC (f64) |
//! | `unifrac_propagate` | `unifrac_propagate.wgsl`         | UniFrac tree propagation (f64) |

pub mod ani;
pub mod batch_fitness;
pub mod batched_multinomial;
pub mod bipartition_encode;
pub mod dada2;
pub mod diversity_fusion;
pub mod dnds;
pub mod felsenstein;
pub mod flat_tree;
pub mod fst_variance;
pub mod gillespie;
pub mod hill_gate;
pub mod hmm;
pub mod kmer_histogram;
pub mod locus_variance;
pub mod multi_obj_fitness;
pub mod ncbi_cache;
pub mod pairwise_hamming;
pub mod pairwise_jaccard;
pub mod pairwise_l2;
pub mod pangenome;
pub mod quality_filter;
pub mod rf_inference;
pub mod smith_waterman;
pub mod snp;
pub mod spatial_payoff;
pub mod stencil_cooperation;
pub mod swarm_nn;
pub mod taxonomy_fc;
pub mod tree_inference;
pub mod unifrac_propagate;
pub mod wright_fisher;

pub use ani::AniBatchF64;
pub use batch_fitness::BatchFitnessGpu;
pub use batched_multinomial::{
    BatchedMultinomialConfig, BatchedMultinomialGpu, multinomial_sample_cpu,
};
pub use bipartition_encode::BipartitionEncodeGpu;
pub use dada2::{Dada2Buffers, Dada2Dimensions, Dada2DispatchArgs, Dada2EStepGpu};
pub use diversity_fusion::{DiversityFusionGpu, DiversityResult, diversity_fusion_cpu};
pub use dnds::DnDsBatchF64;
pub use felsenstein::{FelsensteinGpu, FelsensteinResult, PhyloTree};
pub use flat_tree::FlatTree;
pub use fst_variance::{FstResult, fst_variance_decomposition};
pub use gillespie::{GillespieConfig, GillespieGpu, GillespieModel, GillespieResult};
pub use hill_gate::{HillGateGpu, HillGateParams};
pub use hmm::{HmmBatchForwardF64, HmmForwardArgs, ViterbiResult, hmm_backward, hmm_viterbi};
pub use kmer_histogram::{KmerHistogramGpu, histogram_u32_to_f64};
pub use locus_variance::LocusVarianceGpu;
pub use multi_obj_fitness::MultiObjFitnessGpu;
pub use ncbi_cache::NcbiCache;
pub use pairwise_hamming::PairwiseHammingGpu;
pub use pairwise_jaccard::PairwiseJaccardGpu;
pub use pairwise_l2::PairwiseL2Gpu;
pub use pangenome::PangenomeClassifyGpu;
pub use quality_filter::{QualityConfig, QualityFilterGpu};
pub use rf_inference::RfBatchInferenceGpu;
pub use smith_waterman::{SmithWatermanGpu, SwConfig, SwResult};
pub use snp::SnpCallingF64;
pub use spatial_payoff::SpatialPayoffGpu;
pub use stencil_cooperation::StencilCooperationGpu;
pub use swarm_nn::SwarmNnGpu;
pub use taxonomy_fc::TaxonomyFcGpu;
pub use tree_inference::{FlatForest, TreeInferenceGpu};
pub use unifrac_propagate::{UniFracConfig, UniFracPropagateGpu};
pub use wright_fisher::WrightFisherGpu;
