// SPDX-License-Identifier: AGPL-3.0-or-later

//! Multi-stage pipeline dispatch across substrates.
//!
//! A pipeline is an ordered sequence of stages, each requiring specific
//! capabilities. Data flows between stages via typed intermediate buffers.
//! The planner uses [`InterconnectTopology`] to minimize transfer overhead,
//! preferring `PCIe` P2P over CPU bounce when possible.
//!
//! # Example: NPU classification → GPU refinement → CPU provenance
//!
//! ```text
//! Stage 0: NPU (int8 classify)  → regime labels [0,1,2]
//! Stage 1: GPU (f64 Lyapunov)   ← regime labels → full spectrum
//! Stage 2: CPU (provenance)     ← spectrum → stored results
//! ```
//!
//! Absorbed from groundSpring V61 `metalForge/forge/src/pipeline.rs`.

use super::interconnect::InterconnectTopology;
use crate::device::substrate::{Substrate, SubstrateCapability, SubstrateType};

/// Transfer strategy between pipeline stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferStrategy {
    /// Direct peer-to-peer DMA (NPU↔GPU via `PCIe`, GPU↔GPU via `NvLink`).
    PeerToPeer,
    /// Bounce through CPU host memory.
    HostBounce,
    /// No transfer needed (same device or first stage).
    None,
}

/// A workload descriptor: what capabilities a stage requires.
#[derive(Debug, Clone)]
pub struct StageWorkload {
    /// Human-readable workload name.
    pub name: String,
    /// Capabilities the substrate must satisfy.
    pub required_capabilities: Vec<SubstrateCapability>,
}

impl StageWorkload {
    /// Create a workload with required capabilities.
    pub fn new(name: impl Into<String>, caps: Vec<SubstrateCapability>) -> Self {
        Self {
            name: name.into(),
            required_capabilities: caps,
        }
    }
}

/// A single stage in a multi-substrate pipeline.
#[derive(Debug)]
pub struct PipelineStage {
    /// Stage name for logging.
    pub name: String,
    /// Workload and capability requirements.
    pub workload: StageWorkload,
    /// Output buffer size for transfer cost estimation.
    pub output_bytes: u64,
    /// Policy when preferred substrate is unavailable.
    pub fallback: FallbackPolicy,
}

/// What to do when a stage's preferred substrate is unavailable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackPolicy {
    /// Try the next-best substrate (GPU→CPU, NPU→CPU).
    Degrade,
    /// Skip this stage entirely.
    Skip,
    /// Fail the pipeline immediately.
    Fail,
}

/// How a stage was resolved during planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageResolution {
    /// Preferred substrate assigned.
    Optimal,
    /// Fallback substrate used.
    Degraded,
    /// Stage skipped (fallback policy was Skip).
    Skipped,
}

/// A resolved pipeline stage with substrate assignment.
#[derive(Debug)]
pub struct ResolvedStage<'a> {
    /// The pipeline stage.
    pub stage: &'a PipelineStage,
    /// Assigned substrate (None if skipped).
    pub substrate: Option<&'a Substrate>,
    /// Transfer strategy from previous stage.
    pub transfer: TransferStrategy,
    /// Estimated transfer time (microseconds).
    pub transfer_cost_us: u64,
    /// How this stage was resolved.
    pub reason: StageResolution,
}

/// Builder for multi-substrate pipelines.
#[derive(Debug)]
pub struct SubstratePipeline {
    /// Pipeline name for logging.
    pub name: String,
    /// Ordered stages.
    pub stages: Vec<PipelineStage>,
}

/// A resolved pipeline ready for execution.
#[derive(Debug)]
pub struct ResolvedPipeline<'a> {
    /// Pipeline name.
    pub name: &'a str,
    /// Resolved stages with substrate assignments.
    pub stages: Vec<ResolvedStage<'a>>,
    /// Total estimated transfer overhead (microseconds).
    pub total_transfer_us: u64,
    /// True if all stages got their preferred substrate.
    pub fully_optimal: bool,
}

impl SubstratePipeline {
    /// Create a new pipeline builder.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            stages: Vec::new(),
        }
    }

    /// Add a stage (builder pattern).
    #[must_use]
    pub fn stage(mut self, stage: PipelineStage) -> Self {
        self.stages.push(stage);
        self
    }

    /// Plan the pipeline: assign substrates and compute transfer costs.
    #[must_use]
    pub fn plan<'a>(
        &'a self,
        substrates: &'a [Substrate],
        topology: &InterconnectTopology,
    ) -> ResolvedPipeline<'a> {
        let mut resolved_stages = Vec::with_capacity(self.stages.len());
        let mut total_transfer_us = 0u64;
        let mut fully_optimal = true;
        let mut prev_substrate_idx: Option<usize> = None;

        for stage in &self.stages {
            let (substrate, reason) = match route_workload(&stage.workload, substrates) {
                Some((idx, sub)) => {
                    let _ = idx;
                    (Some(sub), StageResolution::Optimal)
                }
                None => match stage.fallback {
                    FallbackPolicy::Degrade => {
                        let degraded = find_cpu_fallback(substrates);
                        if degraded.is_some() {
                            fully_optimal = false;
                        }
                        (degraded, StageResolution::Degraded)
                    }
                    FallbackPolicy::Skip => {
                        fully_optimal = false;
                        (None, StageResolution::Skipped)
                    }
                    FallbackPolicy::Fail => (None, StageResolution::Optimal),
                },
            };

            let (transfer, transfer_cost) =
                if let (Some(prev_idx), Some(sub)) = (prev_substrate_idx, substrate) {
                    let curr_idx = substrates
                        .iter()
                        .position(|s| std::ptr::eq(s, sub))
                        .unwrap_or(0);
                    if prev_idx == curr_idx {
                        (TransferStrategy::None, 0)
                    } else {
                        let prev_output = resolved_stages
                            .last()
                            .map_or(0, |rs: &ResolvedStage<'_>| rs.stage.output_bytes);
                        let link = topology.best_link(prev_idx, curr_idx);
                        let cost = link.map_or(0, |l| l.tier.transfer_time_us(prev_output));
                        let strategy = link.map_or(TransferStrategy::HostBounce, |l| {
                            if l.tier.is_peer_to_peer() {
                                TransferStrategy::PeerToPeer
                            } else {
                                TransferStrategy::HostBounce
                            }
                        });
                        (strategy, cost)
                    }
                } else {
                    (TransferStrategy::None, 0)
                };

            total_transfer_us += transfer_cost;

            if let Some(sub) = substrate {
                prev_substrate_idx = substrates.iter().position(|s| std::ptr::eq(s, sub));
            }

            resolved_stages.push(ResolvedStage {
                stage,
                substrate,
                transfer,
                transfer_cost_us: transfer_cost,
                reason,
            });
        }

        ResolvedPipeline {
            name: &self.name,
            stages: resolved_stages,
            total_transfer_us,
            fully_optimal,
        }
    }
}

impl PipelineStage {
    /// Create a new pipeline stage.
    pub fn new(name: impl Into<String>, workload: StageWorkload, output_bytes: u64) -> Self {
        Self {
            name: name.into(),
            workload,
            output_bytes,
            fallback: FallbackPolicy::Degrade,
        }
    }

    /// Set fallback policy when preferred substrate is unavailable.
    #[must_use]
    pub const fn with_fallback(mut self, policy: FallbackPolicy) -> Self {
        self.fallback = policy;
        self
    }
}

impl ResolvedPipeline<'_> {
    /// Returns true if all stages have a substrate assigned.
    #[must_use]
    pub fn all_assigned(&self) -> bool {
        self.stages.iter().all(|s| s.substrate.is_some())
    }

    /// Number of stages using peer-to-peer transfer.
    #[must_use]
    pub fn p2p_transfer_count(&self) -> usize {
        self.stages
            .iter()
            .filter(|s| s.transfer == TransferStrategy::PeerToPeer)
            .count()
    }

    /// Number of stages that used fallback substrate.
    #[must_use]
    pub fn degraded_count(&self) -> usize {
        self.stages
            .iter()
            .filter(|s| s.reason == StageResolution::Degraded)
            .count()
    }

    /// Log pipeline summary to tracing.
    pub fn print_summary(&self) {
        tracing::info!("Pipeline: {}", self.name);
        tracing::info!(
            "  Stages: {} | Transfer overhead: {}µs | Optimal: {}",
            self.stages.len(),
            self.total_transfer_us,
            self.fully_optimal,
        );
        for (i, rs) in self.stages.iter().enumerate() {
            let sub_name = rs.substrate.map_or("(skipped)", |s| s.name.as_str());
            let transfer_str = match rs.transfer {
                TransferStrategy::PeerToPeer => "←P2P←",
                TransferStrategy::HostBounce => "←HOST←",
                TransferStrategy::None => "",
            };
            tracing::info!(
                "  [{i}] {:<30} → {:<20} {transfer_str} ({:?})",
                rs.stage.name,
                sub_name,
                rs.reason,
            );
        }
    }
}

/// Capability-based workload routing: find the best substrate that satisfies
/// all required capabilities, preferring GPU > NPU > CPU.
fn route_workload<'a>(
    workload: &StageWorkload,
    substrates: &'a [Substrate],
) -> Option<(usize, &'a Substrate)> {
    let mut best: Option<(usize, &Substrate, u32)> = None;
    for (idx, sub) in substrates.iter().enumerate() {
        if workload
            .required_capabilities
            .iter()
            .all(|cap| sub.has(cap))
        {
            let priority = match sub.substrate_type {
                SubstrateType::NvidiaGpu => 100,
                SubstrateType::AmdGpu => 90,
                SubstrateType::IntelGpu => 80,
                SubstrateType::AppleGpu => 80,
                SubstrateType::Npu => 70,
                SubstrateType::Cpu => 10,
                SubstrateType::Other => 5,
            };
            if best.as_ref().is_none_or(|(_, _, p)| priority > *p) {
                best = Some((idx, sub, priority));
            }
        }
    }
    best.map(|(idx, sub, _)| (idx, sub))
}

fn find_cpu_fallback(substrates: &[Substrate]) -> Option<&Substrate> {
    substrates
        .iter()
        .find(|s| s.substrate_type == SubstrateType::Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::substrate::SubstrateCapability;

    fn gpu_sub() -> Substrate {
        Substrate {
            substrate_type: SubstrateType::NvidiaGpu,
            name: "Test GPU".to_string(),
            backend: "Vulkan".to_string(),
            index: 0,
            capabilities: vec![
                SubstrateCapability::F64Compute,
                SubstrateCapability::ShaderDispatch,
                SubstrateCapability::ScalarReduce,
            ],
        }
    }

    fn npu_sub() -> Substrate {
        Substrate {
            substrate_type: SubstrateType::Npu,
            name: "BrainChip AKD1000".to_string(),
            backend: "PCIe".to_string(),
            index: 0,
            capabilities: vec![
                SubstrateCapability::QuantizedInference { bits: 8 },
                SubstrateCapability::BatchInference { max_batch: 8 },
            ],
        }
    }

    fn cpu_sub() -> Substrate {
        Substrate {
            substrate_type: SubstrateType::Cpu,
            name: "Test CPU".to_string(),
            backend: "native".to_string(),
            index: 0,
            capabilities: vec![
                SubstrateCapability::F64Compute,
                SubstrateCapability::F32Compute,
            ],
        }
    }

    #[test]
    fn pipeline_plans_npu_then_gpu() {
        let subs = vec![gpu_sub(), npu_sub(), cpu_sub()];
        let topo = InterconnectTopology::infer(&subs);

        let pipeline = SubstratePipeline::new("NPU classify → GPU refine")
            .stage(PipelineStage::new(
                "NPU classify",
                StageWorkload::new(
                    "classify",
                    vec![SubstrateCapability::QuantizedInference { bits: 8 }],
                ),
                1024,
            ))
            .stage(PipelineStage::new(
                "GPU Lyapunov",
                StageWorkload::new(
                    "lyapunov",
                    vec![
                        SubstrateCapability::F64Compute,
                        SubstrateCapability::ShaderDispatch,
                    ],
                ),
                8192,
            ));

        let resolved = pipeline.plan(&subs, &topo);
        assert!(resolved.all_assigned());
        assert_eq!(resolved.stages.len(), 2);
        assert_eq!(
            resolved.stages[0].substrate.unwrap().substrate_type,
            SubstrateType::Npu
        );
        assert_eq!(
            resolved.stages[1].substrate.unwrap().substrate_type,
            SubstrateType::NvidiaGpu
        );
    }

    #[test]
    fn pipeline_degrades_when_npu_missing() {
        let subs = vec![gpu_sub(), cpu_sub()];
        let topo = InterconnectTopology::infer(&subs);

        let pipeline = SubstratePipeline::new("degrade test").stage(
            PipelineStage::new(
                "NPU classify",
                StageWorkload::new(
                    "classify",
                    vec![SubstrateCapability::QuantizedInference { bits: 8 }],
                ),
                1024,
            )
            .with_fallback(FallbackPolicy::Degrade),
        );

        let resolved = pipeline.plan(&subs, &topo);
        assert!(resolved.all_assigned());
        assert_eq!(resolved.degraded_count(), 1);
        assert_eq!(
            resolved.stages[0].substrate.unwrap().substrate_type,
            SubstrateType::Cpu
        );
    }

    #[test]
    fn pipeline_skips_when_policy_is_skip() {
        let subs = vec![cpu_sub()];
        let topo = InterconnectTopology::infer(&subs);

        let pipeline = SubstratePipeline::new("skip test").stage(
            PipelineStage::new(
                "GPU work",
                StageWorkload::new(
                    "gpu_only",
                    vec![
                        SubstrateCapability::F64Compute,
                        SubstrateCapability::ShaderDispatch,
                    ],
                ),
                4096,
            )
            .with_fallback(FallbackPolicy::Skip),
        );

        let resolved = pipeline.plan(&subs, &topo);
        assert!(!resolved.all_assigned());
        assert!(resolved.stages[0].substrate.is_none());
        assert_eq!(resolved.stages[0].reason, StageResolution::Skipped);
    }

    #[test]
    fn pipeline_fully_optimal_when_all_match() {
        let subs = vec![gpu_sub(), cpu_sub()];
        let topo = InterconnectTopology::infer(&subs);

        let pipeline = SubstratePipeline::new("optimal").stage(PipelineStage::new(
            "GPU work",
            StageWorkload::new(
                "compute",
                vec![
                    SubstrateCapability::F64Compute,
                    SubstrateCapability::ShaderDispatch,
                ],
            ),
            4096,
        ));

        let resolved = pipeline.plan(&subs, &topo);
        assert!(resolved.fully_optimal);
    }

    #[test]
    fn empty_pipeline_plans_ok() {
        let subs = vec![cpu_sub()];
        let topo = InterconnectTopology::infer(&subs);
        let pipeline = SubstratePipeline::new("empty");
        let resolved = pipeline.plan(&subs, &topo);
        assert!(resolved.all_assigned());
        assert!(resolved.fully_optimal);
        assert_eq!(resolved.total_transfer_us, 0);
    }

    #[test]
    fn same_substrate_no_transfer() {
        let subs = vec![gpu_sub(), cpu_sub()];
        let topo = InterconnectTopology::infer(&subs);

        let pipeline = SubstratePipeline::new("GPU→GPU")
            .stage(PipelineStage::new(
                "first",
                StageWorkload::new(
                    "a",
                    vec![
                        SubstrateCapability::F64Compute,
                        SubstrateCapability::ShaderDispatch,
                    ],
                ),
                4096,
            ))
            .stage(PipelineStage::new(
                "second",
                StageWorkload::new(
                    "b",
                    vec![
                        SubstrateCapability::F64Compute,
                        SubstrateCapability::ShaderDispatch,
                    ],
                ),
                4096,
            ));

        let resolved = pipeline.plan(&subs, &topo);
        assert_eq!(resolved.stages[1].transfer, TransferStrategy::None);
        assert_eq!(resolved.stages[1].transfer_cost_us, 0);
    }
}
