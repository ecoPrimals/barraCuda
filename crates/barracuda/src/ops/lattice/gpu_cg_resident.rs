// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-resident CG infrastructure structs (absorbed from hotSpring).
//!
//! Type definitions for reduce chains, stream observables, and CG configuration.
//! No runtime GPU execution — these are data structures for pipeline orchestration.

use crate::error::{BarracudaError, Result};
use std::sync::mpsc;

// ═══════════════════════════════════════════════════════════════════
//  Reduce chain (generic over bind group type)
// ═══════════════════════════════════════════════════════════════════

/// One step of a multi-pass reduction: (bind_group, num_workgroups).
///
/// Generic over `B` to avoid direct wgpu dependency — use with barracuda's
/// pipeline infrastructure (e.g. `wgpu::BindGroup` when instantiated).
#[derive(Debug, Clone)]
pub struct ReducePass<B> {
    /// Bind group for this reduction pass.
    pub bg: B,
    /// Number of workgroups to dispatch.
    pub num_wg: u32,
}

/// Pre-built reduction chain: dot_buf → target scalar in 2–3 GPU dispatches.
#[derive(Debug, Clone)]
pub struct ReduceChain<B> {
    /// Ordered reduction passes.
    pub passes: Vec<ReducePass<B>>,
}

// ═══════════════════════════════════════════════════════════════════
//  Stream observables
// ═══════════════════════════════════════════════════════════════════

/// Observable scalars for the readback stream.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamObservables {
    /// Wilson plaquette.
    pub plaquette: f64,
    /// Real part of Polyakov loop.
    pub polyakov_re: f64,
    /// ΔH = H_new - H_old (Metropolis).
    pub delta_h: f64,
    /// CG iterations for this trajectory.
    pub cg_iterations: usize,
    /// Whether Metropolis accepted.
    pub accepted: bool,
}

// ═══════════════════════════════════════════════════════════════════
//  Bidirectional stream
// ═══════════════════════════════════════════════════════════════════

/// Bidirectional stream: GPU ↔ CPU with optional control channel.
///
/// Accumulates observables from trajectories. The control `Sender<bool>` allows
/// external code to signal the stream (e.g. NPU screening decisions).
pub struct BidirectionalStream {
    /// Channel for receiving control signals (e.g. from NPU screening).
    control_rx: Option<mpsc::Receiver<bool>>,
    /// Channel for sending observables to an external consumer (e.g. NPU).
    obs_tx: Option<mpsc::Sender<StreamObservables>>,
    /// Running count of trajectories processed.
    trajectories: usize,
    /// Running count of accepted trajectories.
    accepted: usize,
    /// Accumulated CG iterations across all trajectories.
    total_cg: usize,
}

impl BidirectionalStream {
    /// Create a new bidirectional stream.
    ///
    /// Returns `(stream, control_tx)` — the caller holds `control_tx` to send
    /// control signals (e.g. `true`/`false` for screening decisions).
    #[must_use]
    pub fn new() -> (Self, mpsc::Sender<bool>) {
        let (control_tx, control_rx) = mpsc::channel();
        let stream = Self {
            control_rx: Some(control_rx),
            obs_tx: None,
            trajectories: 0,
            accepted: 0,
            total_cg: 0,
        };
        (stream, control_tx)
    }

    /// Attach an observables channel for forwarding to an external consumer.
    pub fn attach_observables_tx(&mut self, tx: mpsc::Sender<StreamObservables>) {
        self.obs_tx = Some(tx);
    }

    /// Record observables from one trajectory.
    ///
    /// Updates internal statistics and optionally forwards to attached consumer.
    pub fn send_observables(&self, obs: StreamObservables) -> Result<()> {
        if let Some(ref tx) = self.obs_tx {
            tx.send(obs).map_err(|_| BarracudaError::ExecutionError {
                message: "observables channel disconnected".to_string(),
            })?;
        }
        Ok(())
    }

    /// Record observables and update internal statistics.
    ///
    /// Use this when the stream both accumulates and optionally forwards.
    pub fn record_observables(&mut self, obs: StreamObservables) -> Result<()> {
        self.trajectories += 1;
        if obs.accepted {
            self.accepted += 1;
        }
        self.total_cg += obs.cg_iterations;
        self.send_observables(obs)
    }

    /// Return (trajectories, accepted, acceptance_rate).
    #[must_use]
    pub fn statistics(&self) -> (usize, usize, f64) {
        let rate = if self.trajectories == 0 {
            0.0
        } else {
            self.accepted as f64 / self.trajectories as f64
        };
        (self.trajectories, self.accepted, rate)
    }

    /// Non-blocking receive of a control signal.
    #[must_use]
    pub fn try_recv_control(&mut self) -> Option<bool> {
        self.control_rx.as_ref().and_then(|rx| rx.try_recv().ok())
    }
}

impl Default for BidirectionalStream {
    fn default() -> Self {
        let (stream, _tx) = Self::new();
        stream
    }
}

// ═══════════════════════════════════════════════════════════════════
//  GpuResidentCgConfig
// ═══════════════════════════════════════════════════════════════════

/// GPU-resident CG pipeline configuration (no pipeline cache).
///
/// Absorbs pipeline parameters from hotSpring; actual pipeline creation
/// remains in the runtime layer.
#[derive(Debug, Clone)]
pub struct GpuResidentCgConfig {
    /// Maximum CG iterations before giving up.
    pub max_iterations: usize,
    /// Relative residual tolerance for convergence.
    pub tolerance: f64,
    /// Vector length (e.g. n_flat = vol * 6 for staggered fermions).
    pub vector_length: usize,
}

impl Default for GpuResidentCgConfig {
    fn default() -> Self {
        Self {
            max_iterations: super::constants::CG_MAX_ITER_DEFAULT,
            tolerance: 1e-12,
            vector_length: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_observables_fields() {
        let obs = StreamObservables {
            plaquette: 0.5,
            polyakov_re: 0.1,
            delta_h: -0.02,
            cg_iterations: 42,
            accepted: true,
        };
        assert_eq!(obs.plaquette, 0.5);
        assert_eq!(obs.polyakov_re, 0.1);
        assert_eq!(obs.delta_h, -0.02);
        assert_eq!(obs.cg_iterations, 42);
        assert!(obs.accepted);
    }

    #[test]
    fn stream_observables_clone_eq() {
        let a = StreamObservables {
            plaquette: 1.0,
            polyakov_re: 0.0,
            delta_h: 0.0,
            cg_iterations: 10,
            accepted: false,
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn bidirectional_stream_new_returns_pair() {
        let (stream, control_tx) = BidirectionalStream::new();
        let (traj, acc, rate) = stream.statistics();
        assert_eq!(traj, 0);
        assert_eq!(acc, 0);
        assert!((rate - 0.0).abs() < 1e-15);
        drop(control_tx);
    }

    #[test]
    fn bidirectional_stream_record_observables_accumulates() {
        let (mut stream, _control_tx) = BidirectionalStream::new();
        // obs_tx is None by default, so send_observables is a no-op
        let obs1 = StreamObservables {
            plaquette: 0.5,
            polyakov_re: 0.0,
            delta_h: 0.1,
            cg_iterations: 100,
            accepted: true,
        };
        let obs2 = StreamObservables {
            plaquette: 0.5,
            polyakov_re: 0.0,
            delta_h: 0.2,
            cg_iterations: 80,
            accepted: false,
        };

        stream.record_observables(obs1).expect("record 1");
        stream.record_observables(obs2).expect("record 2");

        let (traj, acc, rate) = stream.statistics();
        assert_eq!(traj, 2);
        assert_eq!(acc, 1);
        assert!((rate - 0.5).abs() < 1e-15);
    }

    #[test]
    fn bidirectional_stream_send_observables_to_channel() {
        let (mut stream, _control_tx) = BidirectionalStream::new();
        let (obs_tx, obs_rx) = mpsc::channel();
        stream.attach_observables_tx(obs_tx);

        let obs = StreamObservables {
            plaquette: 0.6,
            polyakov_re: 0.2,
            delta_h: -0.01,
            cg_iterations: 50,
            accepted: true,
        };
        stream.send_observables(obs.clone()).expect("send");
        let received = obs_rx.recv().expect("recv");
        assert_eq!(received, obs);
    }

    #[test]
    fn bidirectional_stream_statistics_empty() {
        let (stream, _) = BidirectionalStream::new();
        let (t, a, r) = stream.statistics();
        assert_eq!(t, 0);
        assert_eq!(a, 0);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn gpu_resident_cg_config_default() {
        let cfg = GpuResidentCgConfig::default();
        assert_eq!(cfg.max_iterations, 5000);
        assert!((cfg.tolerance - 1e-12).abs() < 1e-20);
        assert_eq!(cfg.vector_length, 0);
    }

    #[test]
    fn gpu_resident_cg_config_custom() {
        let cfg = GpuResidentCgConfig {
            max_iterations: 1000,
            tolerance: 1e-10,
            vector_length: 384,
        };
        assert_eq!(cfg.max_iterations, 1000);
        assert!((cfg.tolerance - 1e-10).abs() < 1e-20);
        assert_eq!(cfg.vector_length, 384);
    }

    #[test]
    fn reduce_pass_and_chain_generic() {
        #[derive(Debug, Clone)]
        struct MockBindGroup(u32);
        let pass = ReducePass {
            bg: MockBindGroup(1),
            num_wg: 4,
        };
        assert_eq!(pass.bg.0, 1);
        assert_eq!(pass.num_wg, 4);
        let chain = ReduceChain {
            passes: vec![pass.clone(), pass],
        };
        assert_eq!(chain.passes.len(), 2);
    }
}
