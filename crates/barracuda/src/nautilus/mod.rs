// SPDX-License-Identifier: AGPL-3.0-or-later
//! Nautilus Shell — evolutionary reservoir computing via board ensembles.
//!
//! Provenance: ecoPrimals/primalTools/bingoCube → toadStool → barraCuda.
//! barraCuda maintains standalone nautilus capabilities. Other primals
//! discover each other at runtime via capability-based registration;
//! no primal names are hardcoded here.

pub mod board;
pub mod brain;
pub mod evolution;
pub mod population;
pub mod readout;
pub mod shell;
pub mod spectral_bridge;

pub use board::{Board, BoardConfig, ReservoirInput, ResponseVector};
pub use brain::{BetaObservation, DriftMonitor, NautilusBrain, NautilusBrainConfig};
pub use evolution::{EvolutionConfig, SelectionMethod};
pub use population::{FitnessRecord, Population};
pub use readout::LinearReadout;
pub use shell::{GenerationRecord, InstanceId, NautilusShell, ShellConfig};
pub use spectral_bridge::{SpectralFeatures, SpectralPhase};
