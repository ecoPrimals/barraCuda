//! Nautilus Shell — evolutionary reservoir computing via board ensembles.
//!
//! Provenance: ecoPrimals/primalTools/bingoCube → toadStool standalone absorption.
//! ToadStool maintains standalone nautilus capabilities. Other primals (beardog,
//! songbird) may still use bingoCube directly for inter-primal handshakes.

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
