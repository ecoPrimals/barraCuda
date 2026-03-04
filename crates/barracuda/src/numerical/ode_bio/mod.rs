// SPDX-License-Identifier: AGPL-3.0-or-later
//! Biological ODE system implementations for [`BatchedOdeRK4`].
//!
//! Absorbed from wetSpring's 5 quorum-sensing and microbial ecology models
//! (Feb 2026). Each system implements [`OdeSystem`] with both a WGSL derivative
//! (for GPU shader generation) and a CPU derivative (for testing / small batches).
//!
//! | System | Struct | N_VARS | N_PARAMS | Reference |
//! |--------|--------|--------|----------|-----------|
//! | Phenotypic capacitor | [`CapacitorOde`] | 6 | 16 | Mhatre et al. 2020 |
//! | Cooperative QS | [`CooperationOde`] | 4 | 13 | Bruger & Waters 2018 |
//! | Multi-signal QS | [`MultiSignalOde`] | 7 | 24 | Srivastava et al. 2011 |
//! | Bistable switching | [`BistableOde`] | 5 | 21 | Fernandez et al. 2020 |
//! | Phage defense | [`PhageDefenseOde`] | 4 | 11 | Hsueh, Severin et al. 2022 |
//!
//! [`BatchedOdeRK4`]: super::ode_generic::BatchedOdeRK4
//! [`OdeSystem`]: super::ode_generic::OdeSystem

mod params;
mod systems;

pub use params::{
    BistableParams, CapacitorParams, CooperationParams, MultiSignalParams, PhageDefenseParams,
    QsBiofilmParams,
};
pub use systems::{BistableOde, CapacitorOde, CooperationOde, MultiSignalOde, PhageDefenseOde};

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests;
