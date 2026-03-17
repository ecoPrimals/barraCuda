// SPDX-License-Identifier: AGPL-3.0-only
//! `OdeSystem` trait implementations connecting biological parameter structs
//! to the generic `BatchedOdeRK4<S>` GPU/CPU framework.
//!
//! Absorbed from wetSpring v24–v25 (Feb 2026). Each implementation provides:
//! - A self-contained WGSL `deriv` function (with inline helpers: `fmax_d`,
//!   `fpow_d`, `hill_d`, etc.) for GPU shader generation.
//! - A CPU `cpu_derivative` that mirrors the WGSL logic using the param struct.
//!
//! The WGSL uses `exp()` and `log()` in `fpow_d()` — on affected drivers
//! (NVK, NVVM Ada), `compile_shader_f64` / `for_driver_auto` will automatically
//! patch these to `exp_f64()` / `log_f64()` via the transcendental workaround.

mod bistable;
mod capacitor;
mod cooperation;
mod multi_signal;
mod phage_defense;

pub use bistable::BistableOde;
pub use capacitor::CapacitorOde;
pub use cooperation::CooperationOde;
pub use multi_signal::MultiSignalOde;
pub use phage_defense::PhageDefenseOde;

#[cfg(test)]
mod tests;
