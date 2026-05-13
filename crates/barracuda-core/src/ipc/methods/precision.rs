// SPDX-License-Identifier: AGPL-3.0-or-later
//! Precision routing advisory handler.
//!
//! Exposes `PrecisionBrain`'s domain → tier routing over JSON-RPC so that
//! upstream primals (hotSpring, springs) can query the recommended precision
//! tier, hardware hint, and compiler requirements for a given physics domain.

use super::super::jsonrpc::JsonRpcResponse;
use crate::BarraCudaPrimal;
use barracuda::device::{PhysicsDomain, PrecisionBrain};
use serde_json::Value;

/// Parse a domain string (snake_case) into a `PhysicsDomain`.
fn parse_domain(s: &str) -> Option<PhysicsDomain> {
    match s {
        "lattice_qcd" => Some(PhysicsDomain::LatticeQcd),
        "gradient_flow" => Some(PhysicsDomain::GradientFlow),
        "dielectric" => Some(PhysicsDomain::Dielectric),
        "kinetic_fluid" => Some(PhysicsDomain::KineticFluid),
        "eigensolve" => Some(PhysicsDomain::Eigensolve),
        "molecular_dynamics" => Some(PhysicsDomain::MolecularDynamics),
        "nuclear_eos" => Some(PhysicsDomain::NuclearEos),
        "population_pk" => Some(PhysicsDomain::PopulationPk),
        "bioinformatics" => Some(PhysicsDomain::Bioinformatics),
        "hydrology" => Some(PhysicsDomain::Hydrology),
        "statistics" => Some(PhysicsDomain::Statistics),
        "general" => Some(PhysicsDomain::General),
        "inference" => Some(PhysicsDomain::Inference),
        "training" => Some(PhysicsDomain::Training),
        "hashing" => Some(PhysicsDomain::Hashing),
        _ => None,
    }
}

/// Map a `HardwareHint` to its wire-protocol string.
fn hint_to_str(hint: barracuda::device::backend::HardwareHint) -> &'static str {
    use barracuda::device::backend::HardwareHint;
    match hint {
        HardwareHint::Compute => "compute",
        HardwareHint::TensorCore => "tensor_core",
        HardwareHint::RtCore => "rt_core",
        HardwareHint::ZBuffer => "zbuffer",
        HardwareHint::TextureUnit => "texture_unit",
        HardwareHint::RopBlend => "rop_blend",
    }
}

/// Resolve the active dispatch path from the compute device tier.
///
/// - `"wgpu"` — local GPU via wgpu (tiers 1–2), shader execution stays in-process
/// - `"sovereign"` — VFIO/DRM dispatch via toadStool IPC (`compute.dispatch.submit`)
/// - `"unavailable"` — no compute device discovered
fn resolve_dispatch_path(primal: &BarraCudaPrimal) -> &'static str {
    match primal.compute_device() {
        Some(dev) if dev.is_sovereign() => "sovereign",
        Some(_) => "wgpu",
        None => "unavailable",
    }
}

/// `barracuda.precision.route` — Precision routing advisory.
///
/// Routes a physics domain to the recommended precision tier based on
/// the current hardware capabilities and coralReef availability.
///
/// # Wire contract
///
/// ```json
/// { "method": "precision.route",
///   "params": { "domain": "lattice_qcd" } }
/// ```
///
/// Returns:
/// ```json
/// { "recommended_tier": "DF64", "fma_safe": true,
///   "requires_compiler": true, "hardware_hint": "compute",
///   "dispatch_path": "wgpu" }
/// ```
pub(super) fn precision_route(
    primal: &BarraCudaPrimal,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
    let Some(domain_str) = params.get("domain").and_then(Value::as_str) else {
        return JsonRpcResponse::error(
            id,
            -32602,
            "Missing required parameter: \"domain\" (string)",
        );
    };

    let Some(domain) = parse_domain(domain_str) else {
        return JsonRpcResponse::error(
            id,
            -32602,
            format!(
                "Unknown domain: \"{domain_str}\". Valid domains: lattice_qcd, gradient_flow, \
                 dielectric, kinetic_fluid, eigensolve, molecular_dynamics, nuclear_eos, \
                 population_pk, bioinformatics, hydrology, statistics, general, inference, \
                 training, hashing"
            ),
        );
    };

    let dispatch_path = resolve_dispatch_path(primal);

    if let Some(wgpu_dev) = primal.device() {
        let coral_available = barracuda::device::coral_compiler::is_coral_available();
        let brain = PrecisionBrain::from_device_with_coral(&wgpu_dev, coral_available);
        let advice = brain.route_advice(domain);
        let tier = advice.tier;

        return JsonRpcResponse::success(
            id,
            serde_json::json!({
                "recommended_tier": tier.to_string(),
                "fma_safe": advice.fma_safe,
                "requires_compiler": tier.requires_compiler_support(),
                "hardware_hint": hint_to_str(tier.recommended_hardware_hint()),
                "dispatch_path": dispatch_path,
                "rationale": advice.rationale,
                "needs_sovereign_compile": brain.needs_sovereign_compile(domain),
                "adapter": brain.adapter_name(),
            }),
        );
    }

    // No GPU: return the domain's minimum tier requirement as advisory.
    let min_tier = domain.minimum_tier();
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "recommended_tier": min_tier.to_string(),
            "fma_safe": !domain.fma_sensitive(),
            "requires_compiler": min_tier.requires_compiler_support(),
            "hardware_hint": hint_to_str(min_tier.recommended_hardware_hint()),
            "dispatch_path": dispatch_path,
            "rationale": "No GPU available — returning domain minimum tier requirement",
            "needs_sovereign_compile": false,
            "adapter": null,
        }),
    )
}
