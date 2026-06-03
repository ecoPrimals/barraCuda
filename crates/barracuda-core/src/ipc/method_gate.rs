// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pre-dispatch capability gate for JSON-RPC methods (JH-0).
//!
//! Implements the ecosystem-standard `MethodGate` pattern per
//! `primalSpring/wateringHole/METHOD_GATE_STANDARD.md` v1.0.
//!
//! Every incoming RPC call passes through [`MethodGate::check`] *before*
//! the dispatch table. Public methods (health probes, identity, capability
//! advertisement, auth introspection) are always allowed. Protected methods
//! require a valid capability token when enforcement is active.
//!
//! Default mode: **Permissive** — log violations but allow all calls.

use serde_json::Value;

use super::jsonrpc::JsonRpcResponse;

/// Server-defined error code: permission denied (valid identity, lacking scope).
pub const PERMISSION_DENIED: i32 = -32_001;
/// Server-defined error code: unauthorized (identity not established).
pub const UNAUTHORIZED: i32 = -32_000;

/// Access level for a JSON-RPC method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MethodAccessLevel {
    /// Health probes, identity, capability advertisement — always allowed.
    Public,
    /// Requires a valid capability token when enforcement is active.
    Protected,
}

/// Methods that are always public (prefix match).
const PUBLIC_METHOD_PREFIXES: &[&str] = &["health.", "auth.", "mesh.", "btsp."];

/// Methods that are always public (exact match).
const PUBLIC_METHODS: &[&str] = &[
    "identity.get",
    "capabilities.list",
    "capability.list",
    "lifecycle.status",
    "primal.info",
    "primal.capabilities",
    "ping",
    "health",
    "status",
    "check",
];

/// Classify a method string into its access level.
#[must_use]
pub fn classify_method(method: &str) -> MethodAccessLevel {
    if PUBLIC_METHODS.contains(&method) {
        return MethodAccessLevel::Public;
    }
    for prefix in PUBLIC_METHOD_PREFIXES {
        if method.starts_with(prefix) {
            return MethodAccessLevel::Public;
        }
    }
    MethodAccessLevel::Protected
}

/// How the caller connected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionOrigin {
    /// Local Unix domain socket.
    Unix,
    /// TCP loopback (127.0.0.1 / ::1).
    Loopback,
    /// Remote TCP connection.
    Remote,
}

/// Peer credentials extracted from `SO_PEERCRED` on Unix sockets.
#[derive(Debug, Clone)]
pub struct PeerCredentials {
    /// Process ID of the caller (if available).
    pub pid: Option<u32>,
    /// User ID of the caller.
    pub uid: u32,
}

/// Identity and authorization context for an incoming RPC call.
#[derive(Debug, Clone)]
pub struct CallerContext {
    /// Optional bearer / capability token sent in the request.
    pub bearer_token: Option<String>,
    /// Peer credentials from `SO_PEERCRED` (Unix socket only).
    pub peer: Option<PeerCredentials>,
    /// Where the connection came from.
    pub origin: ConnectionOrigin,
}

impl CallerContext {
    /// Build a caller context for loopback TCP with no peer credentials.
    #[must_use]
    pub const fn loopback() -> Self {
        Self {
            bearer_token: None,
            peer: None,
            origin: ConnectionOrigin::Loopback,
        }
    }

    /// Build a caller context for a Unix domain socket connection.
    #[must_use]
    pub const fn unix() -> Self {
        Self {
            bearer_token: None,
            peer: None,
            origin: ConnectionOrigin::Unix,
        }
    }
}

/// Enforcement mode for the method gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnforcementMode {
    /// Log violations but allow all calls (backward-compatible default).
    Permissive,
    /// Reject unauthenticated calls to protected methods.
    Enforced,
}

impl EnforcementMode {
    /// Resolve from `BARRACUDA_AUTH_MODE` env var.
    /// Defaults to `Permissive` if unset or unrecognized.
    #[must_use]
    pub fn from_env() -> Self {
        match std::env::var(crate::env_keys::BARRACUDA_AUTH_MODE)
            .unwrap_or_default()
            .to_lowercase()
            .as_str()
        {
            "enforced" | "enforce" | "strict" => Self::Enforced,
            _ => Self::Permissive,
        }
    }

    /// Human-readable label for diagnostics and `auth.mode` responses.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Permissive => "permissive",
            Self::Enforced => "enforced",
        }
    }
}

/// Pre-dispatch gate that checks caller authorization before method execution.
#[derive(Debug)]
pub struct MethodGate {
    mode: EnforcementMode,
}

impl MethodGate {
    /// Create a gate with the given enforcement mode.
    #[must_use]
    pub const fn new(mode: EnforcementMode) -> Self {
        Self { mode }
    }

    /// Create a gate from the environment (`BARRACUDA_AUTH_MODE`).
    #[must_use]
    pub fn from_env() -> Self {
        Self::new(EnforcementMode::from_env())
    }

    /// Current enforcement mode.
    #[must_use]
    pub const fn mode(&self) -> EnforcementMode {
        self.mode
    }

    /// Pre-dispatch authorization check.
    ///
    /// Returns `Ok(())` if the call should proceed to the dispatch table.
    /// Returns `Err(JsonRpcResponse)` if the call is rejected.
    pub fn check(
        &self,
        method: &str,
        caller: &CallerContext,
        id: &Value,
    ) -> Result<(), JsonRpcResponse> {
        let level = classify_method(method);

        if level == MethodAccessLevel::Public {
            return Ok(());
        }

        let authorized = caller.bearer_token.is_some();
        if authorized {
            return Ok(());
        }

        match self.mode {
            EnforcementMode::Permissive => {
                tracing::warn!(
                    method,
                    caller_origin = ?caller.origin,
                    "method gate: unauthenticated call to protected method (permissive — allowing)"
                );
                Ok(())
            }
            EnforcementMode::Enforced => {
                tracing::warn!(
                    method,
                    caller_origin = ?caller.origin,
                    "method gate: REJECTED unauthenticated call to protected method"
                );
                Err(JsonRpcResponse::error(
                    id.clone(),
                    PERMISSION_DENIED,
                    format!("permission denied: method '{method}' requires a capability token"),
                ))
            }
        }
    }

    /// Handle the `auth.check` introspection method.
    #[must_use]
    pub fn handle_auth_check(&self, caller: &CallerContext, id: Value) -> JsonRpcResponse {
        let authenticated = caller.bearer_token.is_some();
        JsonRpcResponse::success(
            id,
            serde_json::json!({
                "authenticated": authenticated,
                "origin": format!("{:?}", caller.origin).to_lowercase(),
                "has_peer_credentials": caller.peer.is_some(),
            }),
        )
    }

    /// Handle the `auth.mode` introspection method.
    #[must_use]
    pub fn handle_auth_mode(&self, id: Value) -> JsonRpcResponse {
        JsonRpcResponse::success(
            id,
            serde_json::json!({
                "mode": self.mode.as_str(),
                "primal": crate::PRIMAL_NAME,
                "standard": "METHOD_GATE_STANDARD v1.0",
            }),
        )
    }

    /// Handle the `auth.peer_info` introspection method.
    #[must_use]
    pub fn handle_auth_peer_info(&self, caller: &CallerContext, id: Value) -> JsonRpcResponse {
        let peer_info = match &caller.peer {
            Some(creds) => serde_json::json!({
                "available": true,
                "uid": creds.uid,
                "pid": creds.pid,
            }),
            None => serde_json::json!({
                "available": false,
                "reason": "peer credentials not available (token-based auth or TCP connection)",
            }),
        };
        JsonRpcResponse::success(
            id,
            serde_json::json!({
                "origin": format!("{:?}", caller.origin).to_lowercase(),
                "peer": peer_info,
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_methods_are_public() {
        assert_eq!(classify_method("health.check"), MethodAccessLevel::Public);
        assert_eq!(
            classify_method("health.liveness"),
            MethodAccessLevel::Public
        );
        assert_eq!(
            classify_method("health.readiness"),
            MethodAccessLevel::Public
        );
    }

    #[test]
    fn identity_and_capabilities_are_public() {
        assert_eq!(classify_method("identity.get"), MethodAccessLevel::Public);
        assert_eq!(
            classify_method("capabilities.list"),
            MethodAccessLevel::Public
        );
        assert_eq!(
            classify_method("capability.list"),
            MethodAccessLevel::Public
        );
        assert_eq!(
            classify_method("primal.capabilities"),
            MethodAccessLevel::Public
        );
        assert_eq!(classify_method("primal.info"), MethodAccessLevel::Public);
    }

    #[test]
    fn auth_methods_are_public() {
        assert_eq!(classify_method("auth.check"), MethodAccessLevel::Public);
        assert_eq!(classify_method("auth.mode"), MethodAccessLevel::Public);
        assert_eq!(classify_method("auth.peer_info"), MethodAccessLevel::Public);
    }

    #[test]
    fn aliases_are_public() {
        assert_eq!(classify_method("ping"), MethodAccessLevel::Public);
        assert_eq!(classify_method("health"), MethodAccessLevel::Public);
        assert_eq!(classify_method("status"), MethodAccessLevel::Public);
        assert_eq!(classify_method("check"), MethodAccessLevel::Public);
    }

    #[test]
    fn math_methods_are_protected() {
        assert_eq!(
            classify_method("math.sigmoid"),
            MethodAccessLevel::Protected
        );
        assert_eq!(classify_method("stats.mean"), MethodAccessLevel::Protected);
        assert_eq!(
            classify_method("linalg.solve"),
            MethodAccessLevel::Protected
        );
    }

    #[test]
    fn tensor_methods_are_protected() {
        assert_eq!(
            classify_method("tensor.create"),
            MethodAccessLevel::Protected
        );
        assert_eq!(
            classify_method("tensor.matmul"),
            MethodAccessLevel::Protected
        );
    }

    #[test]
    fn ode_and_esn_are_protected() {
        assert_eq!(classify_method("ode.step"), MethodAccessLevel::Protected);
        assert_eq!(
            classify_method("ml.esn_predict"),
            MethodAccessLevel::Protected
        );
    }

    #[test]
    fn btsp_is_protected() {
        assert_eq!(
            classify_method("btsp.negotiate"),
            MethodAccessLevel::Protected
        );
    }

    #[test]
    fn unknown_methods_are_protected() {
        assert_eq!(
            classify_method("unknown.method"),
            MethodAccessLevel::Protected
        );
        assert_eq!(classify_method(""), MethodAccessLevel::Protected);
    }

    #[test]
    fn enforcement_mode_as_str() {
        assert_eq!(EnforcementMode::Permissive.as_str(), "permissive");
        assert_eq!(EnforcementMode::Enforced.as_str(), "enforced");
    }

    #[test]
    fn loopback_context_has_no_peer() {
        let ctx = CallerContext::loopback();
        assert!(ctx.peer.is_none());
        assert!(ctx.bearer_token.is_none());
        assert_eq!(ctx.origin, ConnectionOrigin::Loopback);
    }

    #[test]
    fn unix_context_has_no_peer() {
        let ctx = CallerContext::unix();
        assert!(ctx.peer.is_none());
        assert_eq!(ctx.origin, ConnectionOrigin::Unix);
    }

    #[test]
    fn public_method_always_passes_in_enforced_mode() {
        let gate = MethodGate::new(EnforcementMode::Enforced);
        let caller = CallerContext::loopback();
        let id = serde_json::json!(1);
        assert!(gate.check("health.check", &caller, &id).is_ok());
        assert!(gate.check("identity.get", &caller, &id).is_ok());
        assert!(gate.check("capabilities.list", &caller, &id).is_ok());
        assert!(gate.check("auth.mode", &caller, &id).is_ok());
    }

    #[test]
    fn protected_method_passes_in_permissive_mode() {
        let gate = MethodGate::new(EnforcementMode::Permissive);
        let caller = CallerContext::loopback();
        let id = serde_json::json!(1);
        assert!(gate.check("math.sigmoid", &caller, &id).is_ok());
    }

    #[test]
    fn protected_method_rejected_in_enforced_mode_without_token() {
        let gate = MethodGate::new(EnforcementMode::Enforced);
        let caller = CallerContext::loopback();
        let id = serde_json::json!(1);
        let result = gate.check("math.sigmoid", &caller, &id);
        assert!(result.is_err());
        let err_resp = result.unwrap_err();
        assert!(err_resp.error.is_some());
    }

    #[test]
    fn protected_method_passes_in_enforced_mode_with_token() {
        let gate = MethodGate::new(EnforcementMode::Enforced);
        let caller = CallerContext {
            bearer_token: Some("valid-token".to_owned()),
            peer: None,
            origin: ConnectionOrigin::Unix,
        };
        let id = serde_json::json!(1);
        assert!(gate.check("math.sigmoid", &caller, &id).is_ok());
    }

    #[test]
    fn auth_check_reports_unauthenticated() {
        let gate = MethodGate::new(EnforcementMode::Permissive);
        let caller = CallerContext::loopback();
        let resp = gate.handle_auth_check(&caller, serde_json::json!(1));
        assert!(resp.result.is_some());
        let result = resp.result.unwrap();
        assert_eq!(result["authenticated"], false);
    }

    #[test]
    fn auth_check_reports_authenticated() {
        let gate = MethodGate::new(EnforcementMode::Permissive);
        let caller = CallerContext {
            bearer_token: Some("tok".to_owned()),
            peer: None,
            origin: ConnectionOrigin::Loopback,
        };
        let resp = gate.handle_auth_check(&caller, serde_json::json!(1));
        let result = resp.result.unwrap();
        assert_eq!(result["authenticated"], true);
    }

    #[test]
    fn auth_mode_response() {
        let gate = MethodGate::new(EnforcementMode::Permissive);
        let resp = gate.handle_auth_mode(serde_json::json!(1));
        let result = resp.result.unwrap();
        assert_eq!(result["mode"], "permissive");
        assert_eq!(result["primal"], "barraCuda");
    }

    #[test]
    fn auth_peer_info_no_peer() {
        let gate = MethodGate::new(EnforcementMode::Permissive);
        let caller = CallerContext::loopback();
        let resp = gate.handle_auth_peer_info(&caller, serde_json::json!(1));
        let result = resp.result.unwrap();
        assert_eq!(result["peer"]["available"], false);
    }

    #[test]
    fn auth_peer_info_with_peer() {
        let gate = MethodGate::new(EnforcementMode::Permissive);
        let caller = CallerContext {
            bearer_token: None,
            peer: Some(PeerCredentials {
                pid: Some(1234),
                uid: 1000,
            }),
            origin: ConnectionOrigin::Unix,
        };
        let resp = gate.handle_auth_peer_info(&caller, serde_json::json!(1));
        let result = resp.result.unwrap();
        assert_eq!(result["peer"]["available"], true);
        assert_eq!(result["peer"]["uid"], 1000);
        assert_eq!(result["peer"]["pid"], 1234);
    }
}
