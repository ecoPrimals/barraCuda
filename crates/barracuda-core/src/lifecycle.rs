// SPDX-License-Identifier: AGPL-3.0-or-later
//! Primal lifecycle management.
//!
//! Modeled on the ecoPrimals pattern (sourDough scaffold), owned by barraCuda.
//! Every mature primal internalizes its lifecycle — this is barraCuda's.

use std::fmt;

/// State of the barraCuda primal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimalState {
    /// Not yet started.
    Created,
    /// Starting up (discovering GPU, compiling shaders).
    Starting,
    /// Running normally.
    Running,
    /// Stopping (releasing device handles).
    Stopping,
    /// Stopped.
    Stopped,
    /// Failed.
    Failed,
}

impl PrimalState {
    /// Check if the primal is running.
    #[must_use]
    pub const fn is_running(&self) -> bool {
        matches!(self, Self::Running)
    }

    /// Check if the primal can be started.
    #[must_use]
    pub const fn can_start(&self) -> bool {
        matches!(self, Self::Created | Self::Stopped | Self::Failed)
    }

    /// Check if the primal can be stopped.
    #[must_use]
    pub const fn can_stop(&self) -> bool {
        matches!(self, Self::Running)
    }
}

impl fmt::Display for PrimalState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Created => write!(f, "created"),
            Self::Starting => write!(f, "starting"),
            Self::Running => write!(f, "running"),
            Self::Stopping => write!(f, "stopping"),
            Self::Stopped => write!(f, "stopped"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

/// Lifecycle trait for the barraCuda primal.
///
/// Defines how barraCuda starts (GPU discovery, device init) and stops
/// (resource cleanup). Modeled on the ecoPrimals lifecycle pattern.
pub trait PrimalLifecycle: Send + Sync {
    /// Get the current state.
    fn state(&self) -> PrimalState;

    /// Start the primal (discover GPU, initialize device pool).
    ///
    /// # Errors
    ///
    /// Returns an error if startup fails.
    fn start(
        &mut self,
    ) -> impl std::future::Future<Output = Result<(), crate::error::BarracudaCoreError>> + Send;

    /// Stop the primal (release device handles, clean up).
    ///
    /// # Errors
    ///
    /// Returns an error if shutdown fails.
    fn stop(
        &mut self,
    ) -> impl std::future::Future<Output = Result<(), crate::error::BarracudaCoreError>> + Send;

    /// Reload: stop then start.
    ///
    /// # Errors
    ///
    /// Returns an error if reload fails.
    fn reload(
        &mut self,
    ) -> impl std::future::Future<Output = Result<(), crate::error::BarracudaCoreError>> + Send
    {
        async {
            self.stop().await?;
            self.start().await
        }
    }

    /// Handle a shutdown signal. Default calls `stop()`.
    ///
    /// # Errors
    ///
    /// Returns an error if shutdown fails.
    fn shutdown(
        &mut self,
    ) -> impl std::future::Future<Output = Result<(), crate::error::BarracudaCoreError>> + Send
    {
        async { self.stop().await }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_transitions() {
        assert!(PrimalState::Created.can_start());
        assert!(!PrimalState::Created.can_stop());
        assert!(!PrimalState::Created.is_running());

        assert!(!PrimalState::Running.can_start());
        assert!(PrimalState::Running.can_stop());
        assert!(PrimalState::Running.is_running());

        assert!(PrimalState::Stopped.can_start());
        assert!(!PrimalState::Stopped.can_stop());

        assert!(PrimalState::Failed.can_start());
        assert!(!PrimalState::Failed.can_stop());
    }

    #[test]
    fn state_display() {
        assert_eq!(PrimalState::Created.to_string(), "created");
        assert_eq!(PrimalState::Running.to_string(), "running");
        assert_eq!(PrimalState::Failed.to_string(), "failed");
    }
}
