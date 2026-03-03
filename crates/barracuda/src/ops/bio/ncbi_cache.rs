// SPDX-License-Identifier: AGPL-3.0-only

//! NCBI data cache module — local cache layer for sequence/annotation data.
//!
//! Capability-based, XDG-compliant. No network requests; actual NCBI API
//! calls belong in Spring code. Provenance: wetSpring V19 → toadStool L-008.

use std::path::{Path, PathBuf};

use etcetera::BaseStrategy;

use crate::error::{BarracudaError, Result};

/// Local cache for NCBI accession data (sequences, annotations, etc.).
///
/// Uses XDG cache dir by default (`$XDG_CACHE_HOME/barracuda/ncbi/` or
/// `~/.cache/barracuda/ncbi/`). Accepts explicit cache dir for tests.
#[derive(Debug, Clone)]
pub struct NcbiCache {
    cache_dir: PathBuf,
}

impl NcbiCache {
    /// Create a new cache. If `cache_dir` is `None`, uses XDG-compliant default.
    ///
    /// Default: `$XDG_CACHE_HOME/barracuda/ncbi/` or `~/.cache/barracuda/ncbi/`.
    pub fn new(cache_dir: Option<PathBuf>) -> Result<Self> {
        let base = match cache_dir {
            Some(p) => p,
            None => {
                let strategy = etcetera::choose_base_strategy()
                    .map_err(|e| BarracudaError::Internal(format!("XDG cache dir: {e}")))?;
                strategy.cache_dir()
            }
        };
        Ok(Self {
            cache_dir: base.join(env!("CARGO_PKG_NAME")).join("ncbi"),
        })
    }

    /// Returns the path where data for the given accession would be cached.
    pub fn cache_path(&self, accession: &str) -> PathBuf {
        self.cache_dir.join(sanitize_filename(accession))
    }

    /// Check if the accession data exists in cache.
    pub fn is_cached(&self, accession: &str) -> bool {
        validate_accession(accession).is_ok() && self.cache_path(accession).exists()
    }

    /// Write data to cache for the given accession.
    pub fn store(&self, accession: &str, data: &[u8]) -> Result<()> {
        validate_accession(accession)?;
        let path = self.cache_path(accession);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| BarracudaError::Internal(format!("create cache dir: {e}")))?;
        }
        std::fs::write(&path, data)
            .map_err(|e| BarracudaError::Internal(format!("write cache: {e}")))?;
        Ok(())
    }

    /// Read data from cache for the given accession.
    pub fn load(&self, accession: &str) -> Result<Vec<u8>> {
        validate_accession(accession)?;
        let path = self.cache_path(accession);
        std::fs::read(&path).map_err(|e| BarracudaError::Internal(format!("read cache: {e}")))
    }

    /// Remove all cached data.
    pub fn clear(&self) -> Result<()> {
        if self.cache_dir.exists() {
            std::fs::remove_dir_all(&self.cache_dir)
                .map_err(|e| BarracudaError::Internal(format!("clear cache: {e}")))?;
        }
        Ok(())
    }
}

/// Reject paths that could escape the cache directory.
fn validate_accession(accession: &str) -> Result<()> {
    if accession.is_empty() {
        return Err(BarracudaError::InvalidInput {
            message: "accession cannot be empty".to_string(),
        });
    }
    if accession.contains('/') || accession.contains('\\') {
        return Err(BarracudaError::InvalidInput {
            message: "accession cannot contain path separators".to_string(),
        });
    }
    if accession.contains("..") {
        return Err(BarracudaError::InvalidInput {
            message: "accession cannot contain path traversal".to_string(),
        });
    }
    if accession.contains('\0') {
        return Err(BarracudaError::InvalidInput {
            message: "accession cannot contain null byte".to_string(),
        });
    }
    if Path::new(accession).has_root() {
        return Err(BarracudaError::InvalidInput {
            message: "accession cannot be an absolute path".to_string(),
        });
    }
    Ok(())
}

/// Sanitize accession for use as filename (safe after validate_accession).
fn sanitize_filename(accession: &str) -> String {
    accession
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_and_load_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = NcbiCache::new(Some(dir.path().to_path_buf())).expect("new");
        let data = b"ACGTACGTACGT";
        cache.store("NM_001234", data).expect("store");
        let loaded = cache.load("NM_001234").expect("load");
        assert_eq!(loaded, data);
    }

    #[test]
    fn is_cached_false_before_store_true_after() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = NcbiCache::new(Some(dir.path().to_path_buf())).expect("new");
        assert!(!cache.is_cached("NM_001234"));
        cache.store("NM_001234", b"data").expect("store");
        assert!(cache.is_cached("NM_001234"));
    }

    #[test]
    fn path_traversal_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = NcbiCache::new(Some(dir.path().to_path_buf())).expect("new");
        let bad = ["../etc/passwd", "a/..", "..", "a/b", "a\\b"];
        for bad_acc in bad {
            assert!(
                cache.store(bad_acc, b"x").is_err(),
                "store should reject: {bad_acc}"
            );
            assert!(
                cache.load(bad_acc).is_err(),
                "load should reject: {bad_acc}"
            );
        }
    }

    #[test]
    fn clear_removes_cached_data() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = NcbiCache::new(Some(dir.path().to_path_buf())).expect("new");
        cache.store("NM_001234", b"data").expect("store");
        assert!(cache.is_cached("NM_001234"));
        cache.clear().expect("clear");
        assert!(!cache.is_cached("NM_001234"));
        assert!(cache.load("NM_001234").is_err());
    }

    #[test]
    fn default_cache_dir_uses_xdg() {
        let cache = NcbiCache::new(None).expect("new");
        let path = cache.cache_path("NM_001234");
        assert!(path.to_string_lossy().contains("barracuda"));
        assert!(path.to_string_lossy().contains("ncbi"));
        assert!(path.to_string_lossy().contains("NM_001234"));
    }

    #[test]
    fn empty_accession_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = NcbiCache::new(Some(dir.path().to_path_buf())).expect("new");
        assert!(cache.store("", b"x").is_err());
        assert!(cache.load("").is_err());
    }
}
