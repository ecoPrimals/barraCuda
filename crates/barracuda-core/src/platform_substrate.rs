// SPDX-License-Identifier: AGPL-3.0-or-later
//! G68 Platform Substrate Abstraction — per sourDough reference pattern.
//!
//! Provides platform-neutral wrappers for OS-specific operations so business
//! logic never touches `std::os::unix` or `std::os::windows` directly.
//!
//! barraCuda uses only L1 (links) — L2 (permissions) and L3 (device backends)
//! are not needed since barraCuda has no direct permission manipulation or
//! device I/O outside the wgpu abstraction layer.

use std::io;
use std::path::Path;

/// Create a platform-appropriate link from `original` to `link`.
///
/// - **Unix**: symbolic link
/// - **Windows**: symlink (file or dir) with hard-link fallback
/// - **Other**: hard link
///
/// # Errors
///
/// Returns `io::Error` if the link creation fails.
pub fn platform_link(original: &Path, link: &Path) -> io::Result<()> {
    platform_link_impl(original, link)
}

#[cfg(unix)]
fn platform_link_impl(original: &Path, link: &Path) -> io::Result<()> {
    std::os::unix::fs::symlink(original, link)
}

#[cfg(windows)]
fn platform_link_impl(original: &Path, link: &Path) -> io::Result<()> {
    if original.is_dir() {
        std::os::windows::fs::symlink_dir(original, link)
    } else {
        std::os::windows::fs::symlink_file(original, link)
            .or_else(|_| std::fs::hard_link(original, link))
    }
}

#[cfg(not(any(unix, windows)))]
fn platform_link_impl(original: &Path, link: &Path) -> io::Result<()> {
    std::fs::hard_link(original, link)
}

/// Check if a path is a symbolic link (platform-aware).
#[must_use]
pub fn is_symlink(path: &Path) -> bool {
    std::fs::symlink_metadata(path)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn platform_link_creates_readable_link() {
        let dir = TempDir::new().unwrap();
        let original = dir.path().join("original.txt");
        std::fs::write(&original, "hello").unwrap();

        let link_path = dir.path().join("link.txt");
        platform_link(&original, &link_path).unwrap();

        let content = std::fs::read_to_string(&link_path).unwrap();
        assert_eq!(content, "hello");
    }

    #[cfg(unix)]
    #[test]
    fn platform_link_is_symlink_on_unix() {
        let dir = TempDir::new().unwrap();
        let original = dir.path().join("orig.txt");
        std::fs::write(&original, "data").unwrap();

        let link_path = dir.path().join("sym.txt");
        platform_link(&original, &link_path).unwrap();

        assert!(is_symlink(&link_path));
    }

    #[test]
    fn is_symlink_false_for_regular_file() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("regular.txt");
        std::fs::write(&file, "x").unwrap();
        assert!(!is_symlink(&file));
    }

    #[test]
    fn is_symlink_false_for_nonexistent() {
        assert!(!is_symlink(Path::new("/nonexistent/g68_test_path")));
    }
}
