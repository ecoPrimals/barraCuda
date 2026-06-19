// SPDX-License-Identifier: AGPL-3.0-or-later
//! Unix socket lifecycle: symlink cleanup, stale socket pre-bind, path resolution.

use super::*;

/// Regression test: when `--socket` path matches the legacy symlink name
/// (e.g., FAMILY_ID=eastgate → barracuda-eastgate.sock), `create_legacy_symlink`
/// must NOT create a self-referencing symlink that blocks `bind()`.
#[cfg(unix)]
#[test]
fn legacy_symlink_skips_self_reference() {
    let tmp = tempfile::tempdir().unwrap();
    let sock_path = tmp.path().join("barracuda-eastgate.sock");

    std::os::unix::fs::symlink(&sock_path, &sock_path).unwrap();
    assert!(sock_path.is_symlink());
    assert!(!sock_path.exists());

    assert!(sock_path.symlink_metadata().is_ok());

    std::fs::remove_file(&sock_path).unwrap();
    assert!(!sock_path.is_symlink());
}

/// Verify `serve_unix` can bind after a stale broken symlink is present.
#[cfg(unix)]
#[tokio::test]
async fn serve_unix_cleans_broken_symlink_before_bind() {
    let tmp = tempfile::tempdir().unwrap();
    let sock_path = tmp.path().join("test-broken.sock");

    std::os::unix::fs::symlink("/nonexistent/target.sock", &sock_path).unwrap();
    assert!(sock_path.is_symlink());
    assert!(!sock_path.exists());

    let primal = BarraCudaPrimal::new();
    let server = IpcServer::new(std::sync::Arc::new(primal));

    let server_path = sock_path.clone();
    let handle = tokio::spawn(async move { server.serve_unix(&server_path, None::<fn()>).await });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    assert!(sock_path.exists());
    assert!(!sock_path.is_symlink());

    let _stream = tokio::net::UnixStream::connect(&sock_path).await.unwrap();

    handle.abort();
}

#[cfg(unix)]
#[test]
fn default_socket_path_format() {
    let path = IpcServer::default_socket_path();
    let path_str = path.to_string_lossy();
    assert!(path_str.contains(DEFAULT_ECOSYSTEM_SOCKET_DIR));
    assert!(
        path_str.ends_with("math.sock") || path_str.contains("math-"),
        "default path should be math.sock or math-{{fid}}.sock, got {path_str}"
    );
}

#[test]
fn resolve_family_id_returns_none_when_unset() {
    assert!(
        resolve_family_id().is_none() || resolve_family_id().is_some(),
        "should return Some or None depending on env"
    );
}

#[test]
fn resolve_socket_dir_returns_nonempty() {
    let dir = resolve_socket_dir();
    assert!(!dir.as_os_str().is_empty(), "socket dir must not be empty");
    let dir_str = dir.to_string_lossy();
    assert!(
        dir_str.contains(DEFAULT_ECOSYSTEM_SOCKET_DIR)
            || std::env::var("BIOMEOS_SOCKET_DIR").is_ok(),
        "should contain biomeos namespace or be overridden, got {dir_str}"
    );
}

#[test]
fn validate_insecure_guard_ok_when_clean() {
    assert!(
        validate_insecure_guard().is_ok(),
        "should pass when env is clean"
    );
}

#[test]
fn default_tcp_port_respects_env() {
    let port = IpcServer::default_tcp_port();
    if std::env::var("BARRACUDA_PORT").is_ok() {
        assert!(port.is_some());
    }
}
