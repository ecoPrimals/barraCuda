// SPDX-License-Identifier: AGPL-3.0-or-later
//! Native binary cache for coralReef-compiled GPU binaries.

use super::types::CoralBinary;

/// Cache of native GPU binaries produced by coralReef, keyed by
/// (blake3 hash of shader source, target arch).
static NATIVE_BINARY_CACHE: std::sync::LazyLock<
    std::sync::RwLock<std::collections::HashMap<(String, String), CoralBinary>>,
> = std::sync::LazyLock::new(|| std::sync::RwLock::new(std::collections::HashMap::new()));

/// Look up a cached native binary for the given shader source and arch.
#[must_use]
pub fn cached_native_binary(shader_hash: &str, arch: &str) -> Option<CoralBinary> {
    let cache = NATIVE_BINARY_CACHE
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    cache
        .get(&(shader_hash.to_owned(), arch.to_owned()))
        .cloned()
}

/// Store a native binary in the cache.
pub fn cache_native_binary(shader_hash: &str, arch: &str, binary: CoralBinary) {
    let mut cache = NATIVE_BINARY_CACHE
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    cache.insert((shader_hash.to_owned(), arch.to_owned()), binary);
}

/// Find any cached native binary matching this shader hash, regardless of arch/adapter key.
///
/// Used when the caller does not know the exact compilation target — for example,
/// the sovereign dispatch path in `SovereignDevice` which receives binaries from
/// any architecture supported by the shader compiler primal.
#[must_use]
pub fn cached_native_binary_any_arch(shader_hash: &str) -> Option<CoralBinary> {
    let cache = NATIVE_BINARY_CACHE
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    cache
        .iter()
        .find(|((hash, _), _)| hash == shader_hash)
        .map(|(_, binary)| binary.clone())
}

/// Hash shader source for cache keying (uses blake3 for consistency with barraCuda).
#[must_use]
pub fn shader_hash(source: &str) -> String {
    blake3::hash(source.as_bytes()).to_hex().to_string()
}
