# Model Serialization Format Evolution — P3 Design

**Status**: Prototype implemented (bincode + BLAKE3 header, Wave 76)  
**Priority**: P3 → promoted to P2 (prototype done, awaiting biomeOS shadow mode)  
**Date**: 2026-06-04  
**Wire methods**: `ml.mlp_save`, `ml.mlp_load`

---

## Current State (Wave 75)

`ml.mlp_save` serializes `SimpleMlp` as JSON via `serde_json`. This is:
- Human-readable (debuggable, inspectable)
- Well-tested (roundtrip tests passing)
- Larger than necessary for production (float precision overhead)
- Adequate for development and shadow mode

## Production Requirements

When biomeOS moves from shadow mode to live routing:
- Model files transfer cross-gate over TCP (smaller = faster)
- Deserialization speed matters for cold-start inference
- Integrity verification needed (Dark Forest Invariant 3)

## Format Candidates

### Option A: bincode

- **Pro**: Zero-copy deserialization possible, smallest wire size (~2x smaller than JSON for float-heavy data), fastest deserialize, pure Rust, mature crate
- **Pro**: Already used in ecosystem (plasmidBin artifact format)
- **Con**: Not human-readable, schema migration requires versioning headers
- **Size estimate**: 36×2 weights (f64) + 2 biases + metadata ≈ 600 bytes vs ~2.4KB JSON

### Option B: CBOR (RFC 8949)

- **Pro**: Binary but self-describing (tagged types), compact, schema-tolerant
- **Pro**: Standard format, multiple language implementations
- **Con**: Slightly larger than bincode (~10-15% overhead from type tags)
- **Con**: Additional dependency (`ciborium` crate)
- **Size estimate**: ~700 bytes for same model

### Option C: MessagePack

- **Pro**: Binary, compact, widely supported, schema-tolerant
- **Con**: Additional dependency (`rmp-serde`)
- **Con**: Ecosystem doesn't use it elsewhere

## Recommended Approach

**bincode** — aligned with ecosystem precedent (plasmidBin) and optimizes for the primary constraint (cross-gate transfer speed and cold-start latency).

## Wire Contract Evolution

```json
// ml.mlp_save with format selection (backward compatible)
{
  "model": {...},
  "path": "/data/gate/neural_routing_perceptron.bin",
  "format": "bincode"  // NEW: optional, default "json" for backward compat
}

// ml.mlp_load auto-detects format from magic bytes
{
  "path": "/data/gate/neural_routing_perceptron.bin"
}
```

## File Format Header

```
┌─────────────────────────────────────────┐
│ Magic: "BCML" (4 bytes)                 │
│ Version: u8 (1 = initial)               │
│ Format: u8 (0=json, 1=bincode, 2=cbor)  │
│ Reserved: [u8; 2]                       │
│ Payload length: u32 (LE)                │
│ BLAKE3 checksum: [u8; 32]              │
├─────────────────────────────────────────┤
│ Payload (bincode-encoded SimpleMlp)     │
└─────────────────────────────────────────┘
```

Total header: 44 bytes. Enables format auto-detection and integrity verification.

## BTSP Signing (Phase 3+)

When bearDog BTSP Phase 3 signing is available:
- Append Ed25519 signature after payload
- Verifying gate checks signature before loading
- Prevents model tampering in transit (Dark Forest Invariant 3)

## Implementation Checklist

- [x] Add `bincode` dependency to `barracuda` crate (Wave 76)
- [x] Implement `SimpleMlp::to_binary()` / `from_binary()` (Wave 76)
- [x] Add format header parsing to `ml.mlp_load` via `from_auto()` (Wave 76)
- [x] Add `"format"` param to `ml.mlp_save` (default: "json") (Wave 76)
- [x] BLAKE3 checksum on write, verify on read (Wave 76)
- [x] Backward compat: bare JSON files (no header) still load (Wave 76)
- [x] Integration test: save as bincode, load from bincode (Wave 76)
- [ ] Cross-gate test: transfer model file, verify checksum
- [ ] BTSP signing integration (Phase 3+)

## Dependencies

- `bincode = "2"` (stable, no-std compatible)
- `blake3` (already in ecosystem via bearDog)
