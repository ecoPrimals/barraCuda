# 01 — coralReef Shader Compilation

Demonstrates the barraCuda-to-coralReef shader compilation path.

## What It Shows

- barraCuda discovers coralReef via capability scan (`$XDG_RUNTIME_DIR/ecoPrimals/`)
- WGSL shader sent to coralReef for native GPU binary compilation
- Graceful degradation: if coralReef is unavailable, falls back to wgpu path
- Health probing and capability enumeration

## The Compute Pipeline

```
barraCuda                    coralReef
   |                            |
   |  shader.compile.wgsl       |
   |  { wgsl_source, arch }     |
   |--------------------------->|
   |                            | naga -> codegen -> native
   |  { binary, metadata }      |
   |<---------------------------|
   |                            |
   |  dispatch via wgpu or      |
   |  sovereign IPC (coralReef + toadStool) |
```

## Run

```bash
# Without coralReef (graceful degradation):
cargo run --release

# With coralReef running:
# (start coralReef in another terminal first)
cargo run --release
```
