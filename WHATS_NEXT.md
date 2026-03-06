# barraCuda — What's Next

Prioritized work items, ordered by impact. Updated 2026-03-06.

---

## Immediate (P1)

- **DF64 NVK end-to-end verification**: Run `compile_shader_universal(Precision::Df64)` on
  Yukawa force kernels through NVK/NAK on hardware. Validate SPIR-V passthrough path
  produces correct numerical results, not just valid WGSL. Probe-aware `fp64_strategy()`
  is now in place to auto-fallback if native f64 fails.
- **coralNAK extraction**: When org repo fork lands, apply `specs/coralnak/SCAFFOLD_PLAN.md`
  to create the sovereign NVIDIA shader compiler primal.

## Near-term (P2)

- **Test coverage to 90%**: Evolve CI `--fail-under` from 80 to 90. Add GPU-conditional
  tests for new ops (SCS-CN, Stewart, Blaney-Criddle, autocorrelation).
- **Kokkos validation baseline**: Document `sarkas_gpu` validation results, extract PPPM
  shader performance numbers for apples-to-apples comparison.
- **Kokkos GPU parity benchmarks**: Run barraCuda GPU benchmarks on matching hardware,
  publish comparison data.
- **WGSL optimizer annotation coverage**: Expand `@ilp_region` / `@unroll_hint` annotations
  across science shaders for architecture-specific ILP optimization.

## Medium-term (P3)

- **Multi-GPU dispatch**: Evolve GpuView to span multiple devices with automatic work
  distribution across primary/secondary adapters.
- **Pipeline cache persistence**: Extend `make_pipeline_cache` to load/save validated cache
  blobs from disk (requires safe blob validation layer).
- **Shader hot-reload**: File watcher for `.wgsl` files during development, automatic
  recompilation through sovereign pipeline.
- **Zero-copy evolution**: Pre-allocated buffers for `domain_ops.rs` CPU fallback clones,
  LSTM hidden state clones, RBF assembly allocations (see zero-copy audit).

## Long-term (P4)

- **Sovereign Compute Evolution**: Replace entire non-Rust GPU stack with coral-prefixed
  pure Rust components (coralNak, coralDriver, coralMem, coralQueue, coralGpu).
- **WebGPU browser target**: Compile barraCuda shaders for browser execution via wasm-pack
  and wgpu's WebGPU backend.
- **Distributed compute**: Cross-node GPU dispatch via primal-to-primal IPC for HPC clusters.
