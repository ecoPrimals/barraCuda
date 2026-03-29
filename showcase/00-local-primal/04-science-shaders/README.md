# 04 — Science Shaders

Demonstrates barraCuda's domain-specific GPU compute for real scientific
workloads: Hill kinetics for gene regulation, statistical metrics, and
tolerance architecture.

## What It Shows

- **Hill kinetics**: Activation and repression functions for gene regulatory networks
- **Statistical metrics**: GPU-accelerated dot product, L2 norm, Nash-Sutcliffe efficiency
- **13-tier tolerance architecture**: DETERMINISM through EQUILIBRIUM precision guards
- **Zero-copy uploads**: bytemuck::cast_slice for buffer transfers

## Domains Covered

barraCuda has 816 WGSL shaders across domains:
- Molecular dynamics (Yukawa, Morse, PPPM, Verlet)
- Spectral analysis (FFT, autocorrelation, IPR)
- Nuclear physics (HFB, Skyrme, BCS)
- Bio (HMM, DADA2, k-mer, phylogenetics)
- Optimization (BFGS, Nelder-Mead, CG)
- And more...

This showcase uses the CPU-accessible statistical functions to demonstrate
the scientific computing patterns without requiring GPU f64 support.

## Run

```bash
cargo run --release
```
