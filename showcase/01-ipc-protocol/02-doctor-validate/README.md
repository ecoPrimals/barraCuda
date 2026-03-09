# 02 — Doctor and Validate

Demonstrates barraCuda's self-diagnostic capabilities.

## What It Shows

- `barracuda doctor` — Health check: GPU detection, driver info, feature support
- `barracuda validate` — GPU validation canary: dispatches test shaders, verifies results
- `barracuda version` — Build and version information
- Production readiness: these commands run in CI and on deployment

## Prerequisites

Build the barraCuda binary first:

```bash
cd ../../.. && cargo build --release -p barracuda-core
```

## Run

```bash
./demo.sh
```
