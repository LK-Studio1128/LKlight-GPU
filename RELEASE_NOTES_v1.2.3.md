# LKlight-GPU v1.2.3

**Nucleic-acid scoring safety patch for GPU/Metal batching.** This patch
prevents protein–DNA / RNA jobs from completing with an unsupported scorer and
a misleading zero score.

## Fixed

- Nucleic-acid inputs are validated before CPU or GPU scoring. `dna`, `ddna`,
  `pydock`, `cpydock`, and `vdw` are allowed; incompatible methods fail clearly
  with exit code 2.
- Worker-thread failures are reflected in the process exit status.
- GPU test fixtures are initialized with the current cache/device fields, and
  the CUDA DNA regression uses the same relative tolerance as the CPU grid.

## Validation

- `cargo test --release --lib`: **36 passed** on the CPU fallback build.
- Real protein–DNA regression: macOS CPU/Metal **-5186.346373**; Linux CUDA
  **-5186.345859** with `CUDA full-pose scoring ACTIVE`.
- A 1×20×20 Linux CUDA docking produced ranked scores **37.411519–49.236627**
  (span 11.825108), confirming non-constant scoring.
- Unsupported scorers are rejected with exit code 2 instead of returning `-0`.

## Binaries

- `LKlight-GPU-v1.2.3-linux-x86_64` — Linux x86-64, CUDA
- `LKlight-GPU-v1.2.3-mac-arm64-metal` — macOS Apple Silicon, Metal
- `LKlight-GPU-v1.2.3-mac-arm64-cpu` — macOS Apple Silicon, CPU fallback
- `LKlight-GPU-v1.2.3-win64-cuda.exe` — Windows x64, CUDA
