# LKlight-GPU v1.2.1

**Apple Silicon Metal GPU backend (`--features metal`) — same-source, portable,
bit-consistent acceleration for macOS arm64.**

## Added

- **Metal GPU batch scorer for Apple Silicon (`feature = "metal"`, macOS only).**
  New `src/metal/lk_metal.m` (Objective-C host + embedded MSL kernels, C ABI
  `lk_metal_available/create/score/destroy`) and `src/metal_score.rs` (Rust
  binding mirroring `gpu_score.rs`), wired into the engine scoring path with a
  three-tier fallback **metal → cuda → CPU grid**.
  - The per-atom near-field (10 Å cell-list with clamped electrostatics,
    Lennard-Jones and clash terms) and far-field (0.5 Å potential, trilinear
    interpolation) kernels are a line-for-line MSL port of the CUDA
    `full_score.cu`/`far_field.cu` numerics used by the Windows/Linux GPU
    engines, so GPU behaviour is consistent across vendors.
  - Apple GPUs have no fast fp64, so rigid-body pose transforms stay on the CPU
    in f64 and the transformed coordinates are uploaded to a shared `MTLBuffer`
    (zero-copy on unified memory); only the per-atom f32 accumulation runs on
    the GPU — the same f32 quantisation budget as the CUDA backend.
  - Layout: one threadgroup per pose × 128 threads with a strided atom loop and
    tree reduction (the tuned POC configuration), 14 packed buffers total.
  - If no Metal device is available (or the build lacks the feature) the engine
    transparently falls back to the CPU grid path; the fallback result is
    bit-identical to the standalone LKlight-grid engine (verified).
  - No default-build change: plain `cargo build --release` output is unchanged
    from v1.2.0 (CPU-only); Metal is opt-in via `--features metal`.

## Measured (Mac mini M4, 10-core, 32 GB, unified memory; 1AZP, `dna`,
1000 glowworms × 1000 steps, seed 324324 — same-slot runs)

- **End-to-end 6.22 s** (≈ 6.2 ms/step) vs CPU grid 47.10 s and exact
  219.29 s on the same machine → **≈ 7.6× vs CPU grid, ≈ 35.3× vs exact**.
- Best energy **−7254.95161080**, bit-identical to the RTX 5090 CUDA engine
  final value (both run the same-source f32 batch kernel); deep-pose relative
  deviation vs CPU grid stays in the documented ~1e-4 f32 class.
- Binary depends only on system frameworks (`Metal`, `Foundation`, `objc`,
  `libSystem`) — ad-hoc linker-signed, no absolute paths or brew libraries, so
  it can be copied to any Apple Silicon Mac and run as-is.

## Developer-only

- `dump_metal_dataset` (env-gated in the `Score` trait) and the `poc_dump` bin
  export the exact dataset (receptor / cell-list / far-field / N transformed
  poses + per-pose CPU reference scores) used to develop and validate the Metal
  kernels. Not used by the normal engine path.
