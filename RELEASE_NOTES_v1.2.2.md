# LKlight-GPU v1.2.2 — GPU batching for all four all-atom scoring families (CUDA + Metal)

## Highlights

- **All four all-atom scoring families now run on the GPU batch kernel, on both
  backends** (NVIDIA CUDA and Apple-Silicon Metal): `dna`, `vdw`, `pydock`,
  `cpydock`. The kernels are family-parameterised — a per-family flag word
  selects the far-field / electrostatics / clash terms (dna = all three,
  pydock = far+elec, vdw = LJ only) — and the `dna` path keeps its original
  arithmetic bit-for-bit.
- **cpydock desolvation on the GPU**: a dedicated two-stage kernel
  (per-pose atomic-min over the receptor cell list, then a reduce of
  g(min)·des) implements the C-binary-compatible contact-SASA desolvation
  term; the host combines score = −(E·332/4 + 0.1·V − S) from a **single**
  main-kernel dispatch (E/V components are read unfolded).
- **Metal backend (Apple Silicon)** shares the same family semantics via an
  MSL port (`atomic_fetch_min_explicit` for the min stage); the `dna` best
  energy is bit-identical to the NVIDIA backend.

## Real-machine acceptance (1,000 glowworms × 1,000 steps, 1AZP, seed 324324)

| family | host / backend | GPU (s) | CPU grid (s) | speedup | best Δ |
|---|---|---|---|---|---|
| dna | Mac mini M4 / Metal | 6.99 | 47.10 | 6.7× | bit-identical* |
| vdw | Mac mini M4 / Metal | 5.42 | 36.59 | 6.8× | 2.5e-4 |
| pydock | Mac mini M4 / Metal | 7.02 | 43.11 | 6.1× | 3.5e-4 |
| cpydock | Mac mini M4 / Metal | 26.92 | 202.59 | 7.5× | 1.5e-4 |
| cpydock | Linux RTX 3080 Ti / CUDA | 9.83 | 273.16 | 27.8× | 4.5e-5 |
| dna | Windows RTX 3080 Ti / CUDA | 6.76 | 61.93 | 9.2× | 2.7e-4 |
| vdw | Windows RTX 3080 Ti / CUDA | 2.64 | 36.53 | 13.8× | 4.0e-5 |
| pydock | Windows RTX 3080 Ti / CUDA | 6.75 | 50.55 | 7.5× | 5.3e-5 |
| cpydock | Windows RTX 3080 Ti / CUDA | 8.79 | 239.78 | 27.3× | see note |

\* dna keeps the Table-4 v1.2.1 Metal-row figure (6.22 s on a separate host).
Note: per-step scoring agrees with the CPU grid at the f32 level on every
family (step-1/step-10 corr = 1.0). GSO is a deterministic chaotic system —
on long runs the f32-scale per-step differences can be amplified, so the
converged best energy may land in a different basin (the Windows cpydock GPU
run actually found the *lower* energy, −86.46 vs −78.90). This is an
optimiser property, not a scoring difference.

## Binaries (single file, no runtime deps beyond the NVIDIA driver)

- `LKlight-win64-cuda.exe` — Windows x64 (built with VS BuildTools 2022 +
  CUDA 12.6.2, static cudart; dumpbin-verified no CUDA DLL dependency)
- `LKlight-linux-cuda` — Linux x86-64 (static-pie)
- `LKlight-mac-arm64-metal` — macOS arm64 Apple Silicon (Metal backend,
  system frameworks only)
- `LKlight-mac-arm64-cpu` — macOS arm64 CPU grid build

## Upgrade notes

- CLI, data formats and the CPU-grid fallback are unchanged.
- Restraints / membrane / ANM runs keep using the CPU grid path (identical
  scoring, by design).
- The eight table/statistical-potential families (dfire, dfire2, mj3h, pisa,
  sipper, tobi, sd, ddna) remain on the CPU grid path — they are already
  sub-second and GPU batching would be a net loss.
