# LKlight-GPU v1.2.0

**0.5 Å far-field grid (reference-level accuracy) + GPU batch-scoring
performance optimisations + faster grid setup.**

## Added

- **Far-field grid at 0.5 Å reference resolution (new default).** The host-side
  far-field grid (`src/grid_dna.rs`, byte-identical with LKlight-grid) is built
  at 0.5 Å spacing; the box half-width is `spread = ceil(FIELD_RMAX / spacing) + 2`
  (the old hard-coded ±32 cells assumed 1 Å). Because the CUDA far-field kernels
  are fully parameterised by the grid (spacing / origin / nx,ny,nz / phi) with no
  spacing assumption baked into device code, the GPU shares the accuracy gain
  with **no kernel change**. Verified on Linux RTX 3080 Ti and Windows RTX 5090:
  worst-case per-pose deviation vs. the reference engine drops ~9.6 → ~6.9
  (~1.4×).

## Changed

- **Per-block shared-memory reduction of pose atomics** (`full_score.cu`):
  each thread block reduces per-atom electrostatic/VDW terms into `extern
  __shared__` memory and performs one `atomicAdd` per block via a single thread
  (stride-halving tree reduction), instead of every ligand atom doing a double
  `atomicAdd`. Per-pose atomic traffic drops from ~`nl` to ~`ceil(nl/256)`.
  Out-of-range threads store 0 and still hit the barrier. GSO convergence is
  unchanged (1AZP `dna` best −7254.95161 GPU vs −7254.95220 optimised CPU-grid,
  Δ ≈ 5.9×10⁻⁴, within the f32 batch budget).
- **Persistent device-buffer cache** (`full_score.cu`): `cuda_batch_score` keeps
  a process-static `BatchCache` holding resident device copies of the receptor
  field, constant tables and grid parameters, keyed by host pointers + dims. On
  a hit only the N×7 pose translations/rotations are uploaded and N×2 energies
  read back each call; a new receptor (pointer-key change) frees + rebuilds.
  Eliminates repeated field upload across thousands of GSO scoring calls.
- **Shell-band box scan for faster grid setup (bit-identical).** Same shared
  `grid_dna.rs` pruning as LKlight-grid: written-cell set/order and the `f32`
  field are unchanged; grid build + first score 1.57 s → 1.02 s at 0.5 Å.
- **Ligand C-cache pointer stabilisation (performance + robustness fix).**
  `batch_energy_gpu_scores` rebuilt the ligand parameter vectors on every call,
  so the C-side batch-cache key (base-pointer) never matched and every step did
  a full rebuild (repeated `cudaMalloc`/`cudaFree` + full 0.5 Å field upload).
  Ligand arrays are now cached once per system (`DNA.lig_cuda`, `OnceLock`,
  `cfg(cuda)`) with stable pointers, so the persistent device cache actually
  hits step-to-step. Also clears any sticky CUDA error at the batch entry so a
  single failed step can no longer poison later launches, and reports numeric
  error codes in diagnostics. Measured on Windows RTX 5090, 1000 glowworms ×
  1000 steps, 1AZP `dna`: GPU wall-clock **267.6 s → 4.05 s (~66×)**, batch
  errors → 0, best energy unchanged (−7254.9516108, bit-identical).

## Accuracy contract

- `pydock` / `cpydock` GPU == CPU **bit-identical**; `dna` differs only by the
  expected f32 near-field batch rounding (≤ ~6×10⁻⁴).

## Downloads

| Platform | File | Binary |
|---|---|---|
| Linux x86-64 | `LKlight-linux-cuda` | CUDA-enabled, glibc |
| Windows x64 | `LKlight-win64-cuda.exe` | CUDA-enabled, PE32+ console |

Static cudart: the target machine only needs the NVIDIA driver. Bare binaries
follow the v1.1.0 asset layout; each repo tree carries `README.md`, `LICENSE`
(GPL-3.0), `NOTICE` and `CHANGELOG.md`.

## Citation & DOI

Zenodo DOI from v1.1.0 is retained. See `NOTICE` for LightDock attribution.
