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

## Accuracy contract

- `pydock` / `cpydock` GPU == CPU **bit-identical**; `dna` differs only by the
  expected f32 near-field batch rounding (≤ ~6×10⁻⁴).

## Downloads

| Platform | File | Binary |
|---|---|---|
| Linux x86-64 | `LKlight-GPU-v1.2.0-linux-x86_64.tar.gz` | CUDA-enabled, musl static-PIE |
| Windows x64 | `LKlight-GPU-v1.2.0-win-x64.zip` | CUDA-enabled, PE32+ console |

Each archive contains the binary plus `README.md`, `LICENSE` (GPL-3.0),
`NOTICE` and `CHANGELOG.md`.

## Citation & DOI

Zenodo DOI from v1.1.0 is retained. See `NOTICE` for LightDock attribution.
