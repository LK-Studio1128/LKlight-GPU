# LKlight-GPU

**CUDA-batched docking engine — LKlight-GPU v1.2.0.** Full-featured molecular
docking for protein–nucleic-acid and protein–protein complexes with
**NVIDIA-GPU batching on Windows and Linux**. No GPU present? The same binary
silently falls back to the CPU grid path (functionally identical to
`LKlight-grid`).

LKlight is a high-performance Rust reimplementation of
[LightDock](https://github.com/bioinsilico/LightDock)'s GSO (glowworm swarm
optimisation) docking protocol. This project ships the **CUDA variant**: the
scoring kernel evaluates the ≤ 10 Å near term (cell-list scan, clamped
electrostatics, Lennard-Jones, clash penalty) **and** the 10–30 Å receptor
far-field in a single launch, batched over **all poses of the whole GSO step**
(gridDim.y = pose count), with the rigid-body coordinate transform done on the
device (only N×7 pose parameters are uploaded per step instead of every atom).

Both GPU binaries were built **and verified on real machines**: Windows CUDA on
an RTX 2080 (`CUDA BATCH ACTIVE`), Linux CUDA on an RTX 3080 Ti.

---

## Features

- **12 scoring functions** behind one `Score` trait, identical surface to the
  CPU build: `dfire`, `dfire2`, `dna`, `ddna`, `mj3h`, `pydock`, `cpydock`,
  `sd`, `pisa`, `sipper`, `tobi`, `vdw`.
- **GPU coverage**: `dna` (no restraints/membrane/ANM) uses the **CUDA batch
  kernel**; `vdw` / `pydock` / `cpydock` use the CPU grid path (already
  grid-accelerated 12–48×).
- **Automatic fallback in one binary, zero configuration**: no NVIDIA
  driver → CPU grid; ANM / restraints / membrane → CPU grid (the GPU kernel
  does not emit interface flags; the CPU grid path collects them, so
  restraint semantics stay correct — audit-verified 2026-09-03).
- **Everything else identical to LKlight-grid**: GSO with spatial-hash
  neighbour search, ANM / restraints / membrane support, all CLI commands,
  `tools/run_parallel.py` swarm parallelism, clash/contact audit tools.

## Accuracy & numerical contract

| Guarantee | Value |
|---|---|
| GPU batch vs CPU grid | < 1e-5 (f32 rounding); convergence statistically identical |
| CPU grid vs original | far-field interpolation only: ≤ 0.5 % bound poses; top-5 pose overlap 100 % (±2 Å) |
| Cross-platform reproducibility | same pose → identical score on Windows/Linux/macOS (vdw bit-identical) |

Since **v1.2.0** the far-field grid is built at **0.5 Å spacing**
(near-reference resolution) instead of 1.0 Å; the device kernels are fully
parameterised by the grid, so the GPU shares the accuracy gain with no kernel
change (worst-case per-pose deviation vs. the reference engine drops ~9.6 → ~6.9
on 1AZP). `pydock`/`cpydock` remain bit-identical between GPU and CPU. GPU
batch-scoring adds per-block shared-memory atomics reduction and a persistent
device-buffer cache to cut per-step upload overhead.

## Quick start

```bash
# Windows  (NVIDIA GPU + driver)
BIN=release_bin/LKlight-win64-cuda.exe
# Linux
BIN=release_bin/LKlight-linux-cuda

# Prepare: large-scale runs benefit from more glowworms
$BIN setup receptor.pdb ligand.pdb dna -s 6 -g 200 --seed 42 --noxt --now

# Run all swarms (parallel helper included)
python3 tools/run_parallel.py $BIN 100 dna 6 6

# Rank + export
$BIN rank 6 100
for i in 0 1 2 3 4 5; do $BIN generate lightdock_rec.pdb lightdock_lig.pdb swarm_$i/gso_100.out 200; done
```

**Verify the GPU is active** — the first successful scoring step prints:
```
[gpu_score] CUDA BATCH scoring ACTIVE (200 poses × 12 625 atoms)
```
No such line ⇒ CPU-grid mode (identical results, different speed).

> **When is the GPU worth it?** The batching win grows with
> poses-per-step × steps: roughly ≥ 300 glowworms/swarm **and** ≥ 100 steps for
> a clear advantage (1.6–2.4× vs CPU grid at scale). For small quick jobs the
> CPU grid build (LKlight-grid) is equally fast and needs no NVIDIA hardware.

## Releases (portable; only the NVIDIA *driver* is needed on the target)

| File | Platform | Runtime | Notes |
|---|---|---|---|
| `release_bin/LKlight-linux-cuda` | Linux x86-64 | NVIDIA driver | statically linked cudart; CUDA ACTIVE verified on RTX 3080 Ti |
| `release_bin/LKlight-win64-cuda.exe` | Windows 10/11 x64 | NVIDIA driver | statically linked cudart; CUDA ACTIVE verified on RTX 2080 |

Both binaries fall back to CPU grid automatically when no GPU/driver is found.

## Building from source (Windows CUDA walkthrough)

Prerequisites: Rust stable, an NVIDIA **GPU + driver**, plus on the build host:

| Linux | Windows |
|---|---|
| `nvcc` (CUDA ≥ 12) + glibc/musl toolchain | Visual Studio Build Tools 2022 (C++ workload, `vcvars64.bat`) + CUDA Toolkit ≥ 12 (`%CUDA_PATH%`) |

```bash
cargo build --release                  # CPU grid (any machine)
cargo build --release --features cuda  # CUDA (build.rs compiles src/cuda/*.cu with
                                       #  nvcc and links the static cudart)
cargo test --release                   # 34 tests
```

**Windows notes**
1. Install Rust (`rustup-init.exe -y`), VS Build Tools C++ workload and CUDA Toolkit.
2. Run inside a developer prompt:
   `cmd /c "call <VS>\VC\Auxiliary\Build\vcvars64.bat && set PATH=%USERPROFILE%\.cargo\bin;<CUDA>\bin;<MSVC>\bin\Hostx64\x64;%PATH% && cargo build --release --features cuda"`
   — the explicit MSVC `cl.exe` path in `PATH` is required or nvcc reports
   *"Cannot find compiler 'cl.exe'"*.
3. `build.rs` supports both platforms: `.o` + `ar` on Linux, `.obj` + `lib.exe`
   on Windows, static `cudart` on both.

## Performance (RNA system: 8 218 rec + 12 625 lig atoms)

| Scenario (RTX 3080 Ti / 12 cores) | CPU grid | GPU batch |
|---|---|---|
| 1 swarm × 2000 glow × 20 steps | 23.95 s | **7.57 s (3.2×)** |
| 1 swarm × 200 glow × 100 steps | 16.1 s | **7.85 s** |
| 1 swarm × 1000/2000/5000 glow × 100 steps | 32.7 s / — / — | 9.1 / 10.5 / ~21 s |

Full benchmark: `GPU_BENCH_20260902.md`, `PERF_COMPARE_20260902.md`.
Chinese readme: `README.zh-CN.md`. Engine write-ups: `docs/engines/`.

## Acceptance runs (2026-09-03, real machines)

| Platform | GPU | 100-step run (1AZP) |
|---|---|---|
| Windows Server 2022 | RTX 2080 | `CUDA BATCH ACTIVE` — passed |
| Linux 12-core | RTX 3080 Ti | `CUDA BATCH ACTIVE` — 1.30 s |
| Linux/Windows/macOS CPU fallback | — | grid scores bit-identical to LKlight-grid |

Raw logs: `tests/acceptance/{windows,linux,macos}_acceptance.txt`.

## Known boundaries (not bugs)

- Restraints / membrane / ANM runs use the CPU grid path inside this binary
  (correctness first — interface flags and per-pose deformation are not
  representable in the batch kernel).
- macOS (Apple Silicon) has no NVIDIA hardware → use **LKlight-grid**.
- Very long ligands (> several hundred Å) are geometrically pathological for
  rigid docking; trim to the binding domain first.

## License

See `LICENSE` / `NOTICE`. LKlight is an independent Rust implementation of the
LightDock docking protocol (https://github.com/bioinsilico/LightDock), which is
released under its own open-source licence.
