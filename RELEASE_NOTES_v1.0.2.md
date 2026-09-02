# LKlight v1.0.2

This release focuses on robustness for protein-RNA / protein-DNA docking, especially nucleic-acid phosphate atoms and non-standard atom naming encountered in real PDB files.

## Fixed

### Protein-RNA / protein-DNA phosphate crash
- Fixed a hard panic in AMBER-based scoring paths when a nucleic-acid phosphate atom `P` did not match an exact AMBER atom key.
- The previous failure mode was `*-P not supported`, which aborted the scoring process and caused all LightDock swarms to fail.
- Added generic AMBER fallback atom types and charges for phosphate and common heteroatoms:
  - `*-P`
  - `*-I`
  - `*-B`
  - `*-Z`
  - `*-K`
  - `*-M`

### No-panic atom typing
- Hardened AMBER-typed scorers so unknown atoms no longer crash the engine.
- Affected scoring families:
  - `dna`
  - `pydock`
  - `cpydock`
  - `sd`
- Unknown atoms now fall back safely to a neutral carbon-like type with zero electrostatic charge instead of panicking.

### Setup-stage atom filtering
- Improved Rust setup-stage filtering for:
  - `--noh` hydrogens
  - `--noxt` terminal `OXT`
  - `--now` water molecules
- This prevents filtered atoms from reaching the scoring stage.

## Validation

Regression testing was performed with a synthetic protein-RNA complex containing a raw single-letter RNA residue `U` and a phosphate atom `P`, reproducing the original crash trigger.

Results:
- `dna` scoring completed successfully with no panic.
- `ddna` scoring completed successfully with no panic.
- macOS binary was runtime-tested locally.
- Windows and Linux binaries were rebuilt and type-checked.

## Binaries

Attached prebuilt binaries:

- `LKlight-v1.0.2-mac-arm64.zip`
  - macOS Apple Silicon / arm64
- `LKlight-v1.0.2-win-x64.zip`
  - Windows x86-64 / PE32+
  - Compatible with modern 64-bit Windows systems
- `LKlight-v1.0.2-linux-x86_64.tar.gz`
  - Linux x86_64
  - Static musl build

## Notes

- Windows binary is x86-64 only and does not support 32-bit Windows.
- macOS binary is arm64 only. Intel macOS requires a separate x86_64 build.
