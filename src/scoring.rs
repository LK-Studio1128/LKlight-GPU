use super::qt::Quaternion;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub enum Method {
    CPYDOCK,
    DDNA,
    DFIRE,
    DFIRE2,
    DNA,
    MJ3H,
    PISA,
    PYDOCK,
    SD,
    SIPPER,
    TOBI,
    VDW,
}

pub trait Score: Send + Sync {
    fn energy(
        &self,
        translation: &[f64],
        rotation: &Quaternion,
        rec_nmodes: &[f64],
        lig_nmodes: &[f64],
    ) -> f64;

    /// Exact per-pair evaluation (reference). Default: same as `energy`.
    fn energy_exact(
        &self,
        translation: &[f64],
        rotation: &Quaternion,
        rec_nmodes: &[f64],
        lig_nmodes: &[f64],
    ) -> f64 {
        self.energy(translation, rotation, rec_nmodes, lig_nmodes)
    }

    /// Grid-accelerated evaluation. Default: same as `energy`.
    fn energy_grid(
        &self,
        translation: &[f64],
        rotation: &Quaternion,
        rec_nmodes: &[f64],
        lig_nmodes: &[f64],
    ) -> f64 {
        self.energy(translation, rotation, rec_nmodes, lig_nmodes)
    }

    /// Whether the scorer can evaluate many poses in one batched call
    /// (used by GPU back-ends to amortise launch/sync overhead).
    fn supports_batch(&self) -> bool {
        false
    }

    /// Evaluate one energy per pose. Default implementation falls back to
    /// per-pose `energy` calls, so non-batched scorers behave exactly as before.
    fn batch_energy(&self, translations: &[[f64; 3]], rotations: &[Quaternion]) -> Vec<f64> {
        translations
            .iter()
            .zip(rotations.iter())
            .map(|(t, r)| self.energy(t, r, &[], &[]))
            .collect()
    }

    /// [Metal POC] Export a deterministic N-pose scoring dataset + CPU baseline
    /// for an external Metal kernel (see `DNA::dump_metal_dataset`). Default:
    /// unsupported → error.
    fn dump_metal_dataset(
        &self,
        _n_pose: usize,
        _seed: u64,
        _dir: &std::path::Path,
    ) -> std::io::Result<(f64, f64)> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "dump_metal_dataset not supported by this scoring engine",
        ))
    }
}

pub fn satisfied_restraints(interface: &[usize], restraints: &HashMap<String, Vec<usize>>) -> f64 {
    // Calculate the percentage of satisfied restraints
    if restraints.is_empty() {
        return 0.0;
    }
    let mut num_residues = 0;
    for (_k, atom_indexes) in restraints.iter() {
        for &i in atom_indexes.iter() {
            if interface[i] == 1 {
                num_residues += 1;
                break;
            }
        }
    }
    num_residues as f64 / restraints.len() as f64
}

pub fn membrane_intersection(interface: &[usize], membrane: &[usize]) -> f64 {
    if membrane.is_empty() {
        return 0.0;
    }
    let mut num_beads = 0;
    for &i_bead in membrane.iter() {
        num_beads += interface[i_bead];
    }
    num_beads as f64 / membrane.len() as f64
}
