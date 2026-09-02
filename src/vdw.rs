use super::constants::INTERFACE_CUTOFF2;
use super::nearcell::NearCells;
use super::pydock::PYDOCKDockingModel;
use super::qt::{rot3_apply, Quaternion};
use std::cell::RefCell;
use std::sync::OnceLock;
use super::scoring::{satisfied_restraints, Score};
use pdbtbx::PDB;

const VDW_CUTOFF: f64 = 1.0;
const VDW_DIST_CUTOFF: f64 = 10.0;
const VDW_DIST_CUTOFF2: f64 = VDW_DIST_CUTOFF * VDW_DIST_CUTOFF;

pub struct VDW {
    pub receptor: PYDOCKDockingModel,
    pub ligand: PYDOCKDockingModel,
    pub use_anm: bool,
    /// Receptor cell list (10 Å cells) built once for the grid-accelerated
    /// path. Receptor is rigid (non-ANM) so it is valid for the whole run.
    cells: OnceLock<NearCells>,
}

impl<'a> VDW {
    pub fn new(
        receptor: PDB,
        rec_active_restraints: Vec<String>,
        rec_passive_restraints: Vec<String>,
        rec_nmodes: Vec<f64>,
        rec_num_anm: usize,
        ligand: PDB,
        lig_active_restraints: Vec<String>,
        lig_passive_restraints: Vec<String>,
        lig_nmodes: Vec<f64>,
        lig_num_anm: usize,
        use_anm: bool,
    ) -> Box<dyn Score + 'a> {
        let d = VDW {
            receptor: PYDOCKDockingModel::new(
                &receptor,
                &rec_active_restraints,
                &rec_passive_restraints,
                &rec_nmodes,
                rec_num_anm,
            ),
            ligand: PYDOCKDockingModel::new(
                &ligand,
                &lig_active_restraints,
                &lig_passive_restraints,
                &lig_nmodes,
                lig_num_anm,
            ),
            use_anm,
            cells: OnceLock::new(),
        };
        Box::new(d)
    }

    /// Grid-accelerated evaluation: same pair set and formulas as the exact
    /// all-vs-all scan (every contributing pair has d ≤ 10 Å = the cell edge,
    /// so the 27-cell window enumerates them exactly). Differ from
    /// [`Self::energy_exact`] only by f64 summation order (~1e-13).
    pub fn energy_grid(
        &self,
        translation: &[f64],
        rotation: &Quaternion,
        rec_nmodes: &[f64],
        lig_nmodes: &[f64],
    ) -> f64 {
        if self.use_anm
            || !rec_nmodes.is_empty()
            || !lig_nmodes.is_empty()
        {
            return self.energy_exact(translation, rotation, rec_nmodes, lig_nmodes);
        }
        let rot_mat = rotation.to_matrix();
        let cells = self.cells.get_or_init(|| {
            NearCells::build(&self.receptor.coordinates, VDW_DIST_CUTOFF)
        });

        thread_local! {
            static SCRATCH: RefCell<(Vec<[f64;3]>, Vec<[f64;3]>, Vec<usize>, Vec<usize>)> =
                RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
        }
        SCRATCH.with(|sc| {
        let mut sc = sc.borrow_mut();
        let (rec_c, lig_c, iface_r, iface_l) = &mut *sc;
        let rec_n = self.receptor.coordinates.len();
        let lig_n = self.ligand.coordinates.len();
        if rec_c.len() != rec_n { rec_c.resize(rec_n, [0.0;3]); }
        if lig_c.len() != lig_n { lig_c.resize(lig_n, [0.0;3]); }
        if iface_r.len() != rec_n { iface_r.resize(rec_n, 0); }
        if iface_l.len() != lig_n { iface_l.resize(lig_n, 0); }
        rec_c.copy_from_slice(&self.receptor.coordinates);
        lig_c.copy_from_slice(&self.ligand.coordinates);
        for v in iface_r.iter_mut() { *v = 0; }
        for v in iface_l.iter_mut() { *v = 0; }

        for coordinate in lig_c.iter_mut() {
            let r = rot3_apply(&rot_mat, *coordinate);
            coordinate[0] = r[0] + translation[0];
            coordinate[1] = r[1] + translation[1];
            coordinate[2] = r[2] + translation[2];
        }

        // Near-pair scan: each ligand atom tests the receptor atoms in its 27
        // cells. Every (i,j) with d ≤ 10 Å is visited exactly once; the LJ term
        // and the interface flags use the same formulas as energy_exact.
        let rec_vc = &self.receptor.vdw_charges;
        let lig_vc = &self.ligand.vdw_charges;
        let rec_vr = &self.receptor.vdw_radii;
        let lig_vr = &self.ligand.vdw_radii;
        let mut total_vdw = 0.0f64;
        for (j, la) in lig_c.iter().enumerate() {
            let x = la[0]; let y = la[1]; let z = la[2];
            cells.for_each_near(x, y, z, &mut |i| {
                let dx = x - rec_c[i][0];
                let dy = y - rec_c[i][1];
                let dz = z - rec_c[i][2];
                let d2 = dx*dx + dy*dy + dz*dz;
                if d2 <= VDW_DIST_CUTOFF2 {
                    let vdw_energy = (rec_vc[i] * lig_vc[j]).sqrt();
                    let vdw_radius = rec_vr[i] + lig_vr[j];
                    let p6 = vdw_radius.powi(6) / d2.powi(3);
                    let mut k = vdw_energy * (p6 * p6 - 2.0 * p6);
                    if k > VDW_CUTOFF { k = VDW_CUTOFF; }
                    total_vdw += k;
                }
                if d2 <= INTERFACE_CUTOFF2 {
                    iface_r[i] = 1;
                    iface_l[j] = 1;
                }
            });
        }

        let score = total_vdw * -1.0;
        let perc_r = satisfied_restraints(iface_r, &self.receptor.active_restraints);
        let perc_l = satisfied_restraints(iface_l, &self.ligand.active_restraints);
        score + perc_r * score + perc_l * score
        })
    }

    fn energy_exact(
        &self,
        translation: &[f64],
        rotation: &Quaternion,
        rec_nmodes: &[f64],
        lig_nmodes: &[f64],
    ) -> f64 {
        thread_local! {
            static SCRATCH: RefCell<(Vec<[f64;3]>, Vec<[f64;3]>, Vec<usize>, Vec<usize>)> =
                RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
        }
        let rot_mat = rotation.to_matrix();
        SCRATCH.with(|sc| {
        let mut sc = sc.borrow_mut();
        let (rec_c, lig_c, iface_r, iface_l) = &mut *sc;
        let rec_n = self.receptor.coordinates.len();
        let lig_n = self.ligand.coordinates.len();
        if rec_c.len() != rec_n { rec_c.resize(rec_n, [0.0;3]); }
        if lig_c.len() != lig_n { lig_c.resize(lig_n, [0.0;3]); }
        if iface_r.len() != rec_n { iface_r.resize(rec_n, 0); }
        if iface_l.len() != lig_n { iface_l.resize(lig_n, 0); }
        rec_c.copy_from_slice(&self.receptor.coordinates);
        lig_c.copy_from_slice(&self.ligand.coordinates);
        for v in iface_r.iter_mut() { *v = 0; }
        for v in iface_l.iter_mut() { *v = 0; }
        let rec_num_atoms = rec_n;
        let lig_num_atoms = lig_n;

        let lig_nm_n = if self.ligand.num_anm > 0 {
            self.ligand.nmodes.len() / (3 * self.ligand.num_anm)
        } else { lig_num_atoms };
        let rec_nm_n = if self.receptor.num_anm > 0 {
            self.receptor.nmodes.len() / (3 * self.receptor.num_anm)
        } else { rec_num_atoms };

        for (i_atom, coordinate) in lig_c.iter_mut().enumerate() {
            let r = rot3_apply(&rot_mat, *coordinate);
            coordinate[0] = r[0] + translation[0];
            coordinate[1] = r[1] + translation[1];
            coordinate[2] = r[2] + translation[2];
            if self.use_anm && self.ligand.num_anm > 0 && i_atom < lig_nm_n {
                for i_nm in 0..self.ligand.num_anm {
                    let b = i_nm * lig_nm_n * 3 + i_atom * 3;
                    coordinate[0] += self.ligand.nmodes[b]   * lig_nmodes[i_nm];
                    coordinate[1] += self.ligand.nmodes[b+1] * lig_nmodes[i_nm];
                    coordinate[2] += self.ligand.nmodes[b+2] * lig_nmodes[i_nm];
                }
            }
        }
        if self.use_anm && self.receptor.num_anm > 0 {
            for (i_atom, coordinate) in rec_c.iter_mut().enumerate() {
                if i_atom >= rec_nm_n { break; }
                for i_nm in 0..self.receptor.num_anm {
                    let b = i_nm * rec_nm_n * 3 + i_atom * 3;
                    coordinate[0] += self.receptor.nmodes[b]   * rec_nmodes[i_nm];
                    coordinate[1] += self.receptor.nmodes[b+1] * rec_nmodes[i_nm];
                    coordinate[2] += self.receptor.nmodes[b+2] * rec_nmodes[i_nm];
                }
            }
        }

        let mut total_vdw = 0.0;
        for (i, ra) in rec_c.iter().enumerate() {
            let x1 = ra[0];
            let y1 = ra[1];
            let z1 = ra[2];
            for (j, la) in lig_c.iter().enumerate() {
                let distance2 = (x1 - la[0]) * (x1 - la[0])
                    + (y1 - la[1]) * (y1 - la[1])
                    + (z1 - la[2]) * (z1 - la[2]);

                if distance2 <= VDW_DIST_CUTOFF2 {
                    let vdw_energy =
                        (self.receptor.vdw_charges[i] * self.ligand.vdw_charges[j]).sqrt();
                    let vdw_radius = self.receptor.vdw_radii[i] + self.ligand.vdw_radii[j];
                    let p6 = vdw_radius.powi(6) / distance2.powi(3);
                    let mut k = vdw_energy * (p6 * p6 - 2.0 * p6);
                    if k > VDW_CUTOFF {
                        k = VDW_CUTOFF;
                    }
                    total_vdw += k;
                }

                if distance2 <= INTERFACE_CUTOFF2 {
                    iface_r[i] = 1;
                    iface_l[j] = 1;
                }
            }
        }

        let score = total_vdw * -1.0;
        let perc_r = satisfied_restraints(iface_r, &self.receptor.active_restraints);
        let perc_l = satisfied_restraints(iface_l, &self.ligand.active_restraints);
        score + perc_r * score + perc_l * score
        })
    }
}

impl Score for VDW {
    fn energy(
        &self,
        translation: &[f64],
        rotation: &Quaternion,
        rec_nmodes: &[f64],
        lig_nmodes: &[f64],
    ) -> f64 {
        self.energy_grid(translation, rotation, rec_nmodes, lig_nmodes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qt::Quaternion;
    use std::env;

    fn load_1azp() -> (PDB, PDB) {
        let base = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
        let (rec, _) = pdbtbx::open(
            &format!("{}/tests/1azp/1azp_receptor.pdb", base),
            pdbtbx::StrictnessLevel::Strict,
        )
        .unwrap();
        let (lig, _) = pdbtbx::open(
            &format!("{}/tests/1azp/1azp_ligand.pdb", base),
            pdbtbx::StrictnessLevel::Strict,
        )
        .unwrap();
        (rec, lig)
    }

    #[test]
    fn test_1azp() {
        let (rec_pdb, lig_pdb) = load_1azp();
        let rec = super::super::pydock::PYDOCKDockingModel::new(&rec_pdb, &[], &[], &[], 0);
        let lig = super::super::pydock::PYDOCKDockingModel::new(&lig_pdb, &[], &[], &[], 0);
        let s = VDW {
            receptor: rec,
            ligand: lig,
            use_anm: false,
            cells: OnceLock::new(),
        };
        let t = vec![0., 0., 0.];
        let q = Quaternion::default();
        let g = s.energy_grid(&t, &q, &[], &[]);
        let e = s.energy_exact(&t, &q, &[], &[]);
        // vdw has no far-field term: cell path and all-vs-all scan compute the
        // same pairs/formulas → agreement to f64 summation order.
        assert!((g - e).abs() < 1e-9, "grid {g} vs exact {e}");
        for tt in [[2., 0., 0.], [0., 3., 0.], [-4., 2., 5.], [1., -2., 6.]] {
            let g2 = s.energy_grid(&tt, &q, &[], &[]);
            let e2 = s.energy_exact(&tt, &q, &[], &[]);
            assert!((g2 - e2).abs() < 1e-9, "pose {tt:?}: grid {g2} vs exact {e2}");
        }
    }
}
