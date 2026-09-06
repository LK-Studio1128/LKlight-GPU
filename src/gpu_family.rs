//! Family-agnostic GPU batch scoring for the all-atom scoring families.
//!
//! DNA was the first family wired to the batched GPU kernel (full_score.cu /
//! lk_metal.m). Since the per-family CPU-grid scoring differs only by *which
//! terms are on* (electrostatics near/far, clash penalty; LJ is always on —
//! every all-atom family scores it), the batched kernel is parameterised by a
//! `flags` word and this module provides the model-agnostic host bridge used
//! by VDW / PYDOCK / CPYDOCK.
//!
//! Per-family flags (see `gpu_score`): DNA=7, PYDOCK=3, VDW=0. The host score
//! combination is the same for all of them, score = -(E_raw*FACTOR/EPSILON+V):
//! for VDW the kernel returns E_raw = 0 (no electrostatic flags set).
//!
//! `FamilyGpu` caches the GPU-ready receptor (incl. the 10 Å cell list) and
//! ligand arrays plus the receptor field on each scorer instance (`OnceLock`),
//! mirroring DNA's field/rec_cuda/lig_cuda caches, so host pointers are stable
//! across GSO steps (required for the CUDA persistent-buffer cache to hit).

use crate::gpu_score::{CudaReceptor, F_CLASH, F_ELEC, F_FAR, FLAGS_DNA};
use crate::grid_dna::ReceptorField;
use crate::qt::Quaternion;

/// A molecule that can be lifted onto the GPU. `heavy()` is only needed by
/// DNA's clash term (the only family that uses it); returning `None` yields a
/// zero heavy mask, which is fine because no other family reads it (their
/// `flags` clear F_CLASH).
pub trait FamilyMol {
    fn coords(&self) -> &[[f64; 3]];
    fn ele(&self) -> &[f64];
    fn sqrt_vdw(&self) -> &[f64];
    fn vdw_radii(&self) -> &[f64];
    fn heavy(&self) -> Option<&[u8]>;
}

impl FamilyMol for crate::pydock::PYDOCKDockingModel {
    fn coords(&self) -> &[[f64; 3]] {
        &self.coordinates
    }
    fn ele(&self) -> &[f64] {
        &self.ele_charges
    }
    fn sqrt_vdw(&self) -> &[f64] {
        &self.sqrt_vdw_charges
    }
    fn vdw_radii(&self) -> &[f64] {
        &self.vdw_radii
    }
    fn heavy(&self) -> Option<&[u8]> {
        None
    }
}

impl FamilyMol for crate::cpydock::CPYDOCKDockingModel {
    fn coords(&self) -> &[[f64; 3]] {
        &self.coordinates
    }
    fn ele(&self) -> &[f64] {
        &self.ele_charges
    }
    fn sqrt_vdw(&self) -> &[f64] {
        &self.sqrt_vdw_charges
    }
    fn vdw_radii(&self) -> &[f64] {
        &self.vdw_radii
    }
    fn heavy(&self) -> Option<&[u8]> {
        None
    }
}

/// Cacheable GPU-ready dataset for one scorer instance (receptor + ligand +
/// optional field). Built once, cached in a `OnceLock`, reused every step.
pub struct FamilyGpu {
    pub crec: CudaReceptor,
    pub clig: Vec<f32>,       // nl*3 reference coords (f32)
    pub le: Vec<f32>,
    pub lsv: Vec<f32>,
    pub lv: Vec<f32>,
    pub lh: Vec<u8>,
}

fn build_crec(coords: &[[f64; 3]], ele: &[f64], svdw: &[f64], radii: &[f64],
              heavy: Option<&[u8]>) -> CudaReceptor {
    // Cell-list construction over the receptor coordinates (cell side = 10 Å),
    // identical layout/logic to CudaReceptor::build (CLOSE_DIST == 10.0).
    const CELL: f64 = 10.0;
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for c in coords.iter() {
        for a in 0..3 {
            lo[a] = lo[a].min(c[a]);
            hi[a] = hi[a].max(c[a]);
        }
    }
    for a in 0..3 {
        lo[a] = (lo[a] - 0.01).floor();
    }
    let n = |axis: usize| ((hi[axis] - lo[axis]) / CELL).floor() as i32 + 1;
    let ncx = n(0).max(1);
    let ncy = n(1).max(1);
    let ncz = n(2).max(1);
    let ncells = ((ncx as usize) + 1) * (ncy as usize + 1) * (ncz as usize + 1);
    let mut cell_start = vec![0i32; ncells];
    let nr = coords.len();
    let mut cell_of = vec![0usize; nr];
    for (k, c) in coords.iter().enumerate() {
        let ix = (((c[0] - lo[0]) / CELL).floor() as i64).clamp(0, ncx as i64 - 1) as usize;
        let iy = (((c[1] - lo[1]) / CELL).floor() as i64).clamp(0, ncy as i64 - 1) as usize;
        let iz = (((c[2] - lo[2]) / CELL).floor() as i64).clamp(0, ncz as i64 - 1) as usize;
        let cell = (iz * ncy as usize + iy) * ncx as usize + ix;
        cell_of[k] = cell;
        cell_start[cell] += 1;
    }
    let mut acc = 0;
    for c in cell_start.iter_mut() {
        let n = *c;
        *c = acc;
        acc += n;
    }
    cell_start.push(acc);
    let mut cell_atoms = vec![0i32; nr];
    let mut cursor = cell_start.clone();
    for (k, &cell) in cell_of.iter().enumerate() {
        let pos = cursor[cell] as usize;
        cell_atoms[pos] = k as i32;
        cursor[cell] += 1;
    }
    let mut r_coords = Vec::with_capacity(nr * 3);
    for c in coords.iter() {
        r_coords.push(c[0] as f32);
        r_coords.push(c[1] as f32);
        r_coords.push(c[2] as f32);
    }
    CudaReceptor {
        r_coords,
        r_ele: ele.iter().map(|&x| x as f32).collect(),
        r_svdw: svdw.iter().map(|&x| x as f32).collect(),
        r_vdwr: radii.iter().map(|&x| x as f32).collect(),
        r_heavy: match heavy {
            Some(h) => h.iter().map(|&v| v as u8).collect(),
            None => vec![0u8; nr],
        },
        cell_start,
        cell_atoms,
        ncx,
        ncy,
        ncz,
        c_ox: lo[0] as f32,
        c_oy: lo[1] as f32,
        c_oz: lo[2] as f32,
        c_sp: CELL as f32,
    }
}

/// Build the cached GPU dataset for a scorer. The receptor far-field is NOT
/// owned here: the caller already holds one (DNA/PYDOCK/CPYDOCK build it for
/// the CPU grid path) and passes it per call — VDW simply passes `None` (its
/// flags clear F_FAR). This avoids building the 0.5 Å field twice per run.
pub fn build_family<M: FamilyMol>(
    rec: &M,
    lig: &M,
) -> FamilyGpu {
    let crec = build_crec(rec.coords(), rec.ele(), rec.sqrt_vdw(), rec.vdw_radii(), rec.heavy());
    let mut clig = Vec::with_capacity(lig.coords().len() * 3);
    for c in lig.coords().iter() {
        clig.push(c[0] as f32);
        clig.push(c[1] as f32);
        clig.push(c[2] as f32);
    }
    let heavy: Vec<u8> = match lig.heavy() {
        Some(h) => h.iter().map(|&v| v as u8).collect(),
        None => vec![0u8; lig.coords().len()],
    };
    FamilyGpu {
        crec,
        clig,
        le: lig.ele().iter().map(|&x| x as f32).collect(),
        lsv: lig.sqrt_vdw().iter().map(|&x| x as f32).collect(),
        lv: lig.vdw_radii().iter().map(|&x| x as f32).collect(),
        lh: heavy,
    }
}

/// Lift per-family flags for one run: DNA keeps the clash term, PYDOCK drops
/// it, VDW scores only the LJ (flags = 0; its electrostatic output is 0 and the
/// host combination below still reduces to score = -V).
pub fn family_flags(method: &str) -> u32 {
    match method {
        "dna" => FLAGS_DNA,
        "pydock" | "cpydock" => F_FAR | F_ELEC,
        "vdw" => 0,
        _ => F_FAR | F_ELEC | F_CLASH,
    }
}

/// CUDA batched scoring for a family. Returns per-pose total scores or `None`
/// on any failure (caller falls back to the CPU grid path).
#[cfg(feature = "cuda")]
pub fn batch_cuda_family(
    g: &FamilyGpu,
    field: Option<&ReceptorField>,
    translations: &[[f64; 3]],
    rotations: &[Quaternion],
    flags: u32,
) -> Option<Vec<f64>> {
    extern "C" {
        #[allow(clippy::too_many_arguments)]
        fn cuda_batch_score(
            phi: *const f32, nx: i32, ny: i32, nz: i32, ox: f32, oy: f32, oz: f32, sp: f32,
            r_coords: *const f32, r_ele: *const f32, r_svdw: *const f32, r_vdwr: *const f32,
            r_heavy: *const u8, nr: i32,
            cell_start: *const i32, cell_atoms: *const i32,
            ncx: i32, ncy: i32, ncz: i32, c_ox: f32, c_oy: f32, c_oz: f32, c_sp: f32,
            l_base: *const f32, poses: *const f64, l_ele: *const f32, l_svdw: *const f32,
            l_vdwr: *const f32, l_heavy: *const u8, nl: i32, n_pose: i32, flags: u32,
            out: *mut f64,
        ) -> i32;
    }
    let n_pose = translations.len();
    let nl = (g.clig.len() / 3) as i32;
    if n_pose == 0 || nl == 0 {
        return None;
    }
    // Field is only needed when F_FAR is set (DNA/PYDOCK/CPYDOCK); VDW (flags=0)
    // passes a dummy — the kernel never samples phi without F_FAR.
    let mut dummy: f32 = 0.0;
    let (phi, nxd, nyd, nzd, oxf, oyf, ozf, spf) = match field {
        Some(f) if !f.phi.is_empty() => (
            f.phi.as_ptr(), f.n[0] as i32, f.n[1] as i32, f.n[2] as i32,
            f.origin[0] as f32, f.origin[1] as f32, f.origin[2] as f32, f.spacing as f32,
        ),
        _ if flags & F_FAR != 0 => return None, // needs a field but has none
        _ => (&dummy as *const f32, 0, 0, 0, 0.0f32, 0.0f32, 0.0f32, 0.0f32),
    };
    let crec = &g.crec;
    let mut poses: Vec<f64> = Vec::with_capacity(n_pose * 7);
    for (t, r) in translations.iter().zip(rotations.iter()) {
        poses.push(r.w);
        poses.push(r.x);
        poses.push(r.y);
        poses.push(r.z);
        poses.push(t[0]);
        poses.push(t[1]);
        poses.push(t[2]);
    }
    let mut out = vec![0.0f64; n_pose * 2];
    let ret = unsafe {
        cuda_batch_score(
            phi, nxd, nyd, nzd, oxf, oyf, ozf, spf,
            crec.r_coords.as_ptr(), crec.r_ele.as_ptr(), crec.r_svdw.as_ptr(),
            crec.r_vdwr.as_ptr(), crec.r_heavy.as_ptr(), (crec.r_coords.len() / 3) as i32,
            crec.cell_start.as_ptr(), crec.cell_atoms.as_ptr(),
            crec.ncx, crec.ncy, crec.ncz, crec.c_ox, crec.c_oy, crec.c_oz, crec.c_sp,
            g.clig.as_ptr(), poses.as_ptr(), g.le.as_ptr(), g.lsv.as_ptr(), g.lv.as_ptr(),
            g.lh.as_ptr(), nl, n_pose as i32, flags, out.as_mut_ptr(),
        )
    };
    if ret != 0 {
        return None;
    }
    use std::sync::atomic::{AtomicBool, Ordering};
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        eprintln!("[gpu_family] CUDA family BATCH scoring ACTIVE ({} poses x {} atoms, flags={})",
                  n_pose, nl, flags);
    }
    const FACTOR: f64 = 332.0;
    const EPSILON: f64 = 4.0;
    Some((0..n_pose).map(|k| -(out[2 * k] * FACTOR / EPSILON + out[2 * k + 1])).collect())
}

#[cfg(not(feature = "cuda"))]
pub fn batch_cuda_family(
    _g: &FamilyGpu,
    _field: Option<&ReceptorField>,
    _translations: &[[f64; 3]],
    _rotations: &[Quaternion],
    _flags: u32,
) -> Option<Vec<f64>> {
    None
}

/// Shared accelerator probe (cuda first, then metal), identical semantics to
/// DNA's `batch_accel_available`.
pub fn family_batch_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        return crate::gpu_score::cuda_available();
    }
    #[cfg(feature = "metal")]
    {
        return crate::metal_score::metal_available();
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        false
    }
}

/// CPYDOCK desolvation arrays (per-model): the C-compatible exclusion mask
/// flag(i) = (i even) && hydrogens[i/2]!=0 (1 = excluded from min counting),
/// plus per-atom desolvation coefficient and reference SASA.
pub struct CpySolvSide {
    pub flag: Vec<u8>,
    pub des: Vec<f32>,
    pub asa: Vec<f32>,
}

/// Batched CUDA desolvation for CPYDOCK: returns per-pose S (the desolvation
/// energy already subtracted in score = -(E·332/4 + 0.1·V − S)), or None on
/// failure. Runs the two-stage min/reduce kernels on the shared pose buffer.
#[cfg(feature = "cuda")]
pub fn batch_cuda_cpydock_solv(
    g: &FamilyGpu,
    field: Option<&ReceptorField>,
    solv_r: &CpySolvSide,
    solv_l: &CpySolvSide,
    translations: &[[f64; 3]],
    rotations: &[Quaternion],
) -> Option<Vec<f64>> {
    extern "C" {
        #[allow(clippy::too_many_arguments)]
        fn cuda_cpydock_solv(
            r_coords: *const f32, r_flag: *const u8, nr: i32,
            cell_start: *const i32, cell_atoms: *const i32,
            ncx: i32, ncy: i32, ncz: i32,
            c_ox: f32, c_oy: f32, c_oz: f32, c_sp: f32,
            l_base: *const f32, poses: *const f64,
            l_flag: *const u8, l_ele: *const f32, nl: i32, n_pose: i32,
            r_des: *const f32, r_asa: *const f32,
            l_des: *const f32, l_asa: *const f32,
            out_s: *mut f64,
        ) -> i32;
    }
    let n_pose = translations.len();
    let nl = (g.clig.len() / 3) as i32;
    let nr = (g.crec.r_coords.len() / 3) as i32;
    if n_pose == 0 || nl == 0 || nr == 0 || solv_r.flag.len() != nr as usize
        || solv_l.flag.len() != nl as usize {
        return None;
    }
    let _ = field; // desolv needs no far field
    let crec = &g.crec;
    let mut poses: Vec<f64> = Vec::with_capacity(n_pose * 7);
    for (t, r) in translations.iter().zip(rotations.iter()) {
        poses.push(r.w);
        poses.push(r.x);
        poses.push(r.y);
        poses.push(r.z);
        poses.push(t[0]);
        poses.push(t[1]);
        poses.push(t[2]);
    }
    let mut out = vec![0.0f64; n_pose];
    let ret = unsafe {
        cuda_cpydock_solv(
            crec.r_coords.as_ptr(), solv_r.flag.as_ptr(), nr,
            crec.cell_start.as_ptr(), crec.cell_atoms.as_ptr(),
            crec.ncx, crec.ncy, crec.ncz,
            crec.c_ox, crec.c_oy, crec.c_oz, crec.c_sp,
            g.clig.as_ptr(), poses.as_ptr(),
            solv_l.flag.as_ptr(), g.le.as_ptr(), nl, n_pose as i32,
            solv_r.des.as_ptr(), solv_r.asa.as_ptr(),
            solv_l.des.as_ptr(), solv_l.asa.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if ret != 0 {
        return None;
    }
    Some(out)
}

#[cfg(not(feature = "cuda"))]
pub fn batch_cuda_cpydock_solv(
    _g: &FamilyGpu,
    _field: Option<&ReceptorField>,
    _solv_r: &CpySolvSide,
    _solv_l: &CpySolvSide,
    _translations: &[[f64; 3]],
    _rotations: &[Quaternion],
) -> Option<Vec<f64>> {
    None
}
