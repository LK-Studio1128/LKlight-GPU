//! Full-pose GPU scorer bridge (near pairs + far field in one kernel).
//!
//! With the `cuda` feature, [`full_energy_gpu`] moves BOTH the near-pair terms
//! (d ≤ 10 Å: clamped electrostatics + capped LJ + heavy-atom clash penalty —
//! ~98% of grid-path runtime) and the far-field lookup onto the GPU, one
//! thread per ligand atom over a receptor cell list. Any failure falls back to
//! the CPU grid path, so docking never breaks without CUDA.

use crate::dna::{DNA, DNADockingModel};
use crate::grid_dna::{ReceptorField, CLOSE_DIST};
use crate::qt::Quaternion;

/// GPU-ready receptor description: parameter arrays + 10 Å cell list (built once).
pub struct CudaReceptor {
    pub r_coords: Vec<f32>,
    pub r_ele: Vec<f32>,
    pub r_svdw: Vec<f32>,
    pub r_vdwr: Vec<f32>,
    pub r_heavy: Vec<u8>,
    pub cell_start: Vec<i32>,
    pub cell_atoms: Vec<i32>,
    pub ncx: i32,
    pub ncy: i32,
    pub ncz: i32,
    pub c_ox: f32,
    pub c_oy: f32,
    pub c_oz: f32,
    pub c_sp: f32,
}

impl CudaReceptor {
    /// Build cell list (cell side = CLOSE_DIST = 10 Å) over the receptor.
    pub fn build(rec: &DNADockingModel) -> CudaReceptor {
        let cell = CLOSE_DIST;
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for c in rec.coordinates.iter() {
            for a in 0..3 {
                lo[a] = lo[a].min(c[a]);
                hi[a] = hi[a].max(c[a]);
            }
        }
        for a in 0..3 {
            lo[a] = (lo[a] - 0.01).floor();
        }
        let n = |axis: usize| ((hi[axis] - lo[axis]) / cell).floor() as i32 + 1;
        let ncx = n(0).max(1);
        let ncy = n(1).max(1);
        let ncz = n(2).max(1);
        let ncells = ((ncx as usize) + 1) * (ncy as usize + 1) * (ncz as usize + 1);
        let mut cell_start = vec![0i32; ncells];
        let nr = rec.coordinates.len();
        let mut cell_of = vec![0usize; nr];
        for (k, c) in rec.coordinates.iter().enumerate() {
            let ix = (((c[0] - lo[0]) / cell).floor() as i64).clamp(0, ncx as i64 - 1) as usize;
            let iy = (((c[1] - lo[1]) / cell).floor() as i64).clamp(0, ncy as i64 - 1) as usize;
            let iz = (((c[2] - lo[2]) / cell).floor() as i64).clamp(0, ncz as i64 - 1) as usize;
            let cell = (iz * ncy as usize + iy) * ncx as usize + ix;
            cell_of[k] = cell;
            cell_start[cell] += 1;
        }
        // prefix sums → offsets
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
        let f = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<_>>();
        CudaReceptor {
            r_coords: {
                let mut v = Vec::with_capacity(nr * 3);
                for c in rec.coordinates.iter() {
                    v.push(c[0] as f32);
                    v.push(c[1] as f32);
                    v.push(c[2] as f32);
                }
                v
            },
            r_ele: f(&rec.ele_charges),
            r_svdw: f(&rec.sqrt_vdw_charges),
            r_vdwr: f(&rec.vdw_radii),
            r_heavy: rec.heavy.iter().map(|&h| h as u8).collect(),
            cell_start,
            cell_atoms,
            ncx,
            ncy,
            ncz,
            c_ox: lo[0] as f32,
            c_oy: lo[1] as f32,
            c_oz: lo[2] as f32,
            c_sp: cell as f32,
        }
    }
}

/// Compute the full DNA score for one pose on the GPU.
/// Returns (elec_raw, vdw) in the same units as the CPU grid path;
/// caller computes score = -(elec_raw * FACTOR/EPSILON + vdw).
#[cfg(feature = "cuda")]
pub fn full_energy_gpu(
    field: &ReceptorField,
    crec: &CudaReceptor,
    lig_coords: &[[f64; 3]],
    lig: &DNADockingModel,
) -> Option<(f64, f64)> {
    extern "C" {
        #[allow(clippy::too_many_arguments)]
        fn cuda_full_score(
            phi: *const f32, nx: i32, ny: i32, nz: i32, ox: f32, oy: f32, oz: f32, sp: f32,
            r_coords: *const f32, r_ele: *const f32, r_svdw: *const f32, r_vdwr: *const f32,
            r_heavy: *const u8, nr: i32,
            cell_start: *const i32, cell_atoms: *const i32,
            ncx: i32, ncy: i32, ncz: i32, c_ox: f32, c_oy: f32, c_oz: f32, c_sp: f32,
            l_coords: *const f32, l_ele: *const f32, l_svdw: *const f32, l_vdwr: *const f32,
            l_heavy: *const u8, nl: i32,
            out_elec: *mut f32, out_vdw: *mut f32,
        ) -> i32;
    }
    let nl = lig_coords.len();
    if nl == 0 || field.phi.is_empty() {
        return None;
    }
    let mut lc = Vec::with_capacity(nl * 3);
    for c in lig_coords.iter() {
        lc.push(c[0] as f32);
        lc.push(c[1] as f32);
        lc.push(c[2] as f32);
    }
    let le: Vec<f32> = lig.ele_charges.iter().map(|&q| q as f32).collect();
    let lsv: Vec<f32> = lig.sqrt_vdw_charges.iter().map(|&q| q as f32).collect();
    let lv: Vec<f32> = lig.vdw_radii.iter().map(|&r| r as f32).collect();
    let lh: Vec<u8> = lig.heavy.iter().map(|&h| h as u8).collect();
    let mut oe = vec![0.0f32; nl];
    let mut ov = vec![0.0f32; nl];
    let ret = unsafe {
        cuda_full_score(
            field.phi.as_ptr(), field.n[0] as i32, field.n[1] as i32, field.n[2] as i32,
            field.origin[0] as f32, field.origin[1] as f32, field.origin[2] as f32,
            field.spacing as f32,
            crec.r_coords.as_ptr(), crec.r_ele.as_ptr(), crec.r_svdw.as_ptr(),
            crec.r_vdwr.as_ptr(), crec.r_heavy.as_ptr(), crec.r_coords.len() as i32 / 3,
            crec.cell_start.as_ptr(), crec.cell_atoms.as_ptr(),
            crec.ncx, crec.ncy, crec.ncz, crec.c_ox, crec.c_oy, crec.c_oz, crec.c_sp,
            lc.as_ptr(), le.as_ptr(), lsv.as_ptr(), lv.as_ptr(), lh.as_ptr(), nl as i32,
            oe.as_mut_ptr(), ov.as_mut_ptr(),
        )
    };
    if ret != 0 {
        return None;
    }
    use std::sync::atomic::{AtomicBool, Ordering};
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        eprintln!("[gpu_score] CUDA full-pose scoring ACTIVE ({} lig atoms, {} cells)",
                  nl, crec.cell_start.len() - 1);
    }
    let elec: f64 = oe.iter().map(|&v| v as f64).sum();
    let vdw: f64 = ov.iter().map(|&v| v as f64).sum();
    Some((elec, vdw))
}

/// CPU fallback / non-CUDA build.
#[cfg(not(feature = "cuda"))]
pub fn full_energy_gpu(
    _field: &ReceptorField,
    _crec: &CudaReceptor,
    _lig_coords: &[[f64; 3]],
    _lig: &DNADockingModel,
) -> Option<(f64, f64)> {
    None
}

/// True when a usable CUDA device is present (drives `supports_batch`).
#[cfg(feature = "cuda")]
pub fn cuda_available() -> bool {
    extern "C" {
        fn cudaGetDeviceCount(count: *mut i32) -> i32;
    }
    let mut n: i32 = 0;
    unsafe { cudaGetDeviceCount(&mut n) == 0 && n > 0 }
}

/// Non-CUDA build: no GPU.
#[cfg(not(feature = "cuda"))]
pub fn cuda_available() -> bool {
    false
}

/// Score many poses in one batched GPU kernel launch (near pairs + far field).
/// The host transforms ligand coordinates for every pose (cheap) and uploads a
/// single N×nl×3 buffer; one kernel with gridDim.y = N returns per-pose scores.
/// Returns `None` on any failure so the caller falls back to per-pose CPU.
#[cfg(feature = "cuda")]
pub fn batch_energy_gpu_scores(
    dna: &DNA,
    translations: &[[f64; 3]],
    rotations: &[Quaternion],
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
            l_vdwr: *const f32, l_heavy: *const u8, nl: i32, n_pose: i32, out: *mut f64,
        ) -> i32;
    }
    let n_pose = translations.len();
    let nl = dna.ligand.coordinates.len();
    if n_pose == 0 || nl == 0 {
        return None;
    }
    let field = dna.field.get_or_init(|| {
        ReceptorField::build(&dna.receptor.coordinates, &dna.receptor.ele_charges)
    });
    let crec = dna
        .rec_cuda
        .get_or_init(|| CudaReceptor::build(&dna.receptor));
    if field.phi.is_empty() {
        return None;
    }

    // Upload only the reference ligand coords (once per call, ~nl*3*4 B) plus
    // N×7 pose parameters; the kernel rotates/translates on the device in f64.
    // Previously we CPU-transformed N*nl*3 coords (~30 MB per step at n=200) —
    // that host-side work and memcpy is now amortised into the kernel.
    let base: Vec<f32> = dna
        .ligand
        .coordinates
        .iter()
        .flat_map(|c| c.iter().map(|&v| v as f32))
        .collect();
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
    let le: Vec<f32> = dna.ligand.ele_charges.iter().map(|&q| q as f32).collect();
    let lsv: Vec<f32> = dna.ligand.sqrt_vdw_charges.iter().map(|&q| q as f32).collect();
    let lv: Vec<f32> = dna.ligand.vdw_radii.iter().map(|&r| r as f32).collect();
    let lh: Vec<u8> = dna.ligand.heavy.iter().map(|&h| h as u8).collect();
    let mut out = vec![0.0f64; n_pose * 2];

    let ret = unsafe {
        cuda_batch_score(
            field.phi.as_ptr(), field.n[0] as i32, field.n[1] as i32, field.n[2] as i32,
            field.origin[0] as f32, field.origin[1] as f32, field.origin[2] as f32,
            field.spacing as f32,
            crec.r_coords.as_ptr(), crec.r_ele.as_ptr(), crec.r_svdw.as_ptr(),
            crec.r_vdwr.as_ptr(), crec.r_heavy.as_ptr(), (crec.r_coords.len() / 3) as i32,
            crec.cell_start.as_ptr(), crec.cell_atoms.as_ptr(),
            crec.ncx, crec.ncy, crec.ncz, crec.c_ox, crec.c_oy, crec.c_oz, crec.c_sp,
            base.as_ptr(), poses.as_ptr(), le.as_ptr(), lsv.as_ptr(), lv.as_ptr(), lh.as_ptr(),
            nl as i32, n_pose as i32, out.as_mut_ptr(),
        )
    };
    if ret != 0 {
        return None;
    }
    use std::sync::atomic::{AtomicBool, Ordering};
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        eprintln!("[gpu_score] CUDA BATCH scoring ACTIVE ({} poses × {} atoms)",
                  n_pose, nl);
    }
    const FACTOR: f64 = 332.0;
    const EPSILON: f64 = 4.0;
    Some(
        (0..n_pose)
            .map(|k| -(out[2 * k] * FACTOR / EPSILON + out[2 * k + 1]))
            .collect(),
    )
}

/// Non-CUDA build / GPU failure → caller falls back to per-pose CPU.
#[cfg(not(feature = "cuda"))]
pub fn batch_energy_gpu_scores(
    _dna: &DNA,
    _translations: &[[f64; 3]],
    _rotations: &[Quaternion],
) -> Option<Vec<f64>> {
    None
}
