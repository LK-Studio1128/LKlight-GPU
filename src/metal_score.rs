//! Apple GPU (Metal) batch scorer bridge — Rust side of src/metal/lk_metal.m.
//!
//! Mirrors [`crate::gpu_score`] 1:1 so the two accelerators are drop-in
//! interchangeable behind `Score::batch_energy`:
//!
//!   cuda  feature -> `batch_energy_gpu_scores`  (NVIDIA, full_score.cu)
//!   metal feature -> `batch_energy_metal_scores` (Apple Silicon, lk_metal.m)
//!
//! The host transforms every pose's ligand coordinates in f64 (identical math
//! to the CPU grid path / the validated Metal POC dump) and hands pose-major
//! float3 to the shim, which zero-copies it into a shared MTLBuffer — the
//! Apple GPU has no fp64, so keeping the rigid transform on the CPU (cheap,
//! ~10 MFLOP per 1000-pose step) sidesteps that entirely.
//!
//! Static receptor/field/ligand data is copied into Metal buffers once at
//! context creation (cached on [`crate::dna::DNA`]); only the per-step
//! coordinates change. Any failure returns `None` so the caller falls back to
//! per-pose CPU grid scoring — docking never breaks without a usable GPU.

use crate::dna::{DNA, DNADockingModel};
use crate::grid_dna::ReceptorField;
use crate::gpu_family::FamilyGpu;
use crate::gpu_score::{CudaReceptor, FLAGS_DNA};
use crate::qt::{rot3_apply, Quaternion};
use std::ffi::c_void;

extern "C" {
    fn lk_metal_available() -> i32;
    #[allow(clippy::too_many_arguments)]
    fn lk_metal_ctx_create(
        phi: *const f32,
        nx: i32,
        ny: i32,
        nz: i32,
        ox: f32,
        oy: f32,
        oz: f32,
        sp: f32,
        r_coords: *const f32,
        r_ele: *const f32,
        r_svdw: *const f32,
        r_vdwr: *const f32,
        r_heavy: *const u8,
        nr: i32,
        cell_start: *const i32,
        cell_atoms: *const i32,
        ncx: i32,
        ncy: i32,
        ncz: i32,
        c_ox: f32,
        c_oy: f32,
        c_oz: f32,
        c_sp: f32,
        l_ele: *const f32,
        l_svdw: *const f32,
        l_vdwr: *const f32,
        l_heavy: *const u8,
        nl: i32,
        tg: i32,
        mode: i32,
        r_flag: *const u8,
        r_des: *const f32,
        r_asa: *const f32,
        l_flag: *const u8,
        l_des: *const f32,
        l_asa: *const f32,
    ) -> *mut c_void;
    fn lk_metal_cpydock_solv(
        ctx: *mut c_void,
        lc: *const f32,
        n_pose: i32,
        out_s: *mut f64,
    ) -> i32;
    fn lk_metal_score(
        ctx: *mut c_void,
        lc: *const f32,
        n_pose: i32,
        out_e: *mut f32,
        out_v: *mut f32,
    ) -> i32;
    fn lk_metal_ctx_destroy(ctx: *mut c_void);
}

/// Opaque persistent Metal context (device/pipeline/buffers). Built once per
/// [`crate::dna::DNA`] and kept in a `OnceLock<Option<…>>` so a failed create
/// is remembered and we fall back to CPU for the whole run.
pub struct MetalCtx {
    ptr: *mut c_void,
}
// The engine calls batch scoring from a single thread per step (update_luciferin),
// so the context is never used concurrently; allow sharing through the OnceLock.
unsafe impl Send for MetalCtx {}
unsafe impl Sync for MetalCtx {}

impl Drop for MetalCtx {
    fn drop(&mut self) {
        unsafe { lk_metal_ctx_destroy(self.ptr) }
    }
}

impl MetalCtx {
    fn create(
        field: &ReceptorField,
        crec: &CudaReceptor,
        lig: &DNADockingModel,
        tg: i32,
    ) -> Option<MetalCtx> {
        Self::create_raw(field, crec, lig, tg, FLAGS_DNA as i32)
    }

    /// Create a context for an all-atom family (DNA/PYDOCK/VDW) sharing the
    /// family GPU dataset. `field` is `None` for VDW (flags clear F_FAR; the
    /// shim keeps a 1-element placeholder phi).
    pub fn create_family(
        g: &FamilyGpu,
        field: Option<&ReceptorField>,
        tg: i32,
        flags: u32,
    ) -> Option<MetalCtx> {
        let nl = (g.clig.len() / 3) as i32;
        if nl == 0 {
            return None;
        }
        let mut dummy: f32 = 0.0;
        let (phi, nxd, nyd, nzd, oxf, oyf, ozf, spf) = match field {
            Some(f) if !f.phi.is_empty() => (
                f.phi.as_ptr(), f.n[0] as i32, f.n[1] as i32, f.n[2] as i32,
                f.origin[0] as f32, f.origin[1] as f32, f.origin[2] as f32, f.spacing as f32,
            ),
            _ if flags & crate::gpu_score::F_FAR != 0 => return None,
            _ => (&dummy as *const f32, 1, 1, 1, 0.0f32, 0.0f32, 0.0f32, 1.0f32),
        };
        let crec = &g.crec;
        let ptr = unsafe {
            lk_metal_ctx_create(
                phi, nxd, nyd, nzd, oxf, oyf, ozf, spf,
                crec.r_coords.as_ptr(), crec.r_ele.as_ptr(), crec.r_svdw.as_ptr(),
                crec.r_vdwr.as_ptr(), crec.r_heavy.as_ptr(), (crec.r_coords.len() / 3) as i32,
                crec.cell_start.as_ptr(), crec.cell_atoms.as_ptr(),
                crec.ncx, crec.ncy, crec.ncz, crec.c_ox, crec.c_oy, crec.c_oz, crec.c_sp,
                g.le.as_ptr(), g.lsv.as_ptr(), g.lv.as_ptr(), g.lh.as_ptr(),
                nl, tg, flags as i32,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(MetalCtx { ptr })
        }
    }

    /// (internal) raw create with an explicit mode word.
    fn create_raw(
        field: &ReceptorField,
        crec: &CudaReceptor,
        lig: &DNADockingModel,
        tg: i32,
        mode: i32,
    ) -> Option<MetalCtx> {
        let nl = lig.coordinates.len();
        if nl == 0 || field.phi.is_empty() {
            return None;
        }
        let le: Vec<f32> = lig.ele_charges.iter().map(|&q| q as f32).collect();
        let lsv: Vec<f32> = lig.sqrt_vdw_charges.iter().map(|&q| q as f32).collect();
        let lv: Vec<f32> = lig.vdw_radii.iter().map(|&r| r as f32).collect();
        let lh: Vec<u8> = lig.heavy.iter().map(|&h| h as u8).collect();
        let ptr = unsafe {
            lk_metal_ctx_create(
                field.phi.as_ptr(),
                field.n[0] as i32,
                field.n[1] as i32,
                field.n[2] as i32,
                field.origin[0] as f32,
                field.origin[1] as f32,
                field.origin[2] as f32,
                field.spacing as f32,
                crec.r_coords.as_ptr(),
                crec.r_ele.as_ptr(),
                crec.r_svdw.as_ptr(),
                crec.r_vdwr.as_ptr(),
                crec.r_heavy.as_ptr(),
                (crec.r_coords.len() / 3) as i32,
                crec.cell_start.as_ptr(),
                crec.cell_atoms.as_ptr(),
                crec.ncx,
                crec.ncy,
                crec.ncz,
                crec.c_ox,
                crec.c_oy,
                crec.c_oz,
                crec.c_sp,
                le.as_ptr(),
                lsv.as_ptr(),
                lv.as_ptr(),
                lh.as_ptr(),
                nl as i32,
                tg,
                mode,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(MetalCtx { ptr })
        }
    }
}

/// True when a usable Metal device is present (drives `Score::supports_batch`).
#[cfg(feature = "metal")]
pub fn metal_available() -> bool {
    unsafe { lk_metal_available() != 0 }
}

#[cfg(not(feature = "metal"))]
pub fn metal_available() -> bool {
    false
}

/// Threadgroup size for the kernel tree reduction — must be a power of two.
/// Measured optimum on Apple M4 (1AZP, 506 ligand atoms): 128.
pub const METAL_TG: i32 = 128;

/// Score many poses in one Metal dispatch. Returns per-pose total grid energy
/// (same convention as `gpu_score::batch_energy_gpu_scores`), or `None` on any
/// failure so the caller falls back to per-pose CPU scoring.
#[cfg(feature = "metal")]
pub fn batch_energy_metal_scores(
    dna: &DNA,
    translations: &[[f64; 3]],
    rotations: &[Quaternion],
) -> Option<Vec<f64>> {
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
    let ctx = dna.lig_metal.get_or_init(|| {
        MetalCtx::create(field, crec, &dna.ligand, METAL_TG)
    });
    let ctx = match ctx {
        Some(c) => c,
        None => return None,
    };

    // Identical f64 rigid transform to the CPU grid path / Metal POC dump:
    // pose-major float3, rotations[k] as matrix then rot3_apply.
    let mut lc: Vec<f32> = Vec::with_capacity(n_pose * nl * 3);
    for (t, r) in translations.iter().zip(rotations.iter()) {
        let rot = r.to_matrix();
        for c in dna.ligand.coordinates.iter() {
            let p = rot3_apply(&rot, *c);
            lc.push((p[0] + t[0]) as f32);
            lc.push((p[1] + t[1]) as f32);
            lc.push((p[2] + t[2]) as f32);
        }
    }
    let mut oe = vec![0.0f32; n_pose];
    let mut ov = vec![0.0f32; n_pose];
    let ret = unsafe {
        lk_metal_score(
            ctx.ptr,
            lc.as_ptr(),
            n_pose as i32,
            oe.as_mut_ptr(),
            ov.as_mut_ptr(),
        )
    };
    if ret != 0 {
        return None;
    }
    use std::sync::atomic::{AtomicBool, Ordering};
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[metal_score] Metal BATCH scoring ACTIVE ({} poses × {} atoms, device present)",
            n_pose, nl
        );
    }
    const FACTOR: f64 = 332.0;
    const EPSILON: f64 = 4.0;
    Some(
        (0..n_pose)
            .map(|k| -(oe[k] as f64 * FACTOR / EPSILON + ov[k] as f64))
            .collect(),
    )
}

/// Non-Metal build / no device → caller falls back to per-pose CPU.
#[cfg(not(feature = "metal"))]
pub fn batch_energy_metal_scores(
    _dna: &DNA,
    _translations: &[[f64; 3]],
    _rotations: &[Quaternion],
) -> Option<Vec<f64>> {
    None
}

/// Metal batched scoring for the all-atom families (DNA/PYDOCK/VDW) on a
/// shared per-family context. `base_coords` are the ligand reference
/// coordinates in f64 — the rigid transform runs on the CPU in f64 (identical
/// math to the CPU grid path), then the f32 float3 pose-major buffer is
/// zero-copied to the Metal shared buffer.
#[cfg(feature = "metal")]
/// Raw two-column Metal family batch: per-pose (E·332/4 in kcal, V in the
/// raw cap-1.0 convention) — unfolded so callers can apply per-family weights
/// (CPYDOCK: −(E + 0.1·V − S)) with a single kernel dispatch.
pub fn metal_family_raw(
    ctx: &MetalCtx,
    base_coords: &[[f64; 3]],
    translations: &[[f64; 3]],
    rotations: &[Quaternion],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let n_pose = translations.len();
    let nl = base_coords.len();
    if n_pose == 0 || nl == 0 {
        return None;
    }
    let mut lc: Vec<f32> = Vec::with_capacity(n_pose * nl * 3);
    for (t, r) in translations.iter().zip(rotations.iter()) {
        let rot = r.to_matrix();
        for c in base_coords.iter() {
            let pp = rot3_apply(&rot, *c);
            lc.push((pp[0] + t[0]) as f32);
            lc.push((pp[1] + t[1]) as f32);
            lc.push((pp[2] + t[2]) as f32);
        }
    }
    let mut oe = vec![0.0f32; n_pose];
    let mut ov = vec![0.0f32; n_pose];
    let ret = unsafe {
        lk_metal_score(
            ctx.ptr,
            lc.as_ptr(),
            n_pose as i32,
            oe.as_mut_ptr(),
            ov.as_mut_ptr(),
        )
    };
    if ret != 0 {
        return None;
    }
    use std::sync::atomic::{AtomicBool, Ordering};
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        eprintln!("[metal_score] Metal family BATCH scoring ACTIVE ({} poses × {} atoms)",
                  n_pose, nl);
    }
    const FACTOR: f64 = 332.0;
    const EPSILON: f64 = 4.0;
    Some((
        (0..n_pose).map(|k| oe[k] as f64 * FACTOR / EPSILON).collect(),
        (0..n_pose).map(|k| ov[k] as f64).collect(),
    ))
}

/// Folded Metal family batch: per-pose score = −(E·332/4 + V).
pub fn batch_metal_family_scores(
    ctx: &MetalCtx,
    base_coords: &[[f64; 3]],
    translations: &[[f64; 3]],
    rotations: &[Quaternion],
) -> Option<Vec<f64>> {
    metal_family_raw(ctx, base_coords, translations, rotations)
        .map(|(e, v)| e.iter().zip(v.iter()).map(|(e, v)| -(e + v)).collect())
}

/// Two-column Metal family batch as (E·332/4, V).
pub fn batch_metal_family_parts(
    ctx: &MetalCtx,
    base_coords: &[[f64; 3]],
    translations: &[[f64; 3]],
    rotations: &[Quaternion],
) -> Option<(Vec<f64>, Vec<f64>)> {
    metal_family_raw(ctx, base_coords, translations, rotations)
}

/// Create a Metal context for CPYDOCK: the family kernel runs with
/// flags = FLAGS_PYDOCK (FAR|ELEC, mode = 3 | F_DESOLV) so the *same*
/// `lk_metal_score` call produces the T = -(E·332/4 + V) component, and the
/// context additionally carries the desolvation pipelines + arrays consumed by
/// [`batch_metal_cpydock_solv`] (two-stage min + reduce kernels).
#[cfg(feature = "metal")]
impl MetalCtx {
    pub fn create_cpydock(
        g: &FamilyGpu,
        field: Option<&ReceptorField>,
        tg: i32,
        solv_r: &crate::gpu_family::CpySolvSide,
        solv_l: &crate::gpu_family::CpySolvSide,
    ) -> Option<MetalCtx> {
        let nl = (g.clig.len() / 3) as i32;
        let nr = (g.crec.r_coords.len() / 3) as i32;
        if nl == 0 || nr == 0 || solv_r.flag.len() != nr as usize
            || solv_l.flag.len() != nl as usize {
            return None;
        }
        let mut dummy: f32 = 0.0;
        let (phi, nxd, nyd, nzd, oxf, oyf, ozf, spf) = match field {
            Some(f) if !f.phi.is_empty() => (
                f.phi.as_ptr(), f.n[0] as i32, f.n[1] as i32, f.n[2] as i32,
                f.origin[0] as f32, f.origin[1] as f32, f.origin[2] as f32, f.spacing as f32,
            ),
            _ => (&dummy as *const f32, 1, 1, 1, 0.0f32, 0.0f32, 0.0f32, 1.0f32),
        };
        let crec = &g.crec;
        let mode = crate::gpu_family::family_flags("pydock") | crate::gpu_score::F_DESOLV;
        let ptr = unsafe {
            lk_metal_ctx_create(
                phi, nxd, nyd, nzd, oxf, oyf, ozf, spf,
                crec.r_coords.as_ptr(), crec.r_ele.as_ptr(), crec.r_svdw.as_ptr(),
                crec.r_vdwr.as_ptr(), crec.r_heavy.as_ptr(), nr,
                crec.cell_start.as_ptr(), crec.cell_atoms.as_ptr(),
                crec.ncx, crec.ncy, crec.ncz, crec.c_ox, crec.c_oy, crec.c_oz, crec.c_sp,
                g.le.as_ptr(), g.lsv.as_ptr(), g.lv.as_ptr(), g.lh.as_ptr(),
                nl, tg, mode as i32,
                solv_r.flag.as_ptr(), solv_r.des.as_ptr(), solv_r.asa.as_ptr(),
                solv_l.flag.as_ptr(), solv_l.des.as_ptr(), solv_l.asa.as_ptr(),
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(MetalCtx { ptr })
        }
    }
}

/// CPYDOCK desolvation pass on Metal: two-stage (per-pose atomic min over the
/// cell list, then a one-threadgroup-per-pose reduce of g(min)·des). Returns
/// per-pose S (already in the sign used by score = -(E·332/4 + 0.1·V − S)).
#[cfg(feature = "metal")]
pub fn batch_metal_cpydock_solv(
    ctx: &MetalCtx,
    base_coords: &[[f64; 3]],
    translations: &[[f64; 3]],
    rotations: &[Quaternion],
) -> Option<Vec<f64>> {
    let n_pose = translations.len();
    let nl = base_coords.len();
    if n_pose == 0 || nl == 0 {
        return None;
    }
    let mut lc: Vec<f32> = Vec::with_capacity(n_pose * nl * 3);
    for (t, r) in translations.iter().zip(rotations.iter()) {
        let rot = r.to_matrix();
        for c in base_coords.iter() {
            let pp = rot3_apply(&rot, *c);
            lc.push((pp[0] + t[0]) as f32);
            lc.push((pp[1] + t[1]) as f32);
            lc.push((pp[2] + t[2]) as f32);
        }
    }
    let mut out = vec![0.0f64; n_pose];
    let ret = unsafe {
        lk_metal_cpydock_solv(ctx.ptr, lc.as_ptr(), n_pose as i32, out.as_mut_ptr())
    };
    if ret != 0 {
        return None;
    }
    use std::sync::atomic::{AtomicBool, Ordering};
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        eprintln!("[metal_score] Metal CPYDOCK desolv ACTIVE ({} poses)", n_pose);
    }
    Some(out)
}
