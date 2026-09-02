//! Thin Rust bridge to the CUDA far-field kernel (feature `cuda`).
//!
//! When the `cuda` feature is enabled and a device is available, the
//! 10-30 Å electrostatics far-field term E_far = Σ_j q_j·φ(x_j) is computed by
//! `src/cuda/far_field.cu` (one GPU thread per ligand atom, trilinear gather,
//! atomic reduction). Any failure (no device, driver issue, kernel error)
//! transparently falls back to the CPU trilinear implementation so docking
//! never breaks on machines without CUDA.

use crate::grid_dna::ReceptorField;

/// Compute the far-field energy on the GPU. Returns `None` if CUDA is
/// unavailable or the launch fails (caller falls back to the CPU path).
#[cfg(feature = "cuda")]
pub fn far_field_energy_gpu(
    field: &ReceptorField,
    lig_coords: &[[f64; 3]],
    lig_charges: &[f64],
) -> Option<f64> {
    extern "C" {
        fn cuda_far_field(
            phi: *const f32,
            nx: i32,
            ny: i32,
            nz: i32,
            ox: f32,
            oy: f32,
            oz: f32,
            sp: f32,
            coords: *const f32,
            charges: *const f32,
            n: i32,
            result: *mut f64,
        ) -> i32;
    }
    let n = lig_coords.len();
    if n == 0 || field.phi.is_empty() {
        return None;
    }
    // coords/charges → contiguous f32 buffers
    let mut coords_f = Vec::with_capacity(n * 3);
    for c in lig_coords.iter() {
        coords_f.push(c[0] as f32);
        coords_f.push(c[1] as f32);
        coords_f.push(c[2] as f32);
    }
    let charges_f: Vec<f32> = lig_charges.iter().map(|&q| q as f32).collect();

    let mut result: f64 = 0.0;
    let ret = unsafe {
        cuda_far_field(
            field.phi.as_ptr(),
            field.n[0] as i32,
            field.n[1] as i32,
            field.n[2] as i32,
            field.origin[0] as f32,
            field.origin[1] as f32,
            field.origin[2] as f32,
            field.spacing as f32,
            coords_f.as_ptr(),
            charges_f.as_ptr(),
            n as i32,
            &mut result,
        )
    };
    if ret == 0 {
        use std::sync::atomic::{AtomicBool, Ordering};
        static LOGGED: AtomicBool = AtomicBool::new(false);
        if !LOGGED.swap(true, Ordering::Relaxed) {
            eprintln!("[gpu_field] CUDA far-field ACTIVE (grid {}x{}x{}, {} lig atoms)",
                      field.n[0], field.n[1], field.n[2], n);
        }
        Some(result)
    } else {
        None
    }
}

/// CPU fallback / non-CUDA build: use the host trilinear implementation.
#[cfg(not(feature = "cuda"))]
pub fn far_field_energy_gpu(
    field: &ReceptorField,
    lig_coords: &[[f64; 3]],
    lig_charges: &[f64],
) -> Option<f64> {
    let _ = field;
    let _ = lig_coords;
    let _ = lig_charges;
    None
}
