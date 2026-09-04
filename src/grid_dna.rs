// -*- coding: utf-8 -*-
//! GPU-ready receptor electrostatic field (grid pre-computation).
//!
//! L1 of the CUDA acceleration plan (see GPU_ACCEL_DESIGN.md). Inspired by the
//! UniDock / AutoDock-Vina affinity-grid trick:
//!
//!   * The pair loop is split at CLOSE_DIST = 10 Å.
//!     - d ≤ 10 Å  : exact per-pair terms (clamped electrostatics + Lennard-Jones
//!                   with its 1.0 cap + the heavy-atom clash penalty + interface
//!                   flags). This is a *small* neighbourhood (the old code scanned
//!                   a ±30 Å Z-window and evaluated every pair up to 30 Å).
//!     - 10 < d ≤ 30 Å : pure electrostatics. Here |q_i·q_j|/d² < 0.012 always
//!                   (max |q|≈0.9 ⇒ 0.54/100 = 0.0054), so the ±0.012 clamp can
//!                   never trigger → the interaction is *linear* in the charges
//!                   and factorises into a receptor field φ(x)=Σ_i q_i/d² sampled
//!                   on a grid, looked up by trilinear interpolation.
//!
//!   * Energy of one pose = Σ_{near pairs} exact + Σ_{lig atoms j} q_j·φ(x_j).
//!
//! The φ grid is the piece that is trivially parallelised on a GPU later
//! (one thread per ligand atom → gather/interpolate/reduce); the layout here
//! (`origin`, `n`, `spacing`, flat `phi` buffer) is deliberately kept GPU-copy
//! friendly (f32, contiguous).

pub const CLOSE_DIST: f64 = 10.0;      // below this: exact per-pair evaluation
pub const CLOSE_DIST2: f64 = CLOSE_DIST * CLOSE_DIST;
pub const FIELD_RMAX: f64 = 30.0;      // electrostatics cutoff (matches ELEC_DIST_CUTOFF)
pub const FIELD_RMIN: f64 = CLOSE_DIST; // field excludes the exact-evaluated interior
pub const SPACING: f64 = 0.5;          // grid resolution (Å); 0.5 Å chosen for near-reference
                                       // far-field accuracy (AutoDock-class 0.375–0.5 Å band). 1 Å
                                       // roughly doubles the far-field absolute error.

/// Receptor electrostatic field φ(x) = Σ_{i: 10<|x-r_i|≤30} q_i / |x-r_i|².
#[derive(Clone)]
pub struct ReceptorField {
    pub origin: [f64; 3],
    pub n: [usize; 3],
    pub spacing: f64,
    pub phi: Vec<f32>,
}

impl ReceptorField {
    /// Build the field over a box that covers the receptor plus FIELD_RMAX padding.
    pub fn build(coords: &[[f64; 3]], charges: &[f64]) -> ReceptorField {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for c in coords.iter() {
            for a in 0..3 {
                lo[a] = lo[a].min(c[a]);
                hi[a] = hi[a].max(c[a]);
            }
        }
        for a in 0..3 {
            lo[a] = (lo[a] - FIELD_RMAX - 2.0 * SPACING).floor();
            hi[a] = (hi[a] + FIELD_RMAX + 2.0 * SPACING).ceil();
        }
        let n = [
            ((hi[0] - lo[0]) / SPACING) as usize + 1,
            ((hi[1] - lo[1]) / SPACING) as usize + 1,
            ((hi[2] - lo[2]) / SPACING) as usize + 1,
        ];
        let mut phi = vec![0.0f32; n[0] * n[1] * n[2]];
        let s = SPACING;
        // Scatter-window half-width in grid units covering the whole ±FIELD_RMAX shell.
        // Derived from spacing so it stays correct at any resolution (a hardcoded ±32
        // assumed 1 Å and silently truncated the far field at finer spacing).
        let spread = ((FIELD_RMAX / s).ceil() as i64) + 2;

        // Ring-shell scatter: for each charged receptor atom, add q/d² to every grid
        // point between 10 and 30 Å away.
        //
        // OPT (2026-09-05): the per-atom window is the shell band (FIELD_RMIN,
        // FIELD_RMAX]. A brute box scan visits every point of a (2·spread)³ box and
        // *rejects* most of it (deep interior r<10 plus far corners r>30). We prune
        // the axis sweeps before applying the exact distance test:
        //
        //   * (y,z) rows with dyz² ≥ r2max can never reach the shell on x → skipped.
        //   * per surviving (y,z) row, dx² is a monotone function of the x grid index,
        //     so the only x indices that can possibly lie in (rmin,rmax] have
        //     |dx| ≤ sqrt(r2max − dyz²). We sweep exactly that contiguous x slab
        //     (grown by a ±1 safety margin and clamped to the box) instead of the
        //     whole ±FIELD_RMAX width, and still apply the *identical* exact test
        //     `r2min < d2 <= r2max` to decide every write.
        //
        // This preserves the write set and the write order bit-for-bit: every cell
        // still receives the same q/d² additions in the same per-atom order (each
        // (cell,atom) pair is written at most once), so the f32 accumulator in `phi`
        // is unchanged. The win is a large reduction of rejected distance tests —
        // at 0.5 Å spacing this claws back much of the ~8× setup cost 0.5 Å adds
        // over 1 Å, at zero accuracy cost.
        let r2min = FIELD_RMIN * FIELD_RMIN;   // 100
        let r2max = FIELD_RMAX * FIELD_RMAX;   // 900
        let nx = n[0] as i64;
        let ny = n[1] as i64;
        // Grid coordinate of the atom centre along x for the per-row slab bounds.
        let cxi = |c0: f64| (c0 - lo[0]) / s; // continuous grid x of atom centre
        for (c, &q) in coords.iter().zip(charges.iter()) {
            if q == 0.0 {
                continue;
            }
            let gi = [
                (((c[0] - lo[0]) / s).floor() as i64 - spread).max(0) as usize,
                (((c[1] - lo[1]) / s).floor() as i64 - spread).max(0) as usize,
                (((c[2] - lo[2]) / s).floor() as i64 - spread).max(0) as usize,
            ];
            let gh = [
                (((c[0] - lo[0]) / s).floor() as i64 + spread).min(nx - 1) as usize,
                (((c[1] - lo[1]) / s).floor() as i64 + spread).min(ny - 1) as usize,
                (((c[2] - lo[2]) / s).floor() as i64 + spread).min(n[2] as i64 - 1) as usize,
            ];
            let cxi_f = cxi(c[0]);
            for iz in gi[2]..=gh[2] {
                let zz = lo[2] + iz as f64 * s;
                let dz = zz - c[2];
                for iy in gi[1]..=gh[1] {
                    let yy = lo[1] + iy as f64 * s;
                    let dy = yy - c[1];
                    let dyz2 = dy * dy + dz * dz;
                    // Far-corner pruning: even at dx=0 this row is beyond rmax.
                    if dyz2 >= r2max {
                        continue;
                    }
                    // Tight x slab: indices whose dx ∈ [-b, b], b = sqrt(r2max-dyz²).
                    // dx = lo0 + ix·s − c0 is monotone increasing in ix, so these
                    // form one contiguous ix range around the atom-centre grid coord.
                    let b = (r2max - dyz2).sqrt();
                    let mut xl = ((cxi_f - b / s).floor() as i64 - 1).max(gi[0] as i64) as usize;
                    let mut xh = ((cxi_f + b / s).ceil() as i64 + 1).min(gh[0] as i64) as usize;
                    // bounds must stay within [gi,gh] AND valid ascending
                    if xl > gh[0] || xh < gi[0] {
                        continue;
                    }
                    xl = xl.max(gi[0]);
                    xh = xh.min(gh[0]);
                    let base = (iz as i64 * ny + iy as i64) * nx;
                    let mut ix = xl;
                    while ix <= xh {
                        let dx = lo[0] + ix as f64 * s - c[0];
                        let d2 = dx * dx + dyz2;
                        if d2 > r2min && d2 <= r2max {
                            phi[base as usize + ix] += (q / d2) as f32;
                        }
                        ix += 1;
                    }
                }
            }
        }
        ReceptorField { origin: lo, n, spacing: s, phi }
    }

    #[inline]
    fn index(&self, ix: usize, iy: usize, iz: usize) -> f32 {
        self.phi[(iz * self.n[1] + iy) * self.n[0] + ix]
    }

    /// Trilinear interpolation of φ at an arbitrary point x. Out-of-box points
    /// return 0.0 (they are beyond the 30 Å cutoff by construction).
    #[inline]
    pub fn sample(&self, x: f64, y: f64, z: f64) -> f64 {
        let s = self.spacing;
        let fx = (x - self.origin[0]) / s;
        let fy = (y - self.origin[1]) / s;
        let fz = (z - self.origin[2]) / s;
        let ix = fx.floor();
        let iy = fy.floor();
        let iz = fz.floor();
        if ix < 0.0 || iy < 0.0 || iz < 0.0 {
            return 0.0;
        }
        let ix0 = ix as usize;
        let iy0 = iy as usize;
        let iz0 = iz as usize;
        if ix0 + 1 >= self.n[0] || iy0 + 1 >= self.n[1] || iz0 + 1 >= self.n[2] {
            return 0.0;
        }
        let tx = (fx - ix) as f32;
        let ty = (fy - iy) as f32;
        let tz = (fz - iz) as f32;
        let c000 = self.index(ix0, iy0, iz0) as f64;
        let c100 = self.index(ix0 + 1, iy0, iz0) as f64;
        let c010 = self.index(ix0, iy0 + 1, iz0) as f64;
        let c110 = self.index(ix0 + 1, iy0 + 1, iz0) as f64;
        let c001 = self.index(ix0, iy0, iz0 + 1) as f64;
        let c101 = self.index(ix0 + 1, iy0, iz0 + 1) as f64;
        let c011 = self.index(ix0, iy0 + 1, iz0 + 1) as f64;
        let c111 = self.index(ix0 + 1, iy0 + 1, iz0 + 1) as f64;
        let tx = tx as f64;
        let ty = ty as f64;
        let tz = tz as f64;
        let c00 = c000 * (1.0 - tx) + c100 * tx;
        let c10 = c010 * (1.0 - tx) + c110 * tx;
        let c01 = c001 * (1.0 - tx) + c101 * tx;
        let c11 = c011 * (1.0 - tx) + c111 * tx;
        let c0 = c00 * (1.0 - ty) + c10 * ty;
        let c1 = c01 * (1.0 - ty) + c11 * ty;
        c0 * (1.0 - tz) + c1 * tz
    }

    /// Total far-field contribution for a set of ligand atom coordinates/charges:
    /// Σ_j q_j · φ(x_j)  (in the same "raw e²/Å²" units as the near electrostatics).
    pub fn far_field_energy(&self, lig_coords: &[[f64; 3]], lig_charges: &[f64]) -> f64 {
        let mut e = 0.0f64;
        for (c, &q) in lig_coords.iter().zip(lig_charges.iter()) {
            if q != 0.0 {
                e += q * self.sample(c[0], c[1], c[2]);
            }
        }
        e
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_symmetric_and_local() {
        // one charge at origin: φ = 1/d² for 10 < d ≤ 30, 0 elsewhere
        let coords = vec![[0.0, 0.0, 0.0]];
        let charges = vec![1.0];
        let f = ReceptorField::build(&coords, &charges);
        let at_12 = f.sample(12.0, 0.0, 0.0);
        let at_neg12 = f.sample(-12.0, 0.0, 0.0);
        let at_5 = f.sample(5.0, 0.0, 0.0);
        let at_28 = f.sample(28.0, 0.0, 0.0);
        let at_35 = f.sample(35.0, 0.0, 0.0);
        assert!((at_12 - at_neg12).abs() < 1e-6, "{at_12} vs {at_neg12}");
        assert!((at_12 - 1.0 / 144.0).abs() < 1e-4, "got {at_12}");
        assert!(at_5.abs() < 1e-9, "interior (r<10) should be 0, got {at_5}");
        assert!((at_28 - 1.0 / 784.0).abs() < 1e-5, "r=28 inside cutoff, got {at_28}");
        assert!(at_35.abs() < 1e-9, "beyond 30 Å cutoff should be 0, got {at_35}");
    }
}
