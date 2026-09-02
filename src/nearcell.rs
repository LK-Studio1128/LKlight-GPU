//! Generic receptor near-pair cell list (CPU), shared by the grid-accelerated
//! `energy_grid` paths of vdw / pydock / cpydock.
//!
//! Cell edge == the near-window cutoff (10 Å): any pair closer than the cutoff
//! lies in the same or an adjacent cell, so testing the 3×3×3 neighbourhood
//! enumerates *every* contributing pair exactly — no approximation. The result
//! is the same pair set as the all-vs-all scan, so vdw/pydock/cpydock scores
//! computed through the cell path differ from their exact references only by
//! f64 summation order (~1e-13), i.e. numerically equivalent.

/// Uniform receptor grid over the receptor bounding box.
pub struct NearCells {
    pub cell: f64,
    pub lo: [f64; 3],
    pub n: [i64; 3],
    pub cell_start: Vec<i64>, // n[0]*n[1]*n[2]+1 prefix sums
    pub cell_atoms: Vec<i64>, // atom indices per cell (contiguous runs)
    pub n_atoms: usize,
}

impl NearCells {
    pub fn build(coords: &[[f64; 3]], cell: f64) -> Self {
        let n_atoms = coords.len();
        if n_atoms == 0 || cell <= 0.0 {
            return NearCells {
                cell: cell.max(1e-6),
                lo: [0.0; 3],
                n: [1, 1, 1],
                cell_start: vec![0, 0],
                cell_atoms: Vec::new(),
                n_atoms,
            };
        }
        let mut mn = [f64::INFINITY; 3];
        let mut mx = [f64::NEG_INFINITY; 3];
        for c in coords.iter() {
            for d in 0..3 {
                if c[d] < mn[d] { mn[d] = c[d]; }
                if c[d] > mx[d] { mx[d] = c[d]; }
            }
        }
        let lo = [mn[0] - 1.0, mn[1] - 1.0, mn[2] - 1.0];
        let span = [mx[0] - lo[0] + 1.0, mx[1] - lo[1] + 1.0, mx[2] - lo[2] + 1.0];
        let n = [
            ((span[0] / cell).ceil() as i64).max(1),
            ((span[1] / cell).ceil() as i64).max(1),
            ((span[2] / cell).ceil() as i64).max(1),
        ];
        let total = (n[0] * n[1] * n[2]) as usize;
        let mut counts = vec![0i64; total + 1];
        let key = |c: &[f64; 3]| -> usize {
            let ix = (((c[0] - lo[0]) / cell).floor() as i64).clamp(0, n[0] - 1);
            let iy = (((c[1] - lo[1]) / cell).floor() as i64).clamp(0, n[1] - 1);
            let iz = (((c[2] - lo[2]) / cell).floor() as i64).clamp(0, n[2] - 1);
            ((iz * n[1] + iy) * n[0] + ix) as usize
        };
        for c in coords.iter() {
            counts[key(c) + 1] += 1;
        }
        for i in 0..total {
            counts[i + 1] += counts[i];
        }
        let mut cell_atoms = vec![0i64; n_atoms];
        let mut cursor = counts.clone();
        for (idx, c) in coords.iter().enumerate() {
            let k = key(c);
            let pos = cursor[k] as usize;
            cell_atoms[pos] = idx as i64;
            cursor[k] += 1;
        }
        NearCells {
            cell,
            lo,
            n,
            cell_start: counts,
            cell_atoms,
            n_atoms,
        }
    }

    /// Visit every receptor atom in the 27 cells around (x, y, z). The caller
    /// tests the actual pair distance (all pairs within one cell of the point
    /// may be farther than its cutoff; the 27-cell window is exact for
    /// d ≤ cell).
    #[inline]
    pub fn for_each_near(&self, x: f64, y: f64, z: f64, f: &mut impl FnMut(usize)) {
        if self.n_atoms == 0 {
            return;
        }
        let cx = (((x - self.lo[0]) / self.cell).floor() as i64).clamp(0, self.n[0] - 1);
        let cy = (((y - self.lo[1]) / self.cell).floor() as i64).clamp(0, self.n[1] - 1);
        let cz = (((z - self.lo[2]) / self.cell).floor() as i64).clamp(0, self.n[2] - 1);
        for dz in -1..=1i64 {
            let iz = cz + dz;
            if iz < 0 || iz >= self.n[2] { continue; }
            for dy in -1..=1i64 {
                let iy = cy + dy;
                if iy < 0 || iy >= self.n[1] { continue; }
                for dx in -1..=1i64 {
                    let ix = cx + dx;
                    if ix < 0 || ix >= self.n[0] { continue; }
                    let cell = ((iz * self.n[1] + iy) * self.n[0] + ix) as usize;
                    let b = self.cell_start[cell] as usize;
                    let e = self.cell_start[cell + 1] as usize;
                    for p in b..e {
                        f(self.cell_atoms[p] as usize);
                    }
                }
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn near_set_matches_allpairs() {
        let mut rng = 7u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng % 200) as f64 / 10.0 - 10.0
        };
        let coords: Vec<[f64; 3]> = (0..300).map(|_| [next(), next(), next()]).collect();
        let cell = 5.0f64;
        let cut2 = cell * cell;
        let nc = NearCells::build(&coords, cell);
        let mut near: Vec<(usize, usize)> = Vec::new();
        for (j, c) in coords.iter().enumerate() {
            nc.for_each_near(c[0], c[1], c[2], &mut |i| {
                let dx = c[0] - coords[i][0];
                let dy = c[1] - coords[i][1];
                let dz = c[2] - coords[i][2];
                if i != j && dx * dx + dy * dy + dz * dz <= cut2 {
                    near.push((i.min(j), i.max(j)));
                }
            });
        }
        near.sort_unstable();
        near.dedup();
        let mut all: Vec<(usize, usize)> = Vec::new();
        for i in 0..coords.len() {
            for j in (i + 1)..coords.len() {
                let dx = coords[i][0] - coords[j][0];
                let dy = coords[i][1] - coords[j][1];
                let dz = coords[i][2] - coords[j][2];
                if dx * dx + dy * dy + dz * dz <= cut2 {
                    all.push((i, j));
                }
            }
        }
        assert_eq!(near.len(), all.len(),
            "cell enumerated {} pairs, all-pairs {} (no miss/dup)",
            near.len(), all.len());
    }
}
