use super::glowworm::{distance_sq, Glowworm};
use super::qt::Quaternion;
use super::scoring::Score;
use rand::Rng;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Error, Write};

pub struct Swarm<'a> {
    pub glowworms: Vec<Glowworm<'a>>,
    pos_scratch:   Vec<[f64; 3]>,
    rot_scratch:   Vec<Quaternion>,
}

impl<'a> Default for Swarm<'a> {
    fn default() -> Self {
        Swarm::new()
    }
}

impl<'a> Swarm<'a> {
    pub fn new() -> Self {
        Swarm {
            glowworms:   Vec::new(),
            pos_scratch: Vec::new(),
            rot_scratch: Vec::new(),
        }
    }

    pub fn add_glowworms(
        &mut self,
        positions: &[Vec<f64>],
        scoring: &'a Box<dyn Score>,
        use_anm: bool,
        rec_num_anm: usize,
        lig_num_anm: usize,
    ) {
        for (i, position) in positions.iter().enumerate() {
            // Translation component
            let translation = [position[0], position[1], position[2]];
            // Rotation component
            let rotation = Quaternion::new(position[3], position[4], position[5], position[6]);
            // ANM for receptor
            let mut rec_nmodes: Vec<f64> = Vec::new();
            if use_anm && rec_num_anm > 0 {
                for j in 7..7 + rec_num_anm {
                    rec_nmodes.push(positions[i][j]);
                }
            }
            // ANM for ligand
            let mut lig_nmodes: Vec<f64> = Vec::new();
            if use_anm && lig_num_anm > 0 {
                for j in 7 + rec_num_anm..positions[i].len() {
                    lig_nmodes.push(positions[i][j]);
                }
            }
            let glowworm = Glowworm::new(
                i as u32,
                translation,
                rotation,
                rec_nmodes,
                lig_nmodes,
                scoring,
                use_anm,
            );
            self.glowworms.push(glowworm);
        }
    }

    pub fn update_luciferin(&mut self) {
        if self.glowworms.is_empty() {
            return;
        }
        // Batched GPU path (DNA + CUDA device): score all moved glowworms in a
        // single kernel launch, then update luciferin/step with identical
        // semantics to compute_luciferin. Non-batch scorers keep the rayon
        // per-glowworm path unchanged.
        if self.glowworms[0].scoring_function.supports_batch() {
            let need: Vec<usize> = self
                .glowworms
                .iter()
                .enumerate()
                .filter(|(_, g)| g.moved || g.step == 0)
                .map(|(i, _)| i)
                .collect();
            if !need.is_empty() {
                let tr: Vec<[f64; 3]> = need.iter().map(|&i| self.glowworms[i].translation).collect();
                let ro: Vec<Quaternion> = need.iter().map(|&i| self.glowworms[i].rotation).collect();
                let scores = self.glowworms[0].scoring_function.batch_energy(&tr, &ro);
                for (k, &i) in need.iter().enumerate() {
                    if k < scores.len() {
                        self.glowworms[i].scoring = scores[k];
                    }
                }
            }
            for g in self.glowworms.iter_mut() {
                g.luciferin = (1.0 - g.rho) * g.luciferin + g.gamma * g.scoring;
                g.step += 1;
            }
            return;
        }
        self.glowworms.par_iter_mut().for_each(|gw| gw.compute_luciferin());
    }

    pub fn movement_phase(&mut self, rng: &mut rand::prelude::StdRng) {
        let n = self.glowworms.len();
        if n == 0 {
            return;
        }

        // ── G0: snapshot pre-move poses into reusable scratch (parallel). All
        //    moves in this step read these frozen positions/rotations, exactly as
        //    the serial reference did (move_towards targets a pre-move partner).
        self.pos_scratch.resize(n, [0.0; 3]);
        self.rot_scratch.resize(n, Quaternion::new(1.0, 0.0, 0.0, 0.0));
        self.glowworms
            .par_iter()
            .zip(self.pos_scratch.par_iter_mut().zip(self.rot_scratch.par_iter_mut()))
            .for_each(|(gw, (ps, rs))| {
                *ps = gw.translation;
                *rs = gw.rotation;
            });
        // ANM still cloned (per-glowworm snapshot; empty when use_anm=false)
        let anm_recs: Vec<Vec<f64>> = self.glowworms.iter().map(|gw| gw.rec_nmodes.clone()).collect();
        let anm_ligs: Vec<Vec<f64>> = self.glowworms.iter().map(|gw| gw.lig_nmodes.clone()).collect();

        // ── G1: neighbor search. Each glowworm keeps every *higher-luciferin*
        //    peer within vision range. The scan is what a naive implementation
        //    does pairwise in O(n²); for large swarms we route it through a
        //    spatial hash (cell edge = max vision range, so any pair within
        //    range lies in the same or an adjacent cell) with the candidate set
        //    re-sorted by id so the resulting neighbor list — and therefore
        //    probabilities, random draws and moves — is bit-identical.
        let neighbors: Vec<Vec<u32>> = if n >= 64 {
            self.spatial_neighbor_lists()
        } else {
            self.glowworms
                .par_iter()
                .map(|g1| {
                    let vr2 = g1.vision_range * g1.vision_range;
                    self.glowworms
                        .iter()
                        .filter(|g2| g2.id != g1.id && g1.luciferin < g2.luciferin
                                     && distance_sq(g1, g2) < vr2)
                        .map(|g2| g2.id)
                        .collect()
                })
                .collect()
        };

        // ── G2: install neighbor lists + moving probabilities (parallel; each
        //    glowworm only touches its own fields, reads the frozen luciferin
        //    snapshot — identical results to the serial loop).
        let luciferins: Vec<f64> = self.glowworms.iter().map(|gw| gw.luciferin).collect();
        self.glowworms
            .par_iter_mut()
            .zip(neighbors.into_par_iter())
            .for_each(|(gw, nbrs)| {
                gw.neighbors = nbrs;
                gw.compute_probability_moving_toward_neighbor(&luciferins);
            });

        // ── G3: pre-generate randoms (serial — keeps the reference RNG draw
        //    order identical), then move all glowworms in parallel.
        let randoms: Vec<f64> = (0..n).map(|_| rng.gen()).collect();
        let (gws, pos_s, rot_s) = (&mut self.glowworms, &self.pos_scratch, &self.rot_scratch);
        gws.par_iter_mut()
            .zip(randoms.par_iter())
            .for_each(|(gw, &r)| {
                let nid = gw.select_random_neighbor(r) as usize;
                gw.move_towards(nid as u32, &pos_s[nid], &rot_s[nid], &anm_recs[nid], &anm_ligs[nid]);
                gw.update_vision_range();
            });
    }

    /// Neighbor lists via a uniform spatial hash instead of the O(n²) all-pairs
    /// scan. Cell edge = the swarm's max vision range (5.0 by default): two
    /// points closer than that always fall in the same or an adjacent cell, so
    /// testing the 3×3×3 neighbourhood is exact — no approximation. Results are
    /// id-sorted, matching the reference scan's ascending-id collection order.
    fn spatial_neighbor_lists(&self) -> Vec<Vec<u32>> {
        use std::collections::HashMap;
        let n = self.glowworms.len();
        // Safety net: vision range is capped at max_vision_range per glowworm;
        // use the widest one so the 27-cell window always covers d < vr.
        let cell = self
            .glowworms
            .iter()
            .map(|g| g.max_vision_range)
            .fold(0.0f64, f64::max)
            .max(1e-6);
        let mut cells: HashMap<(i64, i64, i64), Vec<u32>> =
            HashMap::with_capacity(n);
        for gw in &self.glowworms {
            let key = (
                (gw.translation[0] / cell).floor() as i64,
                (gw.translation[1] / cell).floor() as i64,
                (gw.translation[2] / cell).floor() as i64,
            );
            cells.entry(key).or_default().push(gw.id);
        }
        (0..n)
            .into_par_iter()
            .map(|i| {
                let g1 = &self.glowworms[i];
                let vr2 = g1.vision_range * g1.vision_range;
                let cx = (g1.translation[0] / cell).floor() as i64;
                let cy = (g1.translation[1] / cell).floor() as i64;
                let cz = (g1.translation[2] / cell).floor() as i64;
                let mut nb: Vec<u32> = Vec::new();
                for dz in -1..=1i64 {
                    for dy in -1..=1i64 {
                        for dx in -1..=1i64 {
                            if let Some(ids) = cells.get(&(cx + dx, cy + dy, cz + dz)) {
                                for &id in ids {
                                    if id != g1.id && g1.luciferin < self.glowworms[id as usize].luciferin {
                                        if distance_sq(g1, &self.glowworms[id as usize]) < vr2 {
                                            nb.push(id);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                nb.sort_unstable();
                nb
            })
            .collect()
    }

    pub fn save(&mut self, step: u32, output_directory: &str) -> Result<(), Error> {
        let path = format!("{}/gso_{}.out", output_directory, step);
        let mut output = BufWriter::new(File::create(path)?);
        writeln!(
            output,
            "#Coordinates  RecID  LigID  Luciferin  Neighbor's number  Vision Range  Scoring"
        )?;
        for glowworm in self.glowworms.iter() {
            write!(
                output,
                "({:.7}, {:.7}, {:.7}, {:.7}, {:.7}, {:.7}, {:.7}",
                glowworm.translation[0],
                glowworm.translation[1],
                glowworm.translation[2],
                glowworm.rotation.w,
                glowworm.rotation.x,
                glowworm.rotation.y,
                glowworm.rotation.z
            )?;
            if glowworm.use_anm && !glowworm.rec_nmodes.is_empty() {
                for i in 0..glowworm.rec_nmodes.len() {
                    write!(output, ", {:.7}", glowworm.rec_nmodes[i])?;
                }
            }
            if glowworm.use_anm && !glowworm.lig_nmodes.is_empty() {
                for i in 0..glowworm.lig_nmodes.len() {
                    write!(output, ", {:.7}", glowworm.lig_nmodes[i])?;
                }
            }
            writeln!(
                output,
                ")    0    0   {:.8}  {:?} {:.3} {:.8}",
                glowworm.luciferin,
                glowworm.neighbors.len(),
                glowworm.vision_range,
                glowworm.scoring
            )?;
        }
        Ok(())
    }
}
