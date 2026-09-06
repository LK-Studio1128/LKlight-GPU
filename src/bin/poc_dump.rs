//! poc_dump — Metal POC dataset exporter.
//!
//! Builds the 1AZP DNA scoring model exactly as the engine run does (same
//! PDBox, same parameterisation), then writes a deterministic N-pose batch
//! (field / receptor-cell-list / ligand / transformed poses / CPU-grid refs)
//! plus single-thread CPU-grid & CPU-exact baselines for the same poses.
//!
//! Usage:
//!   poc_dump <receptor.pdb> <ligand.pdb> <out_dir> [n_pose] [seed]
//! Defaults: n_pose = 1000, seed = 20260906.

use std::env;
use std::process::ExitCode;

use lklight::dna::DNA;
use lklight::scoring::Score;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: poc_dump <receptor.pdb> <ligand.pdb> <out_dir> [n_pose] [seed]");
        return ExitCode::from(2);
    }
    let rec_path = &args[1];
    let lig_path = &args[2];
    let out_dir = std::path::PathBuf::from(&args[3]);
    let n_pose: usize = args.get(4).map(|s| s.parse().unwrap_or(1000)).unwrap_or(1000);
    let seed: u64 = args.get(5).map(|s| s.parse().unwrap_or(2026_0906)).unwrap_or(2026_0906);

    let (receptor, _e1) =
        pdbtbx::open(rec_path, pdbtbx::StrictnessLevel::Strict).expect("open receptor pdb");
    let (ligand, _e2) =
        pdbtbx::open(lig_path, pdbtbx::StrictnessLevel::Strict).expect("open ligand pdb");

    let scoring = DNA::new(
        receptor, Vec::new(), Vec::new(), Vec::new(), 0,
        ligand, Vec::new(), Vec::new(), Vec::new(), 0,
        false,
    );

    match scoring.dump_metal_dataset(n_pose, seed, &out_dir) {
        Ok((grid_ms, exact_ms)) => {
            println!(
                "POC_DUMP_OK n_pose={} seed={} out={}",
                n_pose, seed, out_dir.display()
            );
            println!(
                "CPU_BASELINE_MS grid={:.3} exact={:.3} (single-thread, {} poses)",
                grid_ms, exact_ms, n_pose
            );
            println!(
                "PER_POSE_MS grid={:.6} exact={:.6}",
                grid_ms / n_pose as f64,
                exact_ms / n_pose as f64
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("dump failed: {e}");
            ExitCode::FAILURE
        }
    }
}
