// build.rs — compiles src/cuda/far_field.cu with nvcc when the `cuda` feature
// is enabled. Produces a static archive consumed by cargo:
//   target/cuda/libfarfield.a   (linked as `static=farfield`)
// and links the CUDA runtime (cudart) found next to nvcc.
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/cuda/far_field.cu");
    println!("cargo:rerun-if-changed=src/cuda/full_score.cu");
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        // Plain CPU build — nothing to do.
        return;
    }

    // Locate nvcc (respect NVCC env override).
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());

    // out dir for the objects + archive
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    std::fs::create_dir_all(&out_dir).unwrap();
    let lib = out_dir.join("libfarfield.a");

    // Compile every kernel under src/cuda/ with nvcc.
    let mut objs: Vec<PathBuf> = Vec::new();
    for src in ["src/cuda/far_field.cu", "src/cuda/full_score.cu"] {
        let stem = src.rsplit('/').next().unwrap().replace(".cu", "");
        let obj = out_dir.join(format!("{stem}.o"));
        let status = Command::new(&nvcc)
            .args(["-O3", "-arch=native", "-c"])
            .arg(src)
            .arg("-o")
            .arg(&obj)
            .status()
            .expect("failed to spawn nvcc — is the CUDA toolkit installed?");
        if !status.success() {
            panic!("nvcc failed to compile {src} (status {status})");
        }
        objs.push(obj);
    }

    // Archive all objects (ar, as used by the toolchain).
    let ar = std::env::var("AR").unwrap_or_else(|_| "ar".to_string());
    let mut cmd = Command::new(&ar);
    cmd.args(["crs"]).arg(&lib);
    for o in &objs {
        cmd.arg(o);
    }
    let status = cmd.status().expect("failed to spawn ar");
    if !status.success() {
        panic!("ar failed to create {lib:?}");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=farfield");
    // nvcc emits C++-flavoured objects (device stubs use __cxa_guard /
    // __gxx_personality) → link the C++ runtime too.
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // Prefer the *static* CUDA runtime (libcudart_static.a): the resulting binary
    // then only needs the NVIDIA *driver* (libcuda.so, installed with any GPU
    // driver) — no CUDA toolkit install required on the target machine, which
    // makes the CUDA build portable across GPU machines. If the static archive
    // is absent we fall back to dynamic cudart.
    let static_cudart = find_libdir("cudart_static.a");
    let libdir = find_libdir("libcudart.so");
    if let Some(dir) = static_cudart {
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-lib=static=cudart_static");
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=pthread");
        println!("cargo:rustc-link-lib=dylib=rt");
    } else if let Some(dir) = libdir {
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-lib=dylib=cudart");
    } else {
        panic!("CUDA feature enabled but neither libcudart_static.a nor libcudart.so found; set CUDA_PATH or install the CUDA toolkit");
    }
}

/// Locate `lib`/`lib64` dir containing `file`, under $CUDA_PATH or /usr/local/cuda*.
fn find_libdir(file: &str) -> Option<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(cp) = std::env::var("CUDA_PATH") {
        roots.push(PathBuf::from(cp));
    }
    roots.push(PathBuf::from("/usr/local/cuda"));
    if let Ok(entries) = std::fs::read_dir("/usr/local") {
        for e in entries.flatten() {
            let n = e.file_name().to_string_lossy().to_string();
            if n.starts_with("cuda-") {
                roots.push(e.path());
            }
        }
    }
    for root in roots {
        for cand in ["lib64", "lib"] {
            let d = root.join(cand);
            if d.join(file).exists() || d.join(format!("lib{}", file)).exists() {
                return Some(d);
            }
        }
    }
    None
}
