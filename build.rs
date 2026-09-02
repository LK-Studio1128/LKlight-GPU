// build.rs — compiles src/cuda/*.cu with nvcc when the `cuda` feature is
// enabled, archives them, and links the CUDA runtime found next to nvcc.
//
//   Linux:   nvcc -c -> .o, ar crs libfarfield.a, static libcudart_static.a
//            (binary then only needs the NVIDIA *driver*; portable across
//             GPU machines). Falls back to dynamic libcudart.so.
//   Windows: nvcc -c -> .obj, lib /OUT:libfarfield.lib (needs the MSVC/VC
//            environment, i.e. run inside a "vcvars64" prompt), static
//            cudart_static.lib from %CUDA_PATH%\lib\x64.
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

    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    std::fs::create_dir_all(&out_dir).unwrap();
    let is_win = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");
    let obj_ext = if is_win { "obj" } else { "o" };
    let lib = if is_win {
        out_dir.join("farfield.lib")
    } else {
        out_dir.join("libfarfield.a")
    };

    // 1) nvcc: compile each kernel.
    let mut objs: Vec<PathBuf> = Vec::new();
    for src in ["src/cuda/far_field.cu", "src/cuda/full_score.cu"] {
        let stem = src.rsplit('/').next().unwrap().replace(".cu", "");
        let obj = out_dir.join(format!("{stem}.{obj_ext}"));
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

    // 2) Archive the objects.
    if is_win {
        // Microsoft librarian (requires the VC environment / vcvars64).
        let lib_exe = std::env::var("LIBEXE").unwrap_or_else(|_| "lib".to_string());
        let mut cmd = Command::new(&lib_exe);
        cmd.arg(format!("/OUT:{}", lib.display()));
        for o in &objs {
            cmd.arg(o);
        }
        let status = cmd.status().expect("failed to spawn lib.exe — run inside a VS developer prompt");
        if !status.success() {
            panic!("lib.exe failed to create {lib:?}");
        }
        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=farfield");
    } else {
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
        // nvcc emits C++-flavoured objects (__cxa_guard / __gxx_personality).
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    // 3) CUDA runtime — prefer static cudart so the target machine only needs
    //    the NVIDIA driver.
    if is_win {
        if let Some(dir) = find_libdir_win("cudart_static.lib") {
            println!("cargo:rustc-link-search=native={}", dir.display());
            println!("cargo:rustc-link-lib=static=cudart_static");
        } else if let Some(dir) = find_libdir_win("cudart.lib") {
            println!("cargo:rustc-link-search=native={}", dir.display());
            println!("cargo:rustc-link-lib=dylib=cudart");
        } else {
            panic!("CUDA feature enabled but no cudart_static.lib/cudart.lib under %CUDA_PATH%\\lib\\x64");
        }
    } else {
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

/// Windows: look under %CUDA_PATH%\lib\x64 (plus %CUDA_PATH% itself).
fn find_libdir_win(file: &str) -> Option<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(cp) = std::env::var("CUDA_PATH") {
        roots.push(PathBuf::from(&cp));
    }
    if let Ok(programs) = std::env::var("ProgramFiles") {
        roots.push(PathBuf::from(&programs).join("NVIDIA GPU Computing Toolkit\\CUDA"));
    }
    for root in roots {
        for cand in ["lib\\x64", "lib", ""] {
            let d = root.join(cand);
            if d.join(file).exists() {
                return Some(d);
            }
        }
    }
    None
}
