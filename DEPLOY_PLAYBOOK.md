# LKlight-CUDA 跨平台部署与换机手册

**版本**：2026-09-02（LKlight-CUDA 独立副本，CPU grid 7× 加速 + CUDA far-field）

---

## 一、构建矩阵（三个平台 × 两种加速档）

| 平台 | CPU 版（默认，全机器可用） | CUDA 版（--features cuda，GPU 机） |
|---|---|---|
| **macOS** (arm64) | `cargo build --release` → 单文件，只依赖系统库 | M 系列无 NVIDIA，不适用（用 CPU 版） |
| **Windows** x64 | `cargo build --release --target x86_64-pc-windows-gnu` → 单 exe，仅依赖 Win10+ 自带 UCRT | 需 NVIDIA 驱动 + CUDA Toolkit ≥13 交叉/本机编译 |
| **Linux** x86-64 | `cargo build --release --target x86_64-unknown-linux-musl` → **static-pie 全静态**，任何发行版可跑 | `--features cuda`（gnu target）→ **静态 cudart**，只需 NVIDIA 驱动 |

## 二、换机可用性（实测验证）

| 产物 | 依赖 | 换机表现 |
|---|---|---|
| macOS LKlight | `/usr/lib/libSystem.B.dylib`（系统自带） | 拷走即跑 ✅ |
| Windows LKlight.exe | `KERNEL32.dll` + `api-ms-win-crt-*`（Win10/11 自带）| 拷走即跑 ✅（无额外 DLL） |
| Linux CPU static-pie | **零动态依赖**（musl 静态） | 拷到任何 x86-64 Linux 即跑 ✅ |
| Linux CUDA 版 | 6 个 glibc 系统库（无 libcudart/无 libcuda 编译期依赖） | 需目标机有 **NVIDIA 驱动**（运行时自动加载 libcuda）；无 GPU/无驱动→far-field 自动回退 CPU，任务不中断 ✅ |

**CUDA 版"换机即降级"**：`gpu_field.rs` 任何 CUDA 失败（无设备/驱动缺失/kernel 错误）都返回 `None` → `dna.rs` 自动走 CPU 三线性查表，引擎照常完成对接（日志不打印 `CUDA ACTIVE` 即 CPU 模式）。

## 三、Linux 服务器构建脚本（建议放进 CI/发布）

```bash
# CPU 全静态版（分发首选）
export PATH=/usr/local/cuda-13.2/bin:$HOME/.cargo/bin:$PATH
cargo build --release --target x86_64-unknown-linux-musl
cp target/x86_64-unknown-linux-musl/release/LKlight LKlight-linux86-static

# CUDA 加速版（GPU 服务器用；build.rs 自动静态链 cudart_static）
cargo build --release --features cuda
cp target/release/LKlight LKlight-linux-cuda
```

## 四、运行时自检

```bash
LKlight --version          # 版本
LKlight score rec.pdb lig.pdb dna --tx 0 --ty 0 --tz 0
#   GPU 生效时首行（或首个打分）打印：
#   [gpu_field] CUDA far-field ACTIVE (grid ...)
#   未打印 = CPU grid 模式（无 GPU 或 CUDA 不可用，功能不受影响）
```

## 五、性能基线（RTX 3080 Ti 服务器 / 蛋白-RNA 真实体系）

| 档位 | 单 swarm（除注明） |
|---|---|
| 原始逐对打分 | 55.3 s（Mac 8 核参考，20glow×100 步） |
| CPU grid（L1） | 7.9 s（Mac）/ 16.1 s（服务器 12 核）→ **7×** |
| **CPU 网格 cell 版（v2，所有机器）** | **2.85 s（Mac）→ 累计 19.4×** |
| CUDA batch（近距+远距一次 kernel） | 8.08 s（服务器）→ 与 CPU cell 持平（打分已非瓶颈） |
| **v3 GSO 框架 + GPU 端变换** | n=2000×20 步 GPU **7.57 s**（旧 11.95 s，1.58×）；Mac 端到端 6×20×100 步 **22.0 s** |

## 六、评分函数覆盖说明

- 网格/GPU 加速路径已接入 **Score trait**（`energy_grid`），对所有评分函数开放：
  - `dna`/`ddna`（AMBER94 全原子）：dna 走网格 + CUDA far-field；ddna 自带 cell-list（非瓶颈）
  - `fastdfire/dfire/dfire2/vdw/sd/pisa/mj3h/sipper/tobi/cpydock/pydock`：走默认等价路径（`energy_grid = energy`），零回归；fastdfire 实测 0.07s 无需加速
- 所有评分在 setup/run/score/rank/generate 全命令统一经 `Score::energy` → 网格/CUDA 自动生效，无分支遗漏
