# LKlight-GPU

**LKlight GPU（CUDA）版（v4 最终版）** —— 全功能分子对接引擎，Linux + NVIDIA 驱动
即用；无 GPU 时自动回退 CPU 网格路径，功能与 `../LKlight-grid` 完全等价。

LKlight 是 Python LightDock（GSO 群智能对接）的 Rust 高性能实现；本目录是 **CUDA
批量加速最终版**的独立发布项目：打分 kernel 把 ≤10Å 近距（cell-list 扫描、clamp
静电、LJ、clash 罚）与 10–30Å 远距场合并，**整步全部构象一次 kernel launch**
（gridDim.y = 构象数），并把刚体坐标变换放到 device（每步只上传 N×7 个 pose 参数）。
大规模（≥300 构象/步、长步数）下相对 CPU 网格再快 1.6–2.4×，单 swarm 可扛 5000
glowworm 近乎线性扩展。

## 一、功能清单（与 CPU 网格版完全一致，无遗漏）

- **评分函数 12 族**：dfire / dfire2 / dna / ddna / mj3h / pydock / cpydock / sd /
  pisa / sipper / tobi / vdw（全命令统一入口）
- **GPU 覆盖**：`dna` 无约束/无膜/无 ANM 时走 **GPU 批量 kernel**（近距+远距一次
  launch，device 端 f64 刚体变换）；`vdw/pydock/cpydock` 走 CPU 网格（其静电/LJ
  已网格化，12–48×）；
- **自动回退（同一二进制，无需配置）**：无 NVIDIA 驱动 → CPU 网格；ANM /
  restraints / 膜 → CPU 网格路径（GPU kernel 不返回界面标记，回退保证约束语义正确，
  已 bug 审计验证）；
- **高级功能**：ANM、restraints、膜、GSO 邻居分箱+并行、`tools/run_parallel.py`
  swarm 并行、clash/扫描体检工具——全部保留。

## 二、换机即用（仅 Linux；本版本只发布 Linux CUDA）

| 文件 | 平台 | 运行时依赖 | 说明 |
|---|---|---|---|
| `release_bin/LKlight-linux-cuda` | Linux x86-64 | **仅 NVIDIA 驱动**（CUDA 运行库已静态链入，无需装 Toolkit）| 有 GPU → 批量加速；无 GPU/驱动 → 自动 CPU |

验证：
```bash
./release_bin/LKlight-linux-cuda run setup.json initial_positions_0.dat 100 dna
# 日志出现 [gpu_score] CUDA BATCH scoring ACTIVE (N poses × atoms) → GPU 生效
# 没有该行 → CPU 网格模式（结果等价，仅速度不同）
```

> Windows/macOS 目前使用 CPU 网格版（见 `../LKlight-grid`）；如需 Windows/mac 的
> CUDA 构建需另行交叉编译（本目录源码支持 `--features cuda` 任意平台，前提是本机
> 有对应平台 nvcc）。

## 三、快速上手

```bash
BIN=release_bin/LKlight-linux-cuda

$BIN setup rec.pdb lig.pdb dna -s 6 -g 200 --seed 42 --noxt --now   # 大规模用 -g 200+
python3 tools/run_parallel.py $BIN 100 dna 6 6                       # 6 swarm 并行
$BIN rank 6 100
for i in 0 1 2 3 4 5; do $BIN generate lightdock_rec.pdb lightdock_lig.pdb swarm_$i/gso_100.out 200; done
```

**GPU 收益条件**：单 swarm 构象数 × 步数越大越值（约 N≥300 且 steps≥100 后明显）；
小任务（几十构象×短步）GPU 与 CPU 持平——此时用 CPU 网格版即可。

## 四、从源码构建（需 NVIDIA 工具链）

```bash
# 前置：Rust stable + nvcc（CUDA ≥ 13；构建机）
# 服务器参考：Ubuntu 22.04 + cuda-nvcc-13-2 + cuda-cudart-dev-13-2

cargo build --release                       # CPU 网格版（任何机器，功能一致）
cargo build --release --features cuda       # CUDA 版（build.rs 自动 nvcc 编译
                                            #  src/cuda/*.cu，静态链 libcudart_static.a
                                            #  → 运行机只需驱动）
cargo test --release                        # 34 项测试
```

## 五、性能参考（RTX 3080 Ti / 12 核，RNA 大体系）

| 场景 | CPU 网格 | GPU 批量 |
|---|---|---|
| 单 swarm 2000 glow × 20 步 | 23.95 s | **7.57 s（3.2×）** |
| 单 swarm 200 glow × 100 步 | 16.1 s | **7.85 s** |
| 1000/2000/5000 glow × 100 步 | 32.7 s / — / — | 9.1 / 10.5 / ~21 s |

收敛与 CPU 网格统计一致（GPU 打分差 <1e-5，f32 级）。完整基准：
`GPU_BENCH_20260902.md`、`PERF_COMPARE_20260902.md`；引擎版本：`docs/engines/`。

## 六、发布信息

- 版本：**v4**（2026-09-03 bug 审计修复后快照；原仓库 git `d4667c6`）
- 二进制：静态链 cudart → `ldd` 零 CUDA 依赖（仅 ~6 个系统库）；测试 34/34。
