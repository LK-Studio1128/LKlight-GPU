# LKlight 引擎版本总览

LKlight 是 Python LightDock 分子对接的 Rust 高性能重写：GSO（群智能优化）搜索 +
12 族评分函数 + 三平台单文件分发。自 2026-09-02 起，从"穿模修复"到"全原子网格/GPU
加速"共形成 **5 个可讲述的引擎版本**，分属两条代码线：

| 版本 | 代码线 | 打分方式 | 相对原版 | 状态 |
|---|---|---|---|---|
| [01 原版 exact](01_original_exact.md) | `byi/LKlight` | 全原子逐对 30Å | 1×（参考真值）| 已发布，软件在用 |
| [02 v1 网格场](02_v1_grid_l1.md) | `byi/LKlight-CUDA` | 近距逐对 + 远距 10–30Å 场查表 | 7× | 历史里程碑 |
| [03 v2 cell-list](03_v2_cell_list.md) | `byi/LKlight-CUDA` | 配体扫受体 cell list | **19.4×**（累计）| 历史里程碑 |
| [04 v3 GPU 批量 + GSO](04_v3_gpu_batch_gso.md) | `byi/LKlight-CUDA` | 整步构象一次 GPU kernel + 框架并行 | 高负载再 +1.6–2.4× | 历史里程碑 |
| [**05 v4 全原子加速（当前）**](05_v4_allatom_grid.md) | `byi/LKlight-CUDA` | **全部全原子评分网格化 + 约束/膜 + swarm 并行** | vdw 48× / pydock 15× / cpydock 12× / dna 19×+ | **当前推荐** |

## 三条代码线 / 三类产物

- **exact 线**（`byi/LKlight`，git `82f6256`）：官方算法逐位复刻（含穿模 clash 修复），
  三平台二进制已发布在 GitHub dist 与软件集成目录 `byi/LK_Studio/LKlight/`。
- **加速线**（`byi/LKlight-CUDA`，git `d4667c6`）：在保持"同 seed 输出逐位一致/收敛一致"
  的前提下叠加四层加速；原版全程未动。
- **release/ 产物（当前 v4）**：

| 文件 | 平台 | 说明 |
|---|---|---|
| `LKlight-mac-arm64` | macOS | CPU 网格，仅系统库 |
| `LKlight-win64.exe` | Windows 10/11 | CPU 网格，仅自带 UCRT |
| `LKlight-linux86-static` | Linux x86-64 | CPU 网格，**static-pie 全静态** |
| `LKlight-linux-cuda` | Linux + NVIDIA 驱动 | **GPU 批量**（无 GPU 自动回退 CPU）|

## 精度共识（贯穿全部版本）

- 网格/GPU 相对 exact：有效位姿能量误差 ≤0.5%（大体系实测平均 0.07%），top-5 解
  ±2Å 100% 重合，排序 Spearman ≥0.9996；
- vdw 无远距项 → 与精确**逐位一致**；pydock/cpydock 仅远距场插值误差（绝对 ≤12 能量单位）；
- GPU 与 CPU 网格差 <1e-5（f32 舍入），收敛统计一致；
- 12 族评分函数中查表/统计势类（dfire/dfire2/mj3h/pisa/sipper/tobi）天然 ≤0.2s，
  无需加速；全原子类（dna/vdw/pydock/cpydock）v4 起全部网格加速。

## 使用入口

```bash
LKlight setup <rec.pdb> <lig.pdb> dna -s 6 -g 20 --seed 42 --noxt --now   # 前处理
LKlight run setup.json initial_positions_0.dat 100 dna                    # 单个 swarm
python3 tools/run_parallel.py <LKlight> 100 dna 6 4                        # 多 swarm 并行
LKlight rank 6 100 && LKlight generate lightdock_rec.pdb lightdock_lig.pdb swarm_0/gso_100.out 20
LKlight score rec.pdb lig.pdb dna --tx 1 --ty 2 --tz 3                    # 单 pose 打分
```

完整性能基准见 `../PERF_COMPARE_20260902.md`，部署矩阵见 `../DEPLOY_PLAYBOOK.md`。
