# LKlight GPU 加速设计方案（参考 UniDock-Pro）

**日期**：2026-09-02
**目标**：给 LKlight（Rust + rayon CPU 并行）加入 GPU 加速，参考 UniDock-Pro（CUDA 加速 AutoDock Vina fork）的设计，先 CPU 全平台可用、再上 GPU，Linux 服务器真机实测。

---

## 1. 现状性能剖析（已有实测数据）

| 指标 | 数据 |
|---|---|
| dna 评分单 swarm 200 glowworms×100 步 | ~103s（8 线程，Z 窗口加速后） |
| fastdfire 同规模 | 快 ~1000×（dna 是 AMBER94 全原子静电+VDW） |
| 打分方式 | 每 glowworm 每步调 `energy()`：O(N_rec × N_lig) 原子对（dna 全原子） |
| 并行度 | `swarm.rs` rayon `par_iter_mut` 对 glowworm 打分并行（CPU 核数） |
| 用户体系 | 200 swarms × 200 glowworms × 100 steps，dna 评分 → 曾 6000s 超时 |

**瓶颈本质**：单次打分重（原子对全遍历）+ 打分次数多（swarms×glowworms×steps 每次都全量重算）。

## 2. UniDock-Pro 的 GPU 设计要点（借鉴来源）

对 `src/cuda/` + `src/lib/` 勘察：

1. **受体亲和力网格预计算**（`precalculate.cu`/`cache.cpp`）：受体按原子类型预计算成三维格点亲和力 map；配体打分变**查表插值 O(N_lig)**，不再逐对遍历。这是 Vina 系（含 UniDock）提速的根基，**CPU 也受益**。
2. **GPU 大规模并行搜索**（`monte_carlo.cu`）：每个配体 pose 绑定一个 GPU warp（`cg::tiled_partition<TileSize>`），成百上千 pose 同时做 MC/局部搜索——搜索个体级并行。
3. **SZV 稀疏体素网格**（`szv_grid.h`）：仅存非空格点，减少内存与访存。
4. 模板编译期展开常量（`kernel.h` BaseConfig），kernel 内零分支热路径。

## 3. LKlight GPU 加速方案（三层递进）

### L1 — 受体亲和力网格预计算打分（CPU 实现，所有平台可用）★ 本轮主交付
- 为 `dna`/`ddna`（AMBER94 全原子）评分建立受体**原子类型格点**：
  - 格距 0.5 Å，范围 = 受体包围盒外扩 10 Å（VDW 截断）
  - 每格点存：对 20 种 AMBER 原子类型的静电势（Σ q_j/r，含 30Å 截断 clamp）与 VDW 参数（Σ sqrt(ε)，Σ r*）
  - 只对**重原子**建格（氢贡献并入其重原子，跳氢避免 clash 罚误伤——与已修的 clash 罚语义一致）
- 打分 = 配体原子查表（三线性插值）× 原子类型 + 配体内部对 + **重原子 clash 罚**（保留已实现语义）
- 预期：dna 打分从 O(N_rec×N_lig) → O(N_lig)，**再提速 10~50×**（Z 窗口基础上）
- 现有 12 个评分函数中先做 dna/ddna（用户痛点），fastdfire 已够快不动

### L2 — 打分查表上 GPU（wgpu 计算管线，跨 Metal/DX12/Vulkan）
- 每 glowworm 打分 = 一个 GPU 计算任务（配体原子并行查表归约）
- 每步 GSO：swarm 内全部 glowworm 打分打包成一批 kernel 提交（替换 rayon 段）
- 后端：本地 Apple M4（Metal 4）开发实测；Linux NVIDIA（Vulkan/CUDA）部署
- 数值一致性：GPU 与 L1 CPU 逐位比对（阈值 1e-4），1AZP 回归 120/120 无深 clash 保持

### L3 — 搜索级并行（远期，可选）
- GSO 每个 swarm 的 glowworm 打分上 GPU 多 warp（类似 UniDock pose-warp 映射）
- 需要 GPU 常驻上下文 + 步间同步，复杂度高，建议 L1/L2 验证后再做

## 4. 数值一致性与回归保障
- 回归集：1AZP native（120 解 0 深 clash 需保持）、真实蛋白-RNA 体系、score 命令逐位对比
- 容差：网格化打分对 CPU 精确打分容差 < 0.5 kcal（0.5Å 格距插值误差）；排序稳定性检查（top10 一致）
- clash 语义不变（重原子对 <0.75×vdW 和线性罚 6/Å）

## 5. 部署与测试计划
| 环境 | 用途 |
|---|---|
| 本地 Mac M4 (Metal 4) | L1 开发 + L2 GPU 路径真机验证 |
| Linux 服务器 117.50.160.76 | L1 CPU 全量实测（12 核对比）；L2 需服务器有 GPU（待确认） |
| Windows | 随发布编译（DX12/Vulkan） |

## 6. 待确认事项
1. 服务器 117.50.160.76 是否含 NVIDIA GPU？（当前 SSH 不通，需开机确认）——决定 L2 用 wgpu-Vulkan 还是 CUDA
2. GPU 后端选型：wgpu（跨平台）优先推荐；CUDA 仅当服务器/目标用户 N 卡确定
3. 本轮交付范围：L1（CPU 网格加速，立即可用可测）+ L2 框架/接口，还是 L1+L2 完整 GPU 实现
