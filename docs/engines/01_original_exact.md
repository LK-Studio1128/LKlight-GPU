# 01 原版 exact（LKlight 基线）

## 概述

Python LightDock 的 Rust 逐行重写，是 LKlight 的**第 0 版与参考真值**：所有评分函数
按官方算法**全原子逐对**打分，不引入任何网格/GPU 近似。2026-09-02 上午完成一次关键
正确性修复（穿模 clash 惩罚），之后保持冻结，作为后续所有加速版的数值基准。

- 代码线：`byi/LKlight`（独立 git，最新 `82f6256`）
- 产物：mac/win/linux 三平台二进制（GitHub dist + 软件集成目录 `LK_Studio/LKlight/`）
- 定位：**参考真值**。任何加速版与它的偏差都能量化（见各版本 MD 与 `PERF_COMPARE`）。

## 算法

1. **搜索**：GSO（群智能优化）。每 swarm 若干 glowworm，各自持有一个对接位姿
   （3 平移 + 4 旋转四元数 + 可选 ANM 模态坐标），每步按荧光素(luciferin)亮度找邻居、
   依概率朝更亮邻居移动、逐步缩视野，100 步左右收敛到结合位。
2. **打分（dna，AMBER94 全原子）**：
   - 静电：`Σ q_i q_j / r²`，每对 clamp 在 ±0.012（×332/4 归一到 ±1 kcal/mol），
     截断 30Å；
   - 范德华：Lennard-Jones `ε[(r_vdW/r)¹² − 2(r_vdW/r)⁶]`，深穿透排斥每对封顶 1.0；
   - **clash 修复**（2026-09-02，本线最重要改动）：重原子线性深嵌惩罚——双方重原子对
     距离 < 0.75×(vdW 半径和) 时每 Å 罚 6.0（氢除外，不误伤氢键）。
     修复前实测：120/120 解深 clash 36~354 处、最大穿透 2.4–3.4Å，且评分随穿模加深
     而"更好"（引擎系统性奖励把核酸插进蛋白）；修复后 1AZP 120/120 解 **0 深 clash**。
3. **12 族评分**：dfire/dfire2/dna/ddna/mj3h/pydock/cpydock/sd/pisa/sipper/tobi/vdw，
   统一 `Score` trait；ANM 柔性、restraints 约束、膜环境为全原子逐对打分的附加项。

## 使用方法

```bash
# 构建（需 Rust stable）
cargo build --release          # → target/release/lklight

# 流程
LKlight setup rec.pdb lig.pdb dna -s 6 -g 20 --seed 42 --noxt --now
LKlight run setup.json initial_positions_0.dat 100 dna     # 每 swarm 一个进程
LKlight rank 6 100
LKlight generate lightdock_rec.pdb lightdock_lig.pdb swarm_0/gso_100.out 20
LKlight score rec.pdb lig.pdb dna --tx 1 --ty 2 --tz 3 --qw 1 --qx 0 --qy 0 --qz 0
```

`--noxt/--now` 跳过 reduce 加氢与去水（输入需自带 H）；加 `--restraints f` 启用约束。

## 效果

| 指标 | 数值 |
|---|---|
| 单 swarm 20 glow×100 步（RNA 大体系）| ~55 s（Mac）/ 71 s（服务器 12 核）|
| 打分正确性 | 官方 PyDockDNA/LightDock 语义，逐位复刻 |
| clash 修复后 pose 质量 | 正常体系 0 深 clash（1AZP 120/120）|
| 精度角色 | **一切加速版的真值基准** |

## 影响 / 适用

- **适用**：中小体系、精度敏感研究、作为加速版的对照；
- **不适用**：大体系/大规模筛选（1.2 万原子 RNA 单 swarm 一分钟级，用户真实任务
  35 swarm×200 glow 需 4 小时+，此前是 LKDock 用户"超时"的直接原因）；
- **历史贡献**：exact 线承载了穿模根因定位与 clash 修复，该修复语义被所有加速版继承
  （加速只改遍历方式，不改打分公式）。
