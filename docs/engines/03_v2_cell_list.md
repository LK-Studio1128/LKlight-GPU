# 03 v2 cell-list（LKlight-CUDA，累计 19.4×）

## 概述

v2 把 CPU 近距也改成**配体原子查受体 cell list**（10Å 均匀网格、27 邻域精确覆盖），
去掉每打分一次的 Z 排序，候选对数再降约 5.7×。**同 seed 输出逐位一致**
（同一批原子对、同一公式，仅遍历方式不同）。打分至此累计提速 **19.4×**
（Mac 55.3s → 2.85s）。同一阶段还完成了 CUDA 远距场 kernel 的接入验证。

- 代码：`src/gpu_score.rs`（CudaReceptor：受体 cell list，CPU/GPU 共用）、`src/cuda/far_field.cu`。

## 算法

1. **受体 cell list（建一次、全程复用）**：受体刚性不动（非 ANM），按 10Å 边长把
   受体原子分箱（`cell_start/cell_atoms` 前缀和布局）。
2. **每打分近距**：对每个**配体原子**求所在 cell，遍历自身+26 邻 cell 内的受体原子
   做距离过滤——任何 d≤10Å 的贡献对必然落在 27 cell 窗口内，**候选不漏不重**，
   且每个配体原子只测约 150 个候选（此前受体扫配体约 1150 个），免排序。
3. 远距静电场仍走 v1 的受体场查表；打分能量 = 近距(clamp 静电+LJ+clash 罚) + 远距场。

**逐位一致保证**：vdw/LJ 项与 clash 罚都是短程（≤10Å），被 cell 窗口完整覆盖；
静电近距精确、远距查表在 10Å 外无 clamp → 与精确路径只在 f64 求和顺序上不同。

## 使用方法

命令不变。GPU 桥首次接入：`cargo build --release --features cuda`（缺 CUDA 自动回退）。

```bash
LKlight run setup.json initial_positions_0.dat 100 dna    # CPU cell 版，快 19.4×
# score 抽查：应与 v1/原版数值一致到 f64 舍入
LKlight score rec.pdb lig.pdb dna --tx 1 --ty 2 --tz 3
```

## 效果

| 指标 | 数值 |
|---|---|
| 单 swarm（RNA 大体系，Mac）| **2.85 s**（累计 19.4×；服务器 16.1 → 8.06 s）|
| 数值 | 与 v1 grid **逐位相同**（score -15179.263684 完全一致）|
| CUDA far-field | 服务器 RTX 3080 Ti 真机 ACTIVE；score 与 CPU grid 差 1e-4（f32）|

## 影响

- 打分从"能跑"到"快"：CPU 路径 19.4× 全平台可用；
- 暴露新瓶颈：打分已不再主导后，GPU 版与 CPU 版在服务器同为 ~8s——**单 pose 粒度
  GPU 调用开销与 GSO 框架段成为下一瓶颈**（v3 处理）；
- 该阶段实测还揭示：用户 1324Å 长链 RNA vs 120Å 蛋白属刚性对接病态案例（几何上无
  干净解），exact 同样如此——不是引擎缺陷，需裁剪配体（工具 `tools/clash_analyze.py`
  可体检任意 pose）。
