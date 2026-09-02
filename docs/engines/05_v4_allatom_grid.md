# 05 v4 全原子加速（当前推荐版，LKlight-CUDA）

## 概述

v4 收尾"适配所有评分函数与所有分支"：把 v1–v3 只为 dna 做的网格加速**推广到其余
全原子评分函数**（vdw/pydock/cpydock），让 **restraints/膜 分支也走网格路径**
（不再回退 30Å 全对），并交付 **swarm 级并行工具**。随后完成一次全面 bug 审计
（commit `d4667c6`）。这是当前唯一推荐发布的引擎。

- 代码：`src/nearcell.rs`（通用受体 cell list）、`vdw.rs/pydock.rs/cpydock.rs`
  （energy_grid 路径）、`dna.rs`（约束/膜 cell 界面收集 + batch 条件修复）、
  `tools/run_parallel.py`。

## 算法

### 通用受体 cell list（nearcell.rs）
与 dna 的 cell 同一思想抽象为通用结构：按坐标建 10Å 均匀网格 + 前缀和索引；
`for_each_near(x,y,z)` 遍历 27 邻域候选。**生产调用均为"受体建 cell、配体做 query"**
（两集合分离，无自配对问题）；任何 d≤10Å 的贡献对必在窗口内 → 与全对扫描同对集。

### vdw（48×，逐位一致）
纯 LJ 的全部贡献都在 ≤10Å（无远距项）→ cell 窗口完整覆盖 → 与精确路径
**逐位一致**（实测多姿态差 0.000）。界面标记（restraints 用）同轮收集。

### pydock（15×）与 cpydock（11.7×）
与 dna 相同的能量核（clamp 静电 + LJ），故复用同一拆分：近距 ≤10Å cell 逐对精确
（含 clamp）+ 远距 10–30Å 受体场查表（10Å 外 clamp 永不触发 → 线性分解精确）。
cpydock 多一个**接触-SASA 脱溶项**：只需每原子 6.4Å（溶化窗）内最近重原子——
远小于 cell 窗口，min 距离在 cell 扫描内**精确**取得，脱溶项无近似；
无 ≤6.4Å 邻居的原子贡献 0，与全对扫描等价。

### dna restraints / 膜 上网格
界面残基标记只需知道"有哪些 ≤ 界面距离(≈3.9Å) 的跨分子原子对"——cell 近距扫描
本就枚举全部 ≤10Å 对 → 顺带写 iface 标记即可，**不必回退 30Å 全对**。
GPU kernel 无法返回标记 → 约束/膜 run 自动走 CPU cell 路径（`supports_batch` 已
修正为在这些场景禁用批量，防止静默忽略约束）。

### ANM（结论性取舍，非遗漏）
受体每 pose 独立变形 → 静态场/cell 每 pose 失效需全量重建（每步 20 pose×重建 ≈
秒级）→ 场加速不适用；ANM 场景多为小体系，保留精确路径并记录在案。

### swarm 并行（tools/run_parallel.py）
swarm 彼此独立 → P 进程并行（multiprocessing 池），实测 P4 = 8.5s vs 串行 35s。

## 使用方法

```bash
cargo build --release                   # CPU：dna/vdw/pydock/cpydock 全部网格加速
cargo build --release --features cuda   # GPU 批量（dna；约束/膜自动 CPU 网格）
LKlight run setup.json initial_positions_0.dat 100 <dna|vdw|pydock|cpydock>

# 多 swarm 并行（替代逐个 run）
python3 tools/run_parallel.py LKlight 100 dna 6 4

# 约束/膜：与之前一样传 --restraints，引擎自动走网格路径
LKlight setup rec.pdb lig.pdb dna -s 6 -g 20 --restraints restraints.txt --noxt --now
```

## 效果（RNA 大体系 20 glow×100 步，服务器实测；修复后 34/34 测试）

| 函数 | v1–v3（仅 dna 加速）| v4 | 提速 | 精度 vs exact |
|---|---|---|---|---|
| vdw | 50.1 s | **1.04 s** | **48×** | 逐位一致 0.000 |
| pydock | 114.1 s | **7.61 s** | **15×** | 0.13–0.27% |
| cpydock | 114.1 s | **9.72 s** | **11.7×** | 绝对差同 pydock（solv 精确）|
| dna | 8.9 s（已加速）| 8.9 s | 19×+（相对 exact）| <0.5% |

restraints/膜 run：收敛与精确路径完全一致（best/med 相同）。

## 影响 / 适用

- **任何评分函数的日常任务都可以直接用 v4**：全原子类全部网格加速，查表类本就快；
- **12 函数大体系矩阵揭示的结论**（`PERF_COMPARE §7`）：PyDock 家族曾是隐藏的
  "比 dna 还慢 6–13 倍"的瓶颈——v4 一次抹平；
- 约束/膜用户从"必然 30Å 全对"变为"自动网格"（约 10×+），语义不变；
- swarm 并行把"35 swarm × 200 glow"这类任务从串行小时级压到分钟级；
- **边界（非 bug）**：ANM 用精确路径；超长链配体（>500Å）刚性对接几何病态，应
  裁剪结合域片段后对接。

## 与 LKDock 软件的关系（待办）

`release/` 四平台 v4 产物已就绪（mac/win/linux static-pie/linux cuda），但软件集成
目录 `byi/LK_Studio/LKlight/` 目前仍是 01 版（exact clash 修复）——**把 v4 替换进
软件发布目录是让用户吃到全部加速的最后一步**。
