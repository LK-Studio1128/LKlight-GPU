# 04 v3 GPU 批量 + GSO 框架（LKlight-CUDA）

## 概述

v3 做两件事，目标是让 GPU 真正发力并把框架开销压平：

1. **完整打分搬上 GPU 并批量**：`full_score.cu` 把近距（cell-list 扫描、clamp 静电、
   LJ、clash 罚）+ 远距场合并进**一个 kernel**；`Score::batch_energy` 把**一整步的
   全部待打分构象合并成一次 kernel launch**（gridDim.y = 构象数，double 归约），
   摊销每次 ~18ms 的 launch/同步开销。
2. **GSO 框架并行与上传瘦身**：邻居搜索 O(n²) → 均匀空间哈希（≥64 glowworm 时
   27-cell 精确窗口，id 排序保序）；位姿快照/邻居安装/移动概率 rayon 并行；批量
   kernel 的**坐标变换下放 device**（每步上传 N×7 个 pose 参数，替代 N×原子数×3
   全坐标——n=2000 时 303MB → 112KB）。

全部改动**同 seed 输出逐位一致**（gso_100.out diff=0）。

## 算法要点

- **单 pose 太慢 → 批量才是 GPU 的主场**：实测每打分一次 GPU 同步约 18ms，
  12 线程排队抢 GPU 会吃掉并行收益（单 pose GPU 与 CPU grid 持平甚至略慢）；
  改为每步一次 kernel 后，2000–5000 构象的扩展近乎水平。
- **device 端 f64 刚体变换**：kernel 内对基准配体坐标做 `R(q)·v + t`（与 CPU 端
  f64 变换逐位一致后转 f32），省掉每步 30MB+ 的 host 变换与上传。
- **框架**：邻居关系是 GSO 的热点（每步全对距离）；≥64 glowworm 用 cell=最大视野
  半径的均匀哈希 + 候选按 id 排序，保证概率/随机移动与参考完全一致。

## 使用方法

```bash
cargo build --release                  # CPU 网格（全平台，含框架优化）
cargo build --release --features cuda  # GPU 批量（Linux + N 卡驱动即可，静态链 cudart）
LKlight run setup.json initial_positions_0.dat 100 dna
# 日志出现 [gpu_score] CUDA BATCH scoring ACTIVE → GPU 生效；无则自动 CPU
```

## 效果（服务器 RTX 3080 Ti / 12 核）

| 场景 | v2 → v3 |
|---|---|
| n=200×100 步 GPU | 9.34 → **7.85 s** |
| n=2000×20 步 GPU | 11.95 → **7.57 s（1.58×）**；CPU 23.95 s → GPU 3.2× |
| n=5000×50 步 GPU | ~11 s（打分近乎免费，扩展近水平）|
| Mac 端到端 6×20×100 步 | **22.0 s**（原 exact 单 swarm 55 s）|

收敛与 CPU 网格统计一致（best/median 相同或 ≤1%）。

## 影响

- **GPU 收益兑现条件明确**：单 swarm 构象数×步数越大，GPU 越值（交叉点约 N≈300、
  步长 ≥100 后 gpu/grid 1.6–2.4×；小任务 GPU≈CPU，用 CPU 版即可）；
- **swarm 级并行是下一个量级**：swarm 彼此独立，实测 P4 并行 = 8.5s vs 串行 35s
  （≈4× 线性）——由 v4 的 `tools/run_parallel.py` 与集成层兑现；
- 四平台产物进入"换机即跑"状态：mac 仅系统库 / win 仅 UCRT / linux static-pie /
  **linux-cuda 零 CUDA 依赖（驱动即可，无 GPU 自动回退）**。
