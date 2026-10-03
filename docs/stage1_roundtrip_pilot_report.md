# Stage 1 Round-trip Textual Guidance Pilot (RTG) — 报告

**Branch**: `codex/anchor-completion-validation`（base 含 `60a40c3`）
**Protocol**: `docs/stage1_roundtrip_pilot_protocol.md`（冻结于任何新臂得分）
+ GO 附录 `docs/stage1_roundtrip_pilot_go_addendum.md`
**Model**: Qwen3-VL-8B-Instruct-1024，native E0 engine，K=256，pre-LLM，
greedy，max_new_tokens=2048。环境 qwen3vl_clean / 单卡 A40。

---

## 0. 结论（分开回答，按预注册规则）

1. **RTG 完整方法是否超过原 EADP（硬剪枝）？**——**是（全量，显著）**：
   R_MAIN025 − E_GATHER = **+0.978 [+0.363, +1.598]**（三任务等权 macro，
   20000 次 image-cluster bootstrap，seed 20261002）；DocVQA +1.451
   [+0.565, +2.339] 与 OCRBench +1.500 [+0.000, +3.037] 驱动，TextVQA
   −0.018 [−0.639, +0.608]（持平）。
2. **RTG 完整方法是否非劣于 EADP+Completion（E_MAIN025）？**——**不确定，
   未支持非劣**：主要比较 R_MAIN025 − E_MAIN025 = **+0.047 [−0.541, +0.637]**；
   CI 下界 −0.541 未过非劣阈值 −0.5（差 0.04 点）。分任务方向混合
   （TV −0.248 / DV +0.390 / OCR 0.000，均跨零）。**不能宣称 RTG 优于
   或等价于 E_MAIN025。**
3. **RTG 的机制是否得到确认（区别于 FLAT/SHUF）？**——**未确认**：
   DEV 上 GO 门第 3 条满足（R>F +0.216 类似量级、R>S 点估计正），
   但 fresh 独立面板上 **R−S = −0.981 [−2.310, +0.288] 反转**、
   R−F = +0.216 [−1.165, +1.587] 跨零。机制线索未复现，属不确定。
4. **与 MAIN025 组合效果**：Completion 对 RTG 锚的提升
   （R_MAIN025 − R_GATHER = +1.498 [−0.233, +3.247]，DEV）与 EADP 锚上的
   幅度相当，交互项 2×2 = −2.364 [−5.781, +1.048]（跨零，描述性）。
5. **非回归（最终方法 R_MAIN025 相对原 EADP）**：RWQA +2.88 [+1.30, +4.56]✓、
   ChartQA +1.92 [+0.52, +3.29]✓、MMStar +0.47 [−0.80, +1.74]、
   POPE +0.04 [−0.25, +0.33]——**无显著负回归**；MMBench 仅官方 headline
   −0.39（逐题配对 CI 未完成，见 §6 缺项）。

## 1. DEV 六组主表（DEV-300，已观察探索集；20000 bootstrap）

| arm | Stage 1 | Completion | TextVQA | DocVQA | OCRBench | macro |
|---|---|---|---:|---:|---:|---:|
| E_GATHER | EADP | — | 78.70 | 78.77 | 86.00 | 81.157 |
| E_MAIN025 | EADP | MAIN025 | 81.30 | 80.50 | 88.00 | 83.267 |
| R_GATHER | RTG | — | 80.60 | 75.39 | 89.00 | 81.66 |
| R_MAIN025 | RTG | MAIN025 | 80.30 | 79.74 | 89.00 | 83.01 |
| F_MAIN025 | FLAT | MAIN025 | 81.10 | 78.53 | 89.00 | 82.88 |
| S_MAIN025 | SHUF | MAIN025 | 80.30 | 76.46 | 89.00 | 81.92 |

（来源：`outputs/stage1_roundtrip_pilot/analysis.json` per_task_scores；
E 两臂为复用并复现的历史分数。三任务等权 macro 由上表四舍五入值重算
会略有出入，以 analysis.json 的精确值为准。）

预设对比（DEV）：R−E_MAIN025 −0.254 [−2.566, +2.168]（主要）；
R_GATHER−E_GATHER +0.506 [−1.954, +3.112]；R−F +0.138 [−1.403, +1.651]；
R−S +1.093 [−0.405, +2.728]；R−R_GATHER +1.350 [−0.025, +2.859]；
2×2 交互 −2.364 [−5.781, +1.048]。**全部跨零或负，不支持非劣/优势。**

## 2. GO 判定与全量（附录 §2 执行）

GO 门 4/4（macro −0.254≥−0.5；DocVQA −0.761≥−1.0；R>F/S 点估计；
TTFT +1.26%≤5%）→ R_MAIN025 加入核心三项全量队列。GO 只反映机械门，
DEV 事实是"不确定、未支持非劣"。

## 3. 全量主结果（R_MAIN025，官方完整 split）

- 生成/评分：11349 题（TV 5000 / DV 5349 / OCR 1000），全部逐题分
  严格复现官方 headline（diff≤0.05，OCRBench/POPE 用精确复算分支 diff=0）。
- 主要比较（R−E_MAIN025）与补充比较（R−E_GATHER）见 §0.1/0.2 表。
- fresh 全行（正式方法 MAIN025 相对 BASE）：TV +0.164 [−0.488, +0.810]、
  DV +0.871 [−0.104, +1.852]、pooled(2814 行) +0.517 [−0.114, +0.990]
  ——**跨零，独立确认不成立**；OCRBench fresh=0 无法独立比较。

## 4. 效率（30 题配对、现场全路径、decode64、交错+15 预热）

| 臂 | TTFT(ms) | Stage-1 | Facility | Completion | decode64 | VRAM(MB) |
|---|---:|---:|---:|---:|---:|---:|
| FULL(keep-all 1024) | 475.5 | — | — | — | 2645 | 17063 |
| E_GATHER | 402.1 | 81.25(含 s1+fac) | — | — | 2640 | 16933 |
| E_MAIN025 | 404.2 | 81.56(含 s1+fac) | — | 2.86 | 2657 | 16933 |
| R_MAIN025 | 404.7 | 2.14 | 78.71 | 2.81 | 2633 | 16933 |

R 相对 E 的 TTFT 增量 +0.13%；RTG 评分免除 instruction-embedding
额外一遍，净成本与 EADP 相当。

## 5. 正确性门与诊断

- T1–T7 全过（协议 §5）：独立逐元素参考（err≤2e-8）、chi-square 恒等式
  （FP64 err 2.2e-16）、退化/多图/剪枝路径、SHUF 排序不变+权重改变例、
  E 臂 9 题 bitwise 复现、新 scorer 现场=bank、λ=0 恒等。
- 诊断：RTG 有效文本 token p50≈19.6（L 均值 21.6；FLAT 上限 21.0）；
  anchors 与 EADP Jaccard 0.65–0.66（300/300）。

## 6. 缺项与未完成（截至 2026-10-03 11:00 前）

- MMBench 的 R−E_GATHER **逐题配对 CI 未完成**（循环评测结果文件 1292 行
  旋转去重，无法与 4876 行预测 1:1 对齐）；仅提供官方 headline −0.39。
- E_MAIN025 非回归剩余臂与 EADP 锚 fresh 消融（P3）：按截止时间在安全
  分片边界停止，完成部分见 `analysis_nonreg_paired.json` 更新与运行台账。
- 非回归 CI 均为补充比较；"MMLU 类知识任务"等未在本轮范围。

## 7. 运行台账与资源

- DEV：4 新臂 × 300 = 1200 生成 + E 臂复用 600；全量：R_MAIN025 11349。
- 实际 GPU 时间：pilot 1430s（预算 7200s）；P0 bank+生成+评分 ≈ 3.6h；
  P1 ≈ 1.2h；P2 ≈ 2.8h（至本报告时点）。共享服务器负载波动
  （15 用户，load 峰值 9.3）导致 0.3–1.8 s/q 波动。
- 偏离：D-7（fresh per-arm bank bug，错误数据未进入任何交付分析）、
  D-8（ChartQA Overall / POPE explode 修复，S0 硬失败按设计拦截）。
- 续跑命令：`rtg_full_run_all.sh` / `final_queue2.sh` 各阶段日志与
  分片（可续跑）均在本目录与 outputs 下。
