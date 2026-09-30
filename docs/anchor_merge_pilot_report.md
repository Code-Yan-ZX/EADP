# Anchor-Merge Pilot — 报告

**Branch**: `codex/anchor-merge-pilot` · base commit `a724122` · 2026-10-01
**Protocol**: `docs/anchor_merge_pilot_protocol.md`(冻结于任何准确率数字之前)
**Model**: Qwen3-VL-8B-Instruct-1024,native E0 engine(DeepStack on,3D mRoPE,
greedy),1024 → 256 visual tokens,pre-LLM,forward-only,无训练。

---

## 0. 一句话结论

**确认改善,且机制对照支持方向**:固定 official-facility 的 256 个 anchors,
把被删 token 按余弦分配进 anchor 组、以 λ=0.25 的组均值修正 anchor 表示
(U025),在 image-disjoint 的 600 题 CONFIRM 面板上相对官方-facility 纯选择
基线 **+1.40 macro 点,95% CI [+0.32, +2.57]**(rescued 11 / broken 3);
收益**不能**由范数变化解释(NORM-ONLY +0.26,CI 跨零;U025 − NORM +1.14,
CI [+0.09, +2.30]),而打乱修正内容(SHUFFLED-DELTA,保范数)则失去全部
收益并转负(U025 − SHUF **+2.05**,CI [+0.87, +3.33])。代价:每次生成
**+4.3 ms** 分组+池化(TTFT 287.5 → 292.6 ms,+1.8%),显存不变。

判定等级(按协议 §9):**确认改善;机制对照支持"内容修正携带了组内信息"
这一方向,但 SHUF 同时制造分布外扰动、NORM 仍可能携带标量信息,故不宣称
"已证明补回了答案证据",也不宣称收益来源已被完全分清。**

---

## 1. 正确性(先于任何准确率运行,`correctness.json`)

| gate | 内容 | 结果 |
|---|---|---|
| G1 | keep-all native vs stock:prefill + 32 步 forced decode logits | 12/12 样本 **bit-exact**(max\|Δ\|=0.0),0 token mismatch |
| G2 | 合并代码 λ=0 vs b1 纯 gather | 特征 **bitwise 相等**(主路+3 路 DS),logits Δ=0,预测一致;bank == 现场重算(S+gid 全等) |
| G3 | 聚合 vs 逐组 Python 参考实现 | 合成 1.2e-7、真实样本 3.2e-8(FP32 域);bf16 输出 cast 与生产路径 bitwise 相等 |
| G4 | 退化与不变量 | N=K 恒等、单成员组 y=a、空 dropped 恒等;12/12 样本 n_vis_kept=256、ds_lengths=[256]×3、每 dropped token 恰属一组、layer_calls/cache 一致 |
| G5 | 可重复性 | 进程内重复预测一致(3/3);**跨进程**合并特征哈希与预测全等(对照 M4 的 1.1–1.3 macro run-to-run 漂移,本轮确定性归约下为零) |

Smoke(5 arms × 30 题):空答案 0、截断 0、异常重复 0;每次生成重建 state
(无已 decode cache 复用)。

## 2. DEV 面板(5 arms × 300 = 1500 次生成)

逐题官方规则得分(0–100),三任务等权 macro。BASE = official-facility 纯选择
(协议中的 b1,与 block8/b2 严格区分)。

| arm | TextVQA | DocVQA | OCRBench | **macro** | Δ vs BASE | 95% CI | rescue/break |
|---|---:|---:|---:|---:|---:|---|---:|
| BASE | 78.70 | 78.77 | 86.00 | **81.157** | — | — | — |
| U025 | 81.00 | 79.96 | 87.00 | **82.652** | **+1.49** | [+0.06, +3.19] | 7/2 |
| U050 | 79.60 | 79.64 | 88.00 | 82.413 | +1.25 | [−0.70, +3.19] | 9/5 |
| U100 | 79.90 | 75.08 | 90.00 | 81.659 | +0.50 | [−2.10, +3.17] | 13/9 |
| S025 | 81.00 | 78.37 | 88.00 | 82.455 | +1.30 | [+0.02, +2.68] | 5/1 |

- 四个候选**全部**点估计优于 BASE → 按冻结规则消耗 CONFIRM。
- **Winner = U025**(macro 最高;并列时先 U025、再取更简单 arm 的规则未触发)。
  冻结于 CONFIRM 任何数字之前(winner_kind=uniform, λ=0.25)。
- 观察(非确认结果):U100(完整组均值)在 DocVQA −3.70 但 OCRBench +4.00,
  收益方向随任务分裂;λ 轴呈倒 U,U025 最稳。DEV 是复用评估集(E0/M12/M13
  的旧 arm 在同一批行上有历史分数),只能用于筛选。

## 3. CONFIRM 面板(4 arms × 600 = 2400 次生成)

CONFIRM 取自 e0 分区的 confirm 池(E0 记录从未读取;M12–M14 未使用;
与 SAGE confirmation/M1 扩展/bank150 按 image 排除),每数据集 200 行。
OCRBench 池仅 164 行,按冻结规则补 36 行未用 DEV 后缀(与 DEV-100
image-disjoint;该 36 行属 E0 DEV 池,标注为部分复用)。manifest 与
sha256 见 `outputs/anchor_merge_pilot/manifest.json`。

| arm | TextVQA | DocVQA | OCRBench | **macro** | Δ vs BASE | 95% CI | rescue/break |
|---|---:|---:|---:|---:|---:|---|---:|
| BASE | 83.30 | 76.53 | 72.50 | **77.442** | — | — | — |
| **U025 (winner)** | 83.60 | 76.91 | 76.00 | **78.838** | **+1.40** | **[+0.32, +2.57]** | **11/3** |
| NORM-ONLY | 83.50 | 76.09 | 73.50 | 77.697 | +0.26 | [−0.57, +1.13] | 4/3 |
| SHUFFLED-DELTA | 82.80 | 75.57 | 72.00 | 76.790 | −0.65 | [−1.82, +0.51] | 5/9 |

机制对照(同题配对、按 image cluster bootstrap,5000 次,seed 20261001):

| contrast | Δ macro | 95% CI | per-ds Δ |
|---|---:|---|---|
| U025 − BASE | **+1.396** | [+0.324, +2.569] | TV +0.30 / DV +0.39 / OCR **+3.50** |
| NORM − BASE | +0.255 | [−0.573, +1.132] | TV +0.20 / DV −0.44 / OCR +1.00 |
| SHUF − BASE | −0.652 | [−1.823, +0.513] | TV −0.50 / DV −0.96 / OCR −0.50 |
| **U025 − NORM** | **+1.141** | **[+0.091, +2.303]** | TV +0.10 / DV +0.82 / OCR +2.50 |
| **U025 − SHUF** | **+2.047** | **[+0.872, +3.330]** | TV +0.80 / DV +1.34 / OCR +4.00 |

解读(按协议 §6 的限定):

1. **主效应可确认**:U025 − BASE 的 CI 不跨零,rescue/break 11/3;
   逐数据集方向一致(OCRBench 贡献最大 +3.5)。
2. **范数解释被排除**:仅把 anchor 缩放到真实 ‖y‖(NORM-ONLY)只有 +0.26
   且 CI 跨零;winner 相对它仍 +1.14(CI 不跨零)→ 收益主要来自 Δ 的
   **方向/内容**,不是"范数变大"。注意 NORM-ONLY 仍可能携带组内标量信息
   (协议已声明它不是完全无内容控制)。
3. **内容对应是必要的**:保范数地打乱 Δ(SHUF)使收益消失并转负。
   与 NORM 合并读:同样幅度的扰动,方向指向组均值时 +1.40、内容错配时
   −0.65。但打乱本身制造分布外扰动,winner 胜过 SHUF **不能单独**证明
   "补回了答案证据";最稳妥的表述是:收益依赖 anchor 与补充内容之间的
   对应关系,且不是可由范数变化复现的尺度效应。
4. 无答案、无截断、无退化重复(全部分 arm × 数据集为 0)。

## 4. 机制诊断(frozen bank,CONFIRM;DEV 数字几乎相同)

- 组大小:均值 3.0(=768/256),p50 2–3,p90 6,max 61–120;**单成员组
  (无 dropped 可吸收)占 7.2–14.7%** —— 这些 anchor 上 λ 修正是精确零。
- ‖m−a‖/‖a‖(主特征):均值 0.42–0.53 → U025 的实际修正 ≈ **11–13% 范数**。
- cos(a, m):主特征 0.93–0.95(组均值与 anchor 高度同向);**DS 流仅
  0.83–0.86** —— 聚合对 DeepStack 流的方向扰动明显更大,而三路 DS 的
  ‖m−a‖/‖a‖ ≈ 0.47–0.52 与主路相当。位置近似(anchor 位置承载组内所有
  内容)仍是本设计的已知近似。

## 5. 效率(配对,30 块 × 交错 + 15 次预热,CUDA 同步,每块全新 state)

| 指标 | BASE | U025 | 差 |
|---|---:|---:|---:|
| 端到端 TTFT median (ms) | 287.5 | 292.6 | **+5.1 (+1.8%)** |
| — 其中 vision (ms) | 110.8 | 110.8 | 0 |
| — 其中 LLM prefill (ms) | 78.8 | 78.4 | −0.4 |
| — 分组+池化 merge (ms) | 0 | **4.25** | +4.25 |
| 固定 64-token 生成 (ms) | 2528.0 | 2521.0 | −7 |
| 峰值显存 p50 / max (MB) | 17001 / 17002 | 17001 / 17002 | ≈0 |
| 实际视觉 token 数 | 256 | 256 | 0 |

official-facility 选择器本身的成本另测:n=18,median **42.65 ms**/图
(与 E0 审计的 42.91 ms 一致)。本轮所有 arm 共享同一冻结 bank,正式对比中
选择成本为零;若部署时现场重选,需计入该 42.65 ms(或换 block8,但那将
改变 anchor 集与本轮结论的绑定)。合并是每图 O(N·K) 余弦 + 确定性组和,
GPU 上 4.25 ms。

## 6. 统计功效与边界

- DEV 300 / CONFIRM 600 的 cluster bootstrap 分辨率约 ±1–2 点:本轮
  CONFIRM 的 +1.40 [CI +0.32, +2.57] 可以与零区分,但 **+0.5 点级别的
  分数据集差异(如 TextVQA +0.30)不可区分**。
- CI 跨零处(NORM − BASE、SHUF − BASE)一律写"不确定",不写"相同"。
- 本轮不外推:仅 Qwen3-VL-8B、1024→256、OCR 三任务、官方 facility 选择器、
  λ=0.25 均匀聚合。λ 轴未搜索(U050/U100 是预注册网格点,非调参产物);
  未加任何训练、OT、PPE、residual rescue。

## 7. 运行台账

- 生成总数:DEV 1500 + CONFIRM 2400 + smoke 150(其中 90 题并入 DEV 分片)+
  gates/smoke 中的短生成 ≈ **4100+ 次贪心生成**。
- 总耗时(含模型加载、两次 gate 全跑、重评分):约 **2.5–3 小时** GPU 时间,
  单卡 A40,期间无其他任务争用(计时窗口均空闲)。
- 失败/未执行:CONFIRM OCRBench 不足 200 题(池 164 + 36 部分复用补齐,
  规则冻结于看分前);除此之外无失败、无跳过。
- 偏离:协议 §10 的 D-1( gate 比较域修正)、D-2(分片持久化 bug)、
  D-3(嵌套 headline 解包)、D-4(OCRBench 逐题重算规则与官方两处偏差:
  `\n` 归一化、math 分支大小写——修复后 27/27 个评分单元逐题均值复现官方
  headline)。所有偏离发生在锁分之前或只影响评分实现,不影响实验定义。

## 8. 交付物

- 代码:`Qwen_vl/scripts/anchor_merge_pilot/`(amp_common / amp_manifest /
  amp_bank / amp_correctness / amp_accuracy / amp_analyze / amp_perf /
  amp_report / run_all.sh,全部可用 `run_all.sh` 顺序复现)。
- 协议:`docs/anchor_merge_pilot_protocol.md`(含偏离记录)。
- 产物:`Qwen_vl/outputs/anchor_merge_pilot/`
  - `manifest.json`(DEV/CONFIRM 样本、image key、sha256)
  - `bank_dev_*.json.gz`、`bank_confirm_*.json.gz`(冻结 S + 分组 + 诊断)
  - `correctness.json`(G1–G5 + 跨进程哈希)
  - `acc/{dev,confirm}/<arm>/K256/<ds>.json`(逐题预测、截断、计时)+ 
    `_pred.tsv` / `_pred_results.xlsx` / `_score.json`(官方分 + 逐题分 +
    headline 复现自检)
  - `analysis_dev.json` / `analysis_confirm.json`(macro、paired CI、
    rescue/break、质量、bank 诊断)
  - `amp_perf.json`(配对计时全量块)
- 本报告:`docs/anchor_merge_pilot_report.md`。

## 9. 下一步建议(不在本轮执行)

若继续此方向,最有证据支持的一步是**学习受限的 anchor 内容修正**
(例如以逐题答案变化为监督、只在 Δ 的低维子空间上学习),因为本轮已证明:
(1) 固定 anchors 下注入组内内容有可确认的 +1.4 点;(2) 收益依赖内容对应
而非范数;(3) λ 均匀修正已可拿到全部收益的大头,说明修正量的学习空间
可能很小。同时值得单测 DS 流:其组均值方向偏离 anchor 更大(cos 0.83 vs
0.95),按流分别调 λ 或对 DS 用保方向修正是两个无训练的现成变体。
