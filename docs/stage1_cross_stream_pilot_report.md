# Stage 1 Cross-Stream Redundancy Pilot — 报告(v2,纠错后)

**Branch**: `codex/stage1-cross-stream-pilot` · 2026-10-02
**Protocol**: `docs/stage1_cross_stream_pilot_protocol.md`(冻结于任何本轮数字
之前,`7db444a`;**D-6 纠错** `753cb99`)· 数据:anchor-merge manifest
DEV-100×3(reuse evaluation set,manifest sha256 `8d55e4e4…`)· 环境:
qwen3vl_clean(python 3.10.20 / torch 2.10.0+cu128 / transformers 4.57.6),
单卡 A40。

> **v2 说明(外部复核发现,见协议 D-6)**:首轮实现的 cross_stream 评分把
> 三路 z **求和**而非协议规定的**取平均**,且 G1 参考实现同错(门验证的是
> 实现而非协议)、G2 多图测试未进入剪枝分支。纠错后按用户指定最小范围
> 重跑:cross_stream bank 重建(anchors **300/300 全部改变**,逐图平均
> ~11–14 个 token 不同,证据 `prefix_mean_fix_anchor_diff.json`)→
> X_GATHER / X_MAIN025 全量重生成重评分(G1–G7 重跑全过)→ 分析/计时
> 重算。**求和版全部产物原样保留**于
> `outputs/stage1_cross_stream_pilot/prefix_sum_run/`(含其报告副本)。
> E/U/V 四组的评分与求和/均值无关(uniform 恒 1、main_residual 单流),
> 不受 D-6 影响,沿用首轮结果。本文所有数字均为**协议口径(均值)**;
> 求和版数字不作为原方案的实验结论。

## 0. 一句话结论

**负结果。** 协议口径下,`cross_stream` Stage-1 未接近 EADP Stage-1:
主要对比 X_MAIN025 − E_MAIN025 = **−3.13 [−6.51, +0.29]**、
X_GATHER − E_GATHER = **−1.35 [−4.62, +2.01]**(macro 点,新 − 旧;
20,000 次 image-cluster bootstrap,seed 20261002)。按预注册 0.5 点筛选
规则,**两者均属"不确定、未支持非劣"**——尚未证明损失超过容忍值,但也
没有任何支持非劣的证据,且 X 主配对点估计进一步偏负(CI 上界 +0.29 接近
排除零)。无超出 uniform / 主路残差对照的线索。额外端到端延迟
+3.1~4.4 ms(<1.2%)。**按本轮约定不进入新图像面板确认。**

## 1. 正确性(`correctness.json`,G1–G7 全过,纠错后重跑)

- **G1** 向量化 FP32 评分 vs 逐点显式参考(参考实现已改为协议的**均值**
  口径):cross_stream rel=2.0e-7、main_residual rel=2.5e-7(<1e-5);
  uniform 恒等;w 有限非负、确定;邻居无自身、并列取更小索引、原矩阵未
  被破坏;零流语义:零流计入且贡献零,使均值缩至双流值的 2/3(按协议
  精确验证)。
- **G2** 双图拼接,**K=16<N=64 实际走剪枝分支**(含防退化断言,每图恰好
  剪至 16):concat 选择 = 独立两图结果拼接(3 个新 scorer 全过)→ 无
  跨图邻域/分组。首轮 K=256 触发 keep-all、未测剪枝路径的缺口已修复。
- **G3** N≤K keep-all;重复 token 确定;零流贡献恒为零且计数正确。
- **G4 E 复现**:每数据集 3 题(共 9),E_GATHER 与旧 BASE、E_MAIN025 与
  旧 MAIN025 的 anchors(bank keep bitwise)与**贪心输出全文 bitwise**
  一致;全量 300 题逐题聚合与 round-2 DEV 完全相同(81.157 / 83.267)。
  旧 shard 未存 logits,prefill-logits 一致性由 G5 的 λ=0 bitwise 门覆盖
  (如实记录)。
- **G5** λ=0 路径 vs 纯 gather:eadp 与 cross_stream(修正 bank)各自
  anchors 下主路/三路 DS 特征 bitwise 相等、prefill logits 差 = 0。
- **G6** 4 个 scorer 冻结 bank 与现场重算 keep/gid 全等。
- **G7** 六组不变量 + DS 完整性(DS_sel = DS[keep] logits bitwise 相等)。
- **评分自检(严格)**:18/18 个 (arm, dataset) 评分单元逐题均值在 1e-6
  内复现官方 headline。

## 2. 六组 DEV 主表(每任务 0–100,macro 三任务等权;X 两行为纠错后)

| arm | Stage 1 | Completion | TextVQA | DocVQA | OCRBench | macro |
|---|---|---|---:|---:|---:|---:|
| E_GATHER | eadp | — | 78.70 | 78.77 | 86.00 | **81.157** |
| E_MAIN025 | eadp | main λ=0.25 | 81.30 | 80.50 | 88.00 | **83.267** |
| X_GATHER | cross_stream(均值) | — | 77.80 | 76.61 | 85.00 | **79.804** |
| X_MAIN025 | cross_stream(均值) | main λ=0.25 | 78.30 | 76.10 | 86.00 | **80.135** |
| U_MAIN025 | uniform | main λ=0.25 | 79.90 | 76.01 | 92.00 | **82.637** |
| V_MAIN025 | main_residual | main λ=0.25 | 80.20 | 75.43 | 84.00 | **79.875** |

## 3. 预注册对比(20,000 次 image-cluster bootstrap,seed 20261002;差值 = 新 − 旧)

| 对比 | Δ macro | 95% CI | 判定(margin 0.5) | rescue/break |
|---|---:|---|---|---|
| **X_MAIN025 − E_MAIN025**(主要) | −3.133 | [−6.505, +0.294] | 不确定(未支持非劣) | 10 / 19 |
| **X_GATHER − E_GATHER**(主要) | −1.353 | [−4.616, +2.005] | 不确定(未支持非劣) | 13 / 14 |
| X_MAIN025 − U_MAIN025(机制) | −2.503 | [−5.719, +0.758] | 不确定 | 9 / 17 |
| X_MAIN025 − V_MAIN025(机制) | +0.259 | [−3.255, +3.782] | 不确定 | 17 / 16 |
| X_MAIN025 − X_GATHER(机制) | +0.331 | [−1.105, +1.903] | 不确定 | 3 / 4 |
| 2×2 交互(联合 bootstrap) | −1.779 | [−4.192, +0.598] | 描述性 | — |

分任务(X − E):主配对 TextVQA −3.00 / DocVQA −4.40 / OCRBench −2.00;
gather 配对 TextVQA −0.90 / DocVQA −2.16 / OCRBench −1.00。**纠错后 X 在
三个任务上均低于 E**,主要拖累仍是 DocVQA。

## 4. 机制对照(有界,不外推)

- **scorer 确实改变了 anchors**:三个新 scorer 与 EADP 的 keep 在 300/300
  样本上全部不同;与 EADP 平均交集 ≈132–167/256(Jaccard 0.35–0.49)。
  求和→均值纠错本身也改变了全部 300/300 个 cross_stream anchors
  (新旧交集 ≈242–245/256),证实该实现偏差可能影响选择,与复核判断一致。
- **X 未跑赢 uniform**:X_MAIN025 − U_MAIN025 = −2.50。仅用
  "facility + 均匀权重" 的 anchors + 同一 completion 已达 82.64。
- **X 与 main_residual 不可分辨**:+0.26 [−3.26, +3.78]。首轮(求和版)
  曾为 +1.68;纠错后连方向优势都消失——"跨流信号相对主路残差有增量"的
  说法在协议口径下不成立。
- **2×2 交互 −1.78 [−4.19, +0.60]**:completion 增益在 cross_stream
  anchors 上更小(描述性,CI 含零)。
- **uniform 接近 EADP 的正确读法**:U_MAIN025 相对 E_MAIN025 为
  TextVQA −1.40 / DocVQA −4.49 / OCRBench +4.00——macro 的接近来自
  DocVQA 与 OCRBench 方向相反的抵消,**不能解读为"Stage-1 importance
  无作用"**。
- **OCRBench 并非无信息**:六组范围 84.00–92.00,对 scorer 明确敏感;
  只是 X 恰好不占优(85/86 vs E 86/88)。
- 与历史 KILL 的关系:这是 Stage-1 第三次否证(grounding `d50da38`、
  visual-calibration `ad08187` 之后)。前两轮"栽在 DocVQA"的机制假设
  本轮**未做案例级或干预级验证**,仅作为待检验假设记录,不作为结论。

## 5. 真实效率(`xsp_perf.json`,纠错后重跑;30 题配对,现场全路径,64 token 固定,中位数)

| arm | TTFT | similarity | Stage-1 评分 | facility | completion | prefill | decode64 | VRAM p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E_GATHER | 399.8 | — | — | — | — | 92.4 | 2631 | 17001 |
| E_MAIN025 | 401.9 | — | — | — | 2.63 | 92.4 | 2648 | 17001 |
| X_GATHER | 404.2 | 0.77 | 5.78 | 78.41 | — | 92.8 | 2638 | 17015 |
| X_MAIN025 | 405.0 | 0.77 | 5.76 | 78.32 | 2.56 | 92.3 | 2642 | 17015 |

(单位 ms / MB。EADP 行 `eadp_stage1_facility` 合计 80.6 ms,含 0.9 ms
instruction embedding,官方路径不拆分,未重复计时段。)

- **TTFT 边界声明**:从 `prepare`(消息/processor)起,含图像预处理、
  encode、全部现场评分/选择/completion、prefill,至首 token logits——
  完整端到端 pre-first-token,不是纯 prefill 或 bank 模式时间。
- cross_stream 相对 E_GATHER 的 TTFT 差 **+4.4 ms(X_GATHER)/ +3.1 ms
  (X_MAIN025)**(<1.2%);decode 与总生成时间无差异;峰值显存 +14 MB。
  评分段(similarity 0.77 + Stage-1 5.8 ≈ 6.5 ms)被 facility(~78 ms)
  淹没;新 scorer 免去 instruction embedding 的差异已计入各自完整路径。

## 6. 诊断要点(`analysis_dev.json` diagnostics)

- importance(`w`,均值口径):eadp mean 0.34 / max 0.87;cross_stream
  mean ≈1.10(=0.1+3×(1/3) 构造,纠错前求和版为 3.10)/ p90 1.35–1.41 /
  max 3.1–3.9;main_residual mean 1.10 / max 3.0–5.2;uniform 恒 1。
- 三路 DS 残差均值(d):流 1 ≈ 0.64、流 2 ≈ 0.59、流 3 ≈ 0.37(TextVQA,
  三数据集方向一致)——第三路流邻居残差系统性更低;逐流 z 归一化吸收了
  尺度差,但该信息未转化为更优 anchors。
- 零范数行:真实数据 0;合成零流行为按协议精确验证(G1/G3)。
- 组大小:各 scorer 组均值 ≈3.0(=1024/256 构造),p90 6、max 38–65。

## 7. 运行台账与资源

- 生成量:DEV 6 组 × 300 = 1800 条(首轮;X 两臂纠错后重新生成 600 条,
  合计实际生成 2400 条)。每条均为真实贪心生成,max_new_tokens=2048,
  无空答案/截断/异常重复。
- GPU 时间(单卡 A40,无争用):首轮(bank 12+9、门、smoke、DEV、计时)
  ≈ 1.5 h;纠错轮(X bank 重建 ≈5 min、门重跑 ≈12 min、X 两臂重生成+
  评分 ≈22 min、计时重跑 ≈8 min)≈ 47 min。**合计 ≈ 2.4 h。**
- 失败:无未执行项;纠错后六组 × 三数据集全部完成并评分。
- 偏离:完整清单见协议 §9(D-1~D-6)。D-6 为外部复核发现的关键实现偏差
  (求和 vs 均值),处理方式:旧结果归档不覆盖、修正公式与两个门缺口、
  按 anchors 对比决定重跑范围(实际全变 → 全量重生成 X 两臂)。

## 8. 结论(对应协议 §8.3,v2 口径)

1. **Stage 1 是否接近 EADP**:否。协议口径下两个主要对比点估计均为负
   (−3.13 / −1.35),按预注册规则属"不确定、未支持非劣"——没有证据
   支持非劣,也没有证据证明损失超过容忍值;不存在正信号。
2. **整体是否接近当前冻结方法(DCC = official EADP + Facility +
   MAIN025)**:否;DCC 维持不动。
3. **是否有超出 uniform / 主路残差的线索**:没有。X < U(−2.50),
   X ≈ V(+0.26,跨零);纠错前"跨流信号优于主路残差"的方向性优势在
   协议口径下消失。
4. **额外延迟**:评分段 ≈6.5 ms,端到端 TTFT +3.1~4.4 ms(<1.2%),
   decode 不变——成本可忽略,问题在质量不在速度。
5. **是否值得进入新图像面板确认**:否。按本轮筛选约定终止
   `cross_stream` 方向(Stage-1 第三次 KILL)。

## 9. 产物

- 代码:`Qwen_vl/scripts/stage1_cross_stream_pilot/{xsp_common,xsp_bank,
  xsp_correctness,xsp_accuracy,xsp_analyze,xsp_perf}.py`(v2 = `753cb99`
  起的均值口径)
- 协议/报告:`docs/stage1_cross_stream_pilot_{protocol,report}.md`
- 数据(`Qwen_vl/outputs/stage1_cross_stream_pilot/`,已入库小型 JSON):
  `correctness.json`(纠错后)、`analysis_dev.json`(纠错后,含逐题分数
  引用)、`xsp_perf.json`(纠错后)、`prefix_mean_fix_anchor_diff.json`
  (D-6 anchors 对比证据)、`acc/<arm>/<ds>_score.json` ×18(逐题得分,
  可独立重算 CI)、`acc/<arm>/<ds>.json`(逐题预测)
- 归档:`prefix_sum_run/`(求和版 banks、X 两臂 shard、analysis/perf/
  correctness、报告副本)
- 大型文件(bank *.json.gz、pred.tsv/results.xlsx)不入库,服务器本地
  保留;bank/shard 元数据含 manifest、bank sha256、code_commit、模型与
  环境,可完整重建。
