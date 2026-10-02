# Stage 1 Cross-Stream Redundancy Pilot — 报告

**Branch**: `codex/stage1-cross-stream-pilot` · 2026-10-02
**Protocol**: `docs/stage1_cross_stream_pilot_protocol.md`(冻结于任何本轮数字之前,
commit `7db444a`)· 数据:anchor-merge manifest DEV-100×3(reuse evaluation set,
manifest sha256 `8d55e4e4…`)· 环境:qwen3vl_clean(python 3.10.20 / torch
2.10.0+cu128 / transformers 4.57.6),单卡 A40。

## 0. 一句话结论

**负结果。** 以主路 kNN 邻居在其他 DeepStack 流中的残差构造的
`cross_stream` Stage-1 **未接近 EADP Stage-1**:主要对比
X_MAIN025 − E_MAIN025 = **−1.71 [−5.14, +1.85]**、X_GATHER − E_GATHER =
**−0.61 [−3.71, +2.63]**(macro 点,新 − 旧;20k image-cluster bootstrap,
seed 20261002),按预注册 0.5 点筛选规则**不支持非劣**(CI 下界 < −0.5),
也未见超出 uniform / 主路残差对照的线索。额外端到端延迟很小
(TTFT +4.7~5.6 ms,<1.5%)。**按本轮约定不建议进入新图像面板确认。**

## 1. 正确性(`correctness.json`,G1–G7 全过)

- **G1** 向量化 FP32 评分 vs 逐点显式参考:cross_stream rel=9.4e-8、
  main_residual rel=2.5e-7(<1e-5);uniform 恒等;w 有限非负、确定;
  邻居无自身、并列取更小索引、原矩阵未被破坏;含重复行/零行合成样本。
- **G2** 双图拼接:concat 评分选择 = 独立两图结果拼接(3 个新 scorer 全过)
  → 无跨图邻域/分组。
- **G3** 全零流贡献恒为零(d_mean=0、w 与双流参考一致)且被计数;N≤K
  直接保留;重复 token 确定。
- **G4 E 复现**:每数据集 3 题(共 9),E_GATHER 与旧 BASE、E_MAIN025 与旧
  MAIN025 的 anchors(bank keep bitwise)与**贪心输出全文 bitwise** 一致;
  全量 300 题逐题聚合与 round-2 DEV 完全相同(81.157 / 83.267,见 §2)。
  旧 shard 未存 logits,prefill-logits 一致性由 G5 的 λ=0 bitwise 门覆盖
  (如实记录)。
- **G5** λ=0 路径 vs 纯 gather:eadp 与 cross_stream 各自 anchors 下主路/三路
  DS 特征 bitwise 相等、prefill logits 差 = 0。
- **G6** 4 个 scorer 冻结 bank 与现场重算 keep/gid 全等(9 样本 × 4)。
- **G7** 六组不变量(n_vis_kept=256、无重复、升序、ds_lengths=[256]×3、
  layer_calls/cache ok)+ DS 完整性(DS_sel = DS[keep] logits bitwise 相等)。
- **评分自检(严格)**:18/18 个 (arm, dataset) 评分单元逐题均值在浮点精度
  (1e-6)内复现官方 headline(Text/DocVQA `Overall=np.mean(hit)*100` 未舍入;
  OCRBench Final Score 为精确计数)。

## 2. 六组 DEV 主表(每任务 0–100,macro 三任务等权)

| arm | Stage 1 | Completion | TextVQA | DocVQA | OCRBench | macro |
|---|---|---|---:|---:|---:|---:|
| E_GATHER | eadp | — | 78.70 | 78.77 | 86.00 | **81.157** |
| E_MAIN025 | eadp | main λ=0.25 | 81.30 | 80.50 | 88.00 | **83.267** |
| X_GATHER | cross_stream | — | 79.20 | 74.46 | 88.00 | **80.552** |
| X_MAIN025 | cross_stream | main λ=0.25 | 79.30 | 77.38 | 88.00 | **81.559** |
| U_MAIN025 | uniform | main λ=0.25 | 79.90 | 76.01 | 92.00 | **82.637** |
| V_MAIN025 | main_residual | main λ=0.25 | 80.20 | 75.43 | 84.00 | **79.875** |

E_GATHER/E_MAIN025 的三个 per-dataset 分数与 round-2 DEV BASE/MAIN025 逐题
完全一致(macro 81.157 / 83.267)——本轮执行路径与历史 bitwise 可复现。

## 3. 预注册对比(20,000 次 image-cluster bootstrap,seed 20261002;差值 = 新 − 旧)

| 对比 | Δ macro | 95% CI | 判定(margin 0.5) | rescue/break |
|---|---:|---|---|---|
| **X_MAIN025 − E_MAIN025**(主要) | −1.709 | [−5.136, +1.851] | 不确定(不支持非劣) | 12 / 17 |
| **X_GATHER − E_GATHER**(主要) | −0.606 | [−3.713, +2.634] | 不确定(不支持非劣) | 13 / 14 |
| X_MAIN025 − U_MAIN025(机制) | −1.079 | [−4.017, +1.805] | 不确定 | 9 / 13 |
| X_MAIN025 − V_MAIN025(机制) | +1.683 | [−1.724, +5.066] | 不确定 | 19 / 14 |
| X_MAIN025 − X_GATHER(机制) | +1.007 | [−0.668, +2.736] | 不确定 | 6 / 3 |
| 2×2 交互(联合 bootstrap) | −1.103 | [−3.538, +1.249] | 描述性 | — |

分任务(X − E):
- 主配对:TextVQA −2.00 / DocVQA −3.13 / OCRBench 0.00;
  gather 配对:TextVQA +0.50 / DocVQA −4.32 / OCRBench +2.00。
- 主要拖累在 DocVQA;OCRBench 对 anchors 变化最不敏感(X 两组均 88.00);
  TextVQA 上 X 与 E 的 gather 基本持平,completion 后被 EADP anchors 反超
  (2×2 交互 −1.10:completion 增益在 cross_stream anchors 上更小)。

## 4. 机制解读(有界,不外推)

- **scorer 确实改变了 anchors**:三个新 scorer 与 EADP 的 keep 在 300/300
  样本上全部不同(非仅热图差异);与 EADP 的平均交集 ≈ 132–167/256
  (Jaccard 0.35–0.49)。
- **cross_stream 没有跑赢 uniform**:X_MAIN025 − U_MAIN025 = −1.08。仅用
  "facility + 无信息均匀权重" 选出的 anchors + 同一 completion,DEV macro
  已达 82.64,高于 cross_stream 的 81.56——DS 三路残差提供的排序信号
  在本 DEV 上没有体现为更好的 anchors。
- **主路残差(V)更差**:79.88,是六组最低;说明"主路邻居在主路自身中的
  可替代性"比"在其他流中的可替代性"信号更弱,即 cross_stream 的跨流部分
  相对主路残差有 +1.68 的方向性改善,但 CI 跨零,不足以支撑"跨流信号存在"
  的结论。
- 与历史 KILL 一致:Stage-1 grounding(entropy/grounding 融合)与 Stage-1
  visual-calibration(kNN-novelty)两轮均已否证;本轮第三种 Stage-1 替代
  信号同样未通过 0.5 非劣筛选。

## 5. 真实效率(`xsp_perf.json`,30 题配对,现场全路径,64 token 固定,中位数)

| arm | TTFT | similarity | Stage-1 评分 | facility | completion | prefill | decode64 | 总生成 | VRAM p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E_GATHER | 392.6 | — | — | — | — | 92.8 | 2666 | 3062.5 | 17001 |
| E_MAIN025 | 396.7 | — | — | — | 2.63 | 92.5 | 2649 | 3054.8 | 17001 |
| X_GATHER | 397.3 | 0.77 | 5.73 | 76.66 | — | 92.7 | 2643 | 3058.0 | 17015 |
| X_MAIN025 | 402.3 | 0.77 | 5.76 | 76.77 | 2.56 | 92.6 | 2649 | 3049.7 | 17015 |

(单位 ms / MB。EADP 行的 `eadp_stage1_facility` 合计 = 78.7/79.4 ms,含
0.9 ms instruction embedding,官方路径不拆分,未重复计时段。)

- **TTFT 边界声明**:从 `prepare`(消息/processor)起,含图像预处理、
  encode、全部现场评分/选择/completion、prefill,至首 token logits——
  是完整端到端 pre-first-token,不是纯 prefill 或 bank 模式时间。
- cross_stream 的评分增量 = similarity 0.77 + Stage-1 5.73 ≈ 6.5 ms,被
  facility(~77 ms,两种 scorer 相同)淹没;相对 E_GATHER 的 TTFT 差
  **+4.7 ms(X_GATHER)/ +5.6 ms(X_MAIN025)**(<1.5%);decode 与总生成时间
  无差异;峰值显存 +14 MB。新 scorer 免去 instruction embedding(~0.9 ms)
  的差异已包含在各自完整路径中。

## 6. 诊断要点

- importance 分布(`w`):eadp mean 0.34 / p99 0.62 / max 0.87(β=2 极化后);
  cross_stream mean 3.10(=0.1+3×1 构造)/ p90 4.05–4.22 / max 9.3–11.7,
  重尾但有限;main_residual mean 1.10 / max 3.0–5.2;uniform 恒 1。
- 三路 DS 残差均值(d):流 1 ≈ 0.64、流 2 ≈ 0.59、流 3 ≈ 0.37(TextVQA
  为例,三数据集方向一致)——第三路 DeepStack 流的邻居残差系统性更低;
  逐流 z 归一化已吸收尺度差,但该信息未转化为更优 anchors。
- 零范数行:真实数据 0;合成零流测试贡献恒零(G1/G3)。
- 组大小(gsize):各 scorer 组均值 ≈ 3.0(=1024/256 构造),p90 6、
  max 38–65,六组共享同一 completion 机制,组间可比。

## 7. 运行台账与资源

- 生成量:DEV 6 组 × 300 = 1800 条预测(含 4 个新 arm 的 90 条 smoke 因
  bank-sha 严格核验而重新生成,见 D-1);每条均为真实贪心生成,
  max_new_tokens=2048,无截断。
- GPU 时间(单卡 A40,无争用):bank 构建(12+9 重建)≈ 20 min;正确性门
  ≈ 10 min;smoke ≈ 5 min;DEV 生成+评分 ≈ 40 min;分析(bootstrap)CPU;
  配对计时 ≈ 7 min;**合计 ≈ 1.5 h**。
- 失败:无未执行项;六组 × 三数据集全部完成并评分。
- 偏离(均发生在任何准确率数字之前或与数字无关,详见 protocol §9):
  - **D-1**:X/U/V 四组的首批 smoke shard 在 bank 重建(w 诊断补全,
    keep/gid 经逐样本核验 bitwise 不变)后未通过严格 bank-sha 续跑核验,
    该 90 条 smoke 预测删除重生成;E 组 shard 与旧 bank 哈希始终一致。
  - **D-2**:正确性脚本三处测试侧缺陷(门键名大小写、uniform 参考值、
    G3 零流测试构造)在首轮门运行中即暴露并修复后重跑,全部门通过。
  - **D-3**:G2/G3 合成样本的 gthw 几何与 token 数不一致(测试构造错误),
    修复后重跑;不影响生产路径。
  - **D-4**:2×2 交互由两次独立 bootstrap 之差改为四 arm 同抽样的联合
    bootstrap(分析侧增强,主表与主要对比不受影响)。
  - **D-5**:bank/shard 元数据记录的 code_commit 为 `7db444a`;此后仅
    测试/分析脚本与 bank 诊断字段变更,评分与选择代码
    (`xsp_common.select_keep`/`stage1_importance`)未改动。

## 8. 结论(对应协议 §8.3)

1. **Stage 1 是否接近 EADP**:否。两个主要对比点估计均为负
   (−1.71 / −0.61),CI 下界 −5.14 / −3.71 均低于 −0.5 margin,
   按预注册规则不支持非劣。
2. **整体是否接近当前冻结方法(DCC = official EADP + Facility + MAIN025)**:
   否,同上;DCC 维持不动。
3. **是否有超出 uniform / 主路残差的线索**:没有。X ≤ U(−1.08),
   X > V(+1.68)但跨零;"主路邻居在其他流中可替代"这一代理信号在本
   DEV 上未产生可辨识的 anchor 质量收益,且 2×2 交互(−1.10)提示其
   anchors 与 completion 的组合增益反而更小。
4. **额外延迟**:评分段 +6.5 ms,端到端 TTFT +4.7~5.6 ms(<1.5%),
   decode 不变——成本可忽略,问题在质量不在速度。
5. **是否值得进入新图像面板确认**:否。负方向点估计 + 无机制线索,
   按本轮筛选约定终止 `cross_stream` 方向(Stage-1 第三次 KILL)。

## 9. 产物

- 代码:`Qwen_vl/scripts/stage1_cross_stream_pilot/{xsp_common,xsp_bank,
  xsp_correctness,xsp_accuracy,xsp_analyze,xsp_perf}.py`
- 协议/报告:`docs/stage1_cross_stream_pilot_{protocol,report}.md`
- 数据(`Qwen_vl/outputs/stage1_cross_stream_pilot/`):
  `correctness.json`、`analysis_dev.json`、`xsp_perf.json`、
  `bank_dev_{scorer}_{ds}.json.gz` ×12、
  `acc/<arm>/{ds}.json|_score.json|_pred.tsv|_results.xlsx`(逐题预测与
  逐题得分,image_key/截断/实际 token 数齐全)
- 哈希:bank/shard 元数据内含 manifest、bank sha256、code_commit、
  模型与环境;correctness.json 记录 base_commit=`ad08187`。
