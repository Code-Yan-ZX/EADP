# 四流共同选集离线诊断：结果

状态：2026-11-09 完成。32/32 个新 TRAIN 父文档、480/480 条真实 QA 生成、31 项门禁以及 7 个 forward reference 的 10 组计时均完成；未训练模型。预注册见 [multistream_oracle_prereg.md](multistream_oracle_prereg.md)。

**结论：signed teacher 出现局部正信号，但没有证明四流选集优于较强的 V-only magnitude teacher，也没有部署方法。** signed 在 DS on 时由 22/32 提高到 27/32，7 修复、2 破坏；其 DS 交互区间仍跨零。magnitude 的 V-only 与 joint 均为 31/32，无独立增益。所有 teacher 都使用该题的 gold 答案及 backward，以下准确率只能解释为答案知情的离线诊断。

## 设置与可解释范围

- 原生 Qwen3-VL-8B-Instruct，1024-square、1024→256 hard selection，原生 interleaved mRoPE；同一个空间选集同步作用于主 V 和三条 DeepStack。DS-off 只关闭三次注入，保留物理视觉序列。
- 新增 32 个 DocVQA TRAIN 独立父文档，每页一个既有数值 QA；另有一个独立开发父文档。排除 VAL、旧 fit/CF/binding/LayoutLite/CatchUp，以及上一轮 attention-calibration 的全部 33 个父文档。不按完整模型正确或压缩失败筛选；全部 32 题保留。
- 同一次 full1024、DS-on 的 mean answer-token NLL backward 给出四个 leaf 梯度。目标是原 `answers[0]`，不加 EOS。magnitude 为逐坐标 `Σ|g*x|`，signed 为 `−Σ(g*x)`；joint 直接按原始尺度求四流和，无归一化、无 signed clipping。
- 每种分数取稳定 Top256，再按原 raster 顺序送入 stock native decoder；四个选集分别做 DS on/off。正常 EOS、greedy、最多 32 新 token。主指标为预注册 numeric exact，ANLS 和原始别名 literal exact 为辅助。

使用 gold、全序列一阶梯度以及对当前题答案的选择，使本轮不具备 held-out 部署准确率含义。门控缩放的导数也不等于删掉 768 个 token 的有限干预：物理剪枝会改变序列长度、softmax 和联合删除关系。因此这些 teacher 不是精确删除 oracle，也不是严格的准确率上界。没有定位到可核验的 GeoPrune 官方代码，本轮未复现 GeoPrune，不声称超过它。

## 主指标

所有单元格分母均为 32；全部 15 臂均无截断、无待补输出。

| 选集/参考 | DS on exact | DS off exact |
| --- | ---: | ---: |
| 完整 1024-token 模型 | 31 | 29 |
| B2，K256 | 23 | 22 |
| 固定随机选集，K256 | 18 | 14 |
| magnitude V-only，K256 | 31 | 29 |
| magnitude joint，K256 | 31 | 29 |
| signed V-only，K256 | 22 | 22 |
| signed joint，K256 | 27 | 23 |
| R-res512，原生 256 tokens | 23 | — |

magnitude 两种选集的生成文本，在各自 DS 状态下均与对应完整模型 **32/32 逐字符串相同**。这是当前 gold teacher 的恢复能力，不是四流教师的增量，也不能直接蒸馏为部署收益。

预注册 paired 比较使用相同父文档 bootstrap，10,000 次、seed=20261109；下表单位为百分点。所有题各有一个独立父文档。

| joint − V-only | 修复 / 破坏 | exact 差值 | 95% bootstrap CI |
| --- | ---: | ---: | --- |
| magnitude，DS on | 0 / 0 | 0.000 | [0.000, 0.000] |
| magnitude，DS off | 0 / 0 | 0.000 | [0.000, 0.000] |
| signed，DS on | 7 / 2 | +15.625 | [0.000, +34.375] |
| signed，DS off | 5 / 4 | +3.125 | [−15.625, +21.875] |

交互量 `(joint_on − V_on) − (joint_off − V_off)`：magnitude 为 0；signed 为 **+12.500pp，95%CI [−6.250, +34.375]pp**。signed 的点估计与 DS 相关解释相容，但小样本及跨零交互区间不足以确立独有 DS 因果贡献。DS-off 还改变了定义所有分数时的 teacher-on 执行分布。零差值的退化区间不证明两个算法等价。

相对强参考，magnitude V-only 和 joint 都对 B2 得到 8 修复、0 破坏；signed joint 对 B2 为 6 修复、2 破坏，+12.500pp，CI [−3.125,+28.125]；对 R-res 为 7 修复、3 破坏，+12.500pp，CI [−6.250,+31.250]。不得从多个小样本比较中挑出有利一项称为统计显著或新方法有效。

辅助 ANLS：完整模型和两个 magnitude-on 臂均为 0.9948，B2 0.8369，R-res 0.8625，signed V-on 0.7693、joint-on 0.9252。literal exact 下 signed joint-on 为 26/32，numeric exact 为 27/32；其余格式差异与全部逐题预测保存在分析产物中。全未截断敏感性仍是全部 32 题，full-correct 31 题分层仅为辅助，没有改变主分母。

## 分数与选集诊断

四张分数图的来源、原始四流和、signed 符号、稳定 TopK、共同 on/off 选集均通过 CPU 产物校验。magnitude joint 相对 V-only 平均改变 **26.06/256** 个选择，平均 Jaccard 0.8157；signed 平均改变 **129.19/256**，Jaccard 0.3302。两类真实生效的选择变化不能仅凭 overlap 大小解释为效用。

按页面先归一化后平均的 magnitude mass 份额约为 V **45.3%**，三条 DS 分别 **18.3% / 18.3% / 18.1%**。这个量混合了特征尺度与 Jacobian，不能称为答案信息量。score participation 的页面均值分别为 mag V 121.0、mag joint 116.6、signed V 87.1、signed joint 111.7；它不是“真正有效 token 数”。完整逐 token activation/gradient L2、分位数、正负/零计数均已保存。

teacher NLL 的中位数为 1.41e−5，范围 0 至 1.5834，其中 1 页记录为 0。接近饱和的答案分布使小梯度/符号稳定性仍需谨慎解释；本轮没有用该诊断重新选页、改目标或调 score。

## 成本：离线标签与部署参考分开

以下是同一独立开发页、A40、bf16、SDPA、batch 1 的同步 wall-clock 中位数。每臂 3 次 warm-up，10 个完整 paired blocks、轮换顺序；每次重新进行预处理、视觉编码及适用的 selector。16-token 总时间使用统一 ignore-EOS。仅测七个真正 forward-only 参考；没有把省略 teacher 的选集生成时间冒充部署 TTFT。

| forward reference | TTFT ms | 固定 16-token 总时间 ms |
| --- | ---: | ---: |
| full | 449.9 | 990.9 |
| full_no_ds | 445.6 | 987.5 |
| rres512 | 175.4 | 727.2 |
| b2 | 320.7 | 866.6 |
| b2_no_ds | 320.8 | 862.2 |
| random_on | 317.2 | 862.1 |
| random_off | 314.5 | 863.2 |

R-res 在这张开发页的 TTFT 明显低于完整编码后选择；DS-off 没有带来可观的速度收益。它仍编码全部三路 DS feature，只是不执行 decoder 注入。单页计时不能推广为跨数据集效率结论。

32 页离线标签成本独立记录：gold tokenization 中位数 0.23ms，full decoder forward 205.5ms，backward 281.3ms，scores/ranking/诊断 11.2ms，总 label 步骤 **563.3ms/页**，32 页累计 18.07s。此前特征提取与 B2 准备另计，中位数 248.2ms/页；不能把这些共享/离线成本隐藏掉。非 checkpointed teacher 的峰值 allocated 为 **21.85GiB**，reserved 21.95GiB；相应页 baseline allocated 约 16.41GiB。无学生训练成本，因为本轮没有训练。

## 门禁、首次入口失败与审计

最终主实验前 **31/31 门禁通过**，包括四流视觉输出/完整模型 prefill/四步 forced decode 的 bit identity、native teacher-forced/cached logits 与 NLL 对齐、四流梯度与无参数更新、稳定 K256、on/off 同位置及物理序列、cache 和异常恢复。实际配置 checkpointing=False。

保留的失败不是不利样本排除。最初 preflight 曾在模型加载时中断，记录于 `gates_interrupted_preflight.log`。其后首次联合入口把模型构造放进 `inference_mode`，在 `score_full_streams` 读取参数 `_version` 时失败；异常恢复又触发 inference tensor 的 `requires_grad` 限制。该错误发生在主 teacher forward/backward **之前**，未形成任何完整主样本 teacher/QA 记录。首次日志 `experiment_failed_attempt_1.log` 与 freeze `run_manifest_failed_attempt_1.json` 已归档；入口修复后重跑门禁，再重新冻结主实验。旧失败产物没有混入本轮 480 条预测。

完成后的独立 CPU 审计核对了 run/gates/data/dev 与所有冻结源码哈希，未见变更；回查 33 个原 parquet QA 和原图 SHA，问题、全部答案别名、父文档、image ID、TRAIN 划分均一致。32 主题与开发集、排除父文档无交集。对 480 条预测核查物理视觉数、DS 长度、cache、gold target IDs、固定随机 seed/选集、on/off 原位置及保留索引，未见异常。DS-off 的旧 `native_state_n_vis=0` 仅是已登记的元信息缺陷，实际 full/K256 物理输入按独立字段验证。

CPU `analysis.py` 完整运行成功，四流图/分数和/TopK/27 个 paired contrasts 与两项交互均已计算。latency 的 run-manifest 和 benchmark 源码哈希也匹配最终 freeze；两份结果均 complete。

## 决策

保留 signed joint 的局部正信号，不把它写成“四流无用”；同时，四流未提高已经达到 full 输出的 magnitude V-only teacher，不能说已经找到优于强单流教师的新上限。当前不直接启动多流 scorer 训练，也不把 gold 准确率写成可部署方法效果。后续若要推进，须在新预注册中解决无需答案/无需 backward 的可预测性，并在独立数据上验证；本轮不追加归一化、符号变换、alias 搜索或小模块。

## 产物与复核

- 数据/冻结：`Qwen_vl/outputs/multistream_oracle/data/`、`run_manifest.json`、`gates.json`。
- 原始预测与标签：`Qwen_vl/outputs/multistream_oracle/predictions.json`。
- CPU 分析：`Qwen_vl/outputs/multistream_oracle/analysis.json`。
- 七参考计时：`Qwen_vl/outputs/multistream_oracle/latency.json`；主日志 `experiment_run2.log`。
- 最终 freeze SHA256：`2c8288bd604f0c94f3a6bd27beb7c0b3cae1f79e09f606acaf625ccfcc76f6ff`；预测 SHA256：`91f912cd569c5b7e1caab97f23105395f8e59b58572f94736aa2470a6f16a07c`。

复核命令仅使用 CPU：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dell/miniconda3/envs/qwen3vl_clean/bin/python \
  Qwen_vl/scripts/multistream_oracle/analysis.py
```
