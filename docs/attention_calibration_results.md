# 固定选集 attention 校准：原生 Qwen3-VL 探索实验

日期：2026-11-09。分支 `codex/attention-calibration`，基于 `a724122d`。
预注册：`docs/attention_calibration_prereg.md`；主数据和实现均在首条主样本预测之前冻结。

## 结论

**不推进本轮相似度加权的有效簇大小公式。** 在32个独立源文档的真实数值问答上，
该方案19/32，普通簇大小补偿21/32；相对普通补偿0修复、2破坏，净差−6.25个百分点，
源文档配对bootstrap 95% CI `[−15.625, 0]`个百分点。小样本不能证明总体必然下降，
但按照冻结规则没有继续调指数或训练此公式的依据。

普通簇大小补偿和RESTORE式prefill校准均为21/32，相比相同显式mask的零偏置对照
20/32均为2修复、1破坏，净增3.125个百分点，95% CI `[−6.25, +12.5]`。这是未决的
局部正向点估计，不能称为可靠提升。RESTORE-style相对普通count净增0问（1修复、1破坏）；
continuous方案与count的32条生成文本和token序列全部相同，未显示距离项的独立收益。

显式mask会改变SDPA计算路径：VisionZip的native baseline为19/32，零偏置mask对照
20/32（1修复、0破坏）；B2分别18/32与17/32（0修复、1破坏）。因此不能把相对native
baseline的差额全部归因于校准。主要方法对照使用各自的maskzero。

## 设置与可复查性

- 数据：从DocVQA TRAIN固定32个新父文档、每页一条已标注数值QA，另1个独立开发页。
  排除历史fit、VAL和此前CF/binding/LayoutLite/CatchUp面板父文档及已用图像。主面板
  不根据模型正确性、截断或方法分歧筛选；32个问题均进入主分母，所有输出均EOS停止，
  无截断。保留完整答案别名和原图/数据源哈希。这是项目内探索面板，不是官方测试结果
  或预训练未见文档证明。
- Qwen3-VL-8B，A40，bf16，Transformers 4.57.6 / SDPA，1024输入视觉token、
  K256、原生3D interleaved mRoPE和三条DeepStack；greedy，上限32生成token。
  R-res512以512方形图产生256个原生token，视觉编码工作量与其他臂不同。
- 主carrier为固定VisionZip原生适配：D205/C51，相同anchors、聚合后的V/3DS及位置；
  不是作者完整原版Qwen3复现，也没有使用PPE/字段oracle。次carrier为hard-pruning的
  EADP scoring + block8（B2）。各carrier的所有对照共享选集和输入流。
- `count`给视觉key加`log(n)`；`effective`用
  `n_eff(a)=1+Σ_i clamp(cos(key_i,key_a),0,1)^8`代替n；`bias_const`在视觉key加
  `log(4)`。RESTORE式适配使用`log(n)+log(2−D_mRoPE)`、原始Qwen3轴交错及全key距离项。
  官方Qwen2.5代码的prefill处理对应这里的`restore_prefill`；`restore_continuous`
  额外在decode使用该公式。所有参数事前固定，无模型训练或test-time teacher。此为
  校准算子适配，不是完整[RESTORE](https://arxiv.org/html/2606.01711v2)复现。
- 统计为原始答案别名数值精确匹配，DocVQA官方ANLS及字面完全匹配为辅助指标；
  10,000次共享源文档配对bootstrap，seed 20261109。所有28项预设对照保存在
  `Qwen_vl/outputs/attention_calibration/analysis.json`。

## 真实QA结果

| 固定carrier及方法 | 数值精确正确 / 32 | ANLS | 严格字面匹配 / 32 |
| --- | ---: | ---: | ---: |
| Full N1024 | 30 | 0.9705 | 29 |
| R-res512，原生K256 | 20 | 0.8043 | 20 |
| VisionZip native | 19 | 0.7961 | 18 |
| VisionZip maskzero | 20 | 0.8086 | 19 |
| VisionZip count | 21 | 0.8352 | 20 |
| VisionZip effective | 19 | 0.7961 | 18 |
| VisionZip constant | 19 | 0.7777 | 18 |
| VisionZip RESTORE prefill | 21 | 0.8260 | 20 |
| VisionZip RESTORE continuous | 21 | 0.8352 | 20 |
| B2 native | 18 | 0.6458 | 18 |
| B2 maskzero | 17 | 0.6302 | 17 |
| B2 constant | 17 | 0.6877 | 17 |
| B2 RESTORE prefill / continuous | 16 / 16 | 0.6535 / 0.6535 | 16 / 16 |

`effective`相对`bias_const`为1修复、1破坏，净差0，CI `[−9.375,+9.375]`个百分点，
也没有通过同时优于count和constant的预设条件。B2 constant相对maskzero为3修复、
3破坏；B2 RESTORE prefill与continuous均为0修复、1破坏。没有证据表明校准在该硬剪枝
carrier上稳定获益。

相对R-res，count为6修复、5破坏，净增1问，CI `[−15.625,+25]`个百分点；样本不足以
判断优劣。虽然本面板上count名义上多1问，但R-res的质量与计算量共同构成必要强参考。

`effective`并非数值恒等干预：全部样本`n_eff`总和均值333.13（范围313.99–370.24），
而count总和固定1024；对count方法有4/32条生成文本不同。但这版大幅减弱簇补偿并未带来
观察到的修复，不说明所有内容感知校准无效，也不能仅凭分数低将失败归因于权重过弱。

## 端到端计时

独立开发页、batch=1、同A40/SDPA/bf16，模型加载除外；每臂3次预热、10个轮转配对block。
每次重新图像预处理、视觉编码、选择、准备bias、prefill并做首token选择；固定16-token
生成使用ignore-EOS，包含首token。单位毫秒，为10次中位数，完整单次记录见`latency.json`。

| 方法 | TTFT | 端到端16-token |
| --- | ---: | ---: |
| Full | 403.12 | 936.45 |
| R-res512 | 137.30 | 670.93 |
| VisionZip native | 304.83 | 833.37 |
| VisionZip maskzero | 328.35 | 1330.84 |
| VisionZip count | 330.64 | 1338.15 |
| VisionZip effective | 329.76 | 1340.90 |
| VisionZip RESTORE prefill / continuous | 334.15 / 335.13 | 1304.00 / 1429.70 |
| B2 native / maskzero | 285.34 / 309.20 | 825.43 / 1315.89 |

同block配对，count相对maskzero的TTFT中位差为+0.30ms；相对VisionZip native和R-res
分别为+24.25ms、+192.38ms。当前逐层Python hook、显式mask及同步检查使decode成本明显
增加；该测量反映本地未优化实现，不能推断高性能fused kernel必然有相同开销，也不能
声称本版部署加速。单页10次不是多样本的延迟分布。

## 门禁、偏差与产物

最终134项CPU/GPU检查全通过：stock keep-all prefill及四步forced decode一致；按原生
interleaved mRoPE验证phase、GPU数值算子、causal mask、GQA、K256、3DS、cache和全部12个
carrier实际接线。零mask在开发页的native logits不是bit-identical（VisionZip最大差0.375，
B2最大差0.453125，所测首token/强制步argmax未改变），对应单独maskzero主对照。

第一次启动正式入口时误导入VLMEvalKit同名`run.py`，在加载模型及写入冻结manifest之前
退出；日志保留为`experiment.log`。重命名为`calibration_run.py`后重跑全部134项门禁，
并在主QA前写入run manifest，完整成功运行日志为`experiment_run2.log`。首轮及扩展门禁
attempt均保留；没有使用这次入口错误前的任何主QA输出。

代码及复现入口：`Qwen_vl/scripts/attention_calibration/`；主要数据、输入/代码冻结、
逐题预测、分析和计时分别位于`Qwen_vl/outputs/attention_calibration/`的`data/`、
`run_manifest.json`、`predictions.json`、`analysis.json`、`latency.json`。结果仅支持此
模型、数据、配置和实现下的探索性判断；小N、多个比较、未优化hook和单页计时均限制
推广。另一个多流选集oracle单独预注册、使用新样本，不用本面板调参。
