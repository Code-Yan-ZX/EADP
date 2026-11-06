# E1 设计准备：位置 × DeepStack

状态：CPU 设计；没有 E1 GPU 运行、准确率或通过结论。此文不修改 E0，也不代表身份门已通过。先完成 16 个 ANCHOR 的 E0 全部检查及独立终端验证，才能运行 E1 正式面板；单题 smoke 不够。

## 固定因素和不同因素

同一实例和 Qwen3-VL-8B checkpoint、1024×1024、BF16/FA2、greedy、batch 1、max_new_tokens=2048；源 recipe 在 `reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json`，EADP 运行实际确认 K256、α=0.5、β=2、q=0.2（q 需从实例与源码核实，不能单由方案假定）。每题记录其原空间实际 greedy 顺序、排序 ID 和特征重构哈希，四臂必须同一集合。可以每臂重新计算后精确比对；如缓存选点则明确缓存边界，E1 的时间不充作 E4 真实性能。

| 臂 | prefill 位置 | decode | DeepStack |
|---|---|---|---|
| P1-D0 | 压缩序列 1D 连续位置复制到三轴 | 旧连续序列位置 | 无 DS 参数 |
| P1-D1 | 与 P1-D0 相同 | 同左 | 三路实际所选 DS |
| P3-D0 | 在未剪枝完整 input_ids/grid 上算三轴 mRoPE，再按保留序列 ID gather | 原图逻辑位置，验证 delta 和实际 cache | 无 DS 参数 |
| P3-D1 | 与 P3-D0 相同 | 同左 | 三路实际所选 DS |

当前 `e0_identity/adapter.py` 固定 P3，`get_rope_index` 从原始输入求坐标，prefill hook 改写 `position_ids`，并设置 `rope_deltas = full_delta + (1024-K)`。当前 off 不显式清空已有 kwargs，因此 E1 实现须保证 D0 `visual_pos_masks` 和 `deepstack_visual_embeds` 均为 `None`；D1 构造长度等于压缩序列、恰好 K256 个 true 的 mask，三层每路行按真实排序 ID gather。不得对完整原图的 spatial feature 重新排序或在 DS 改变选择。

P1 应采用相同 hook 机制明确提供压缩序列的连续位置与对应 decode delta，而非把 P3 的 delta 继续沿用；其预期须锚定同一 author 栈旧 `inputs_embeds`+compact attention_mask 路径。检查 `qwen_author/.../modeling_qwen3_vl.py:1016-1020,1177-1221`：无原始 `input_ids` 时前缀 `position_ids` 来自 attention mask 的 cumsum，单 batch 无 padding 时 P1 三轴均为 0..L-1，delta 为 0；生成时旧路径 cache position 和 delta 如何传递要在运行时实证，不能只依公式宣称正确。即使两因素分别可操作，也需验证 P1-D0 在固定16锚的输入、选择、输出与旧实现重现范围内一致；历史精度值只有在完整输入/生成身份闭合时才复用。P3 的 K256 则不称原生 FULL 身份，只验证其为原始坐标取子集。

## 面板与可追溯输入

`data_manifest/panel_manifest.jsonl` **预期**提供 `task,qid,group_id,split`；`full_manifest.jsonl` 提供原问题与图像索引关系，`images.jsonl` 提供源图指纹。当前构建和独立验证结束前这些文件不得视作已冻结，先核结果完整、原始 TSV 和 builder/verifier 哈希、三任务 DEV 数、组不交叉和缺额；任何构建中断/不完整不作为清单。输出为完整 panel（计划每任务 DEV 128 组，总 384 个），按任务和冻结排序使用，每组一题。

接入需按 `(task,qid)` 与各 TSV 唯一连接，使用作者/VLMEvalKit对应 task 的 prompt 构造，不把 DocVQA `_build_vqa_prompt` 默认为 TextVQA/AI2D 的适配。冻结每题图像文件、文件与解码 RGB hashes、question/prompt/messages、task、group 与 processor `input_ids/attention_mask/pixel_values/image_grid_thw` 指纹；独立复核运行时实际 processor 输入与冻结版本相同。图像文件可能由缓存重编码，须分别核 embedded 图与实际文件的解码像素，不能把相同文件名当身份。生成阶段无需读取 GT；对齐所有完整 arm 并验收后，在独立进程按任务固定规则评分，AI2D 原生 fallback 与其他 judge 另列。

## 验证与产物

1. CPU 单元验证 P1/P3 四臂配置矩阵、已知位置张量 gather、完整非视觉 token 保留、DS/no-DS 精确区分、输入 ID 和每题四臂完整配对；故意注入错序、DS 少一行和缺臂要拒绝。分别冻结代码、模型权重、processor、源输入、manifest、环境及 recipe。
2. E0 完整门过后先做固定锚 smoke：P1-D0 对同 recipe 旧路径、P3 两臂对 E0；固定前缀至少 32 decode 步验证位置/实际每层 KV cache 长度，首次 logits 和输出身份按预设规则；逐题核四臂实际选择与特征对齐。同等条件变化限位置/DS，不因准确率选择 P1。
3. smoke 合格才执行 DEV 四臂，逐题写 qid、组、原输出 token/回答、EOS/截断、顺序及空间 ID、原始位置/实际输入与各路注入摘要、wall-clock 和失败信息。持久化 manifest SHA 与原始记录；完整结束、独立核验身份/题数和退出链之前不做前缀评分或调参。
4. 完整后按图聚类 paired bootstrap，报告每任务 P3/P1 内 DS 效应、每 DS 条件内位置效应与差中差。先从实际 smoke 和长尾测时估算成本，再安排运行；失败、中断、缺题保持未完成，恢复前确认身份，不得静默跳过或重复 qid。

当前尚缺 E1 单独实现、按三任务生成输入验证、输出验收与评分；不能将现有 E0/历史 DS 测试改名即视为 E1。
