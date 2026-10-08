# NeXT TextVQA 模型、输入与生成身份核验

核验对象是2026年10月8日严格官方TextVQA入口：NeXT EADP K32精确51.996%，论文54.2%；K64为55.352%，论文57.0%；K128为57.958%，论文59.2%。本核验只读源代码和产物，无GPU模型启动，无生产代码、队列、参数、预测修改，也未重复全部评分。

现有证据未发现本机读入错误checkpoint类型、错误题面/裁剪数、修改baseline数学或上下文截断。剩余可直接检验的是CUDA执行与官方loader的实际调度是否影响相同数值输入。不能由本结果保证会追平论文，也不能把跨论文差值自动叠加到我们的方法。

## 已确认

- `llava_arch.py`、`builder.py`、`clip_encoder.py`、`mm_utils.py`、`conversation.py`、`llava_llama.py`和官方`model_vqa_loader.py`，当前工作区均与官方clone逐字相同；SHA见`identity_report.json`。
- 模型配置是LLaVA v1.6 Vicuna 7B、CLIP336、vision层-2、AnyRes、`spatial_unpad`。本地和[公开HF revision](https://huggingface.co/api/models/liuhaotian/llava-v1.6-vicuna-7b/revision/deae57a8c0ccb0da4c2661cc1891cc9d06503d11?blobs=true)唯一JSON语义差异是`model_type: llava → llava_llama`，匹配官方EADP注册类名；实际直接构造`LlavaLlamaForCausalLM`，其余模型字段没有差异。
- tokenizer模型/配置、special tokens、generation config、safetensors index、训练元信息与HF缓存字节一致。3个大权重文件大小和safetensors头部一致，公开API的LFS SHA已保存，但本地权重payload未全量hash，以免同时读取14GB与急迫队列争抢I/O；不能将size/header一致写成权重全量SHA一致。
- 5000条TextVQA OCR题面文件与LLaVA `eval.zip`内文件逐字相同。5000题含3166张不同图片；题目身份按完整prompt区分，未按image_id误去重。
- 复用先前seed20261008、按CLIP文本长度分层且不看答案/预测选择的16题。当前wrapper与官方`CustomDataset + collate_fn`的LLM input_ids、float32 RGB图像张量、float16图像张量、原图尺寸全部逐元素相同，每图张量都是`[1,5,3,336,336]`。输入图片逐项SHA保存。
- Guidance删除单词/短语回答指令后缀，LLM仍保留完整OCR和该指令；与官方入口一致。greedy、temperature=0、beams=1、top_p=None、max_new_tokens=128、use_cache=True，与官方默认/运行协议一致。EADP臂没有安装AnchorZip。
- 全5000题最大LLM input_ids长度480；即使FULL使用2880视觉token，也没有一题超过4096上下文上限。K32同样没有上下文截断，不能据此解释掉点。
- 完整runtime记录中K32实际保留156–159，K64保留316–319，少1–4个来自官方配额向下取整；不能把参数32误写成NeXT总视觉token32。

## 与论文比较仍有边界

官方发布代码固定672×672局部画布，生成1全局+4局部crop。它在`prepare_inputs_labels_for_multimodal`里无条件去掉merge类型中的`_unpad`，包括FULL：因此本轮FULL实际是2880视觉token、不去padding、不添加newline，而不是vanilla LLaVA NeXT的`spatial_unpad`。这是官方发布代码与vanilla代码的区别，当前工作区与官方发布代码一致，不能归咎本机；论文FULL是否外引vanilla结果仍需作者配置证据。此前本机FULL60.24与论文60.3本来接近，也不支持仅凭这个代码区别断言2.2点差距来源。

官方loader用DataLoader 4 workers，CPU预处理后`non_blocking=True`转GPU；当前wrapper串行PIL预处理、阻塞`half().cuda()`。CPU数值输入已验证相同，但这不保证两个执行入口在存在跨CUDA stream依赖时语义相同。尤其CLIP forward新建流，以及NeXT入口`torch.cat`在caller stream生成五crop批次，需由独立执行对照验证。此处仅记录候选，未证明因果。

## 最小后续对照

固定现有16题面板、K32及alpha=.5/beta=2，保持模型/LLM prompt/图像/128 greedy不变：分别比较当前wrapper、原版官方loader、以及只补显式caller→CLIP消费者流依赖的独立内存控制。记录输入/特征/保留集合/实际配额和最终回答，固定顺序至少重复2遍。先判断相同输入是否稳定，再决定完整5000题是否需要受控重跑。面板不用于推算benchmark准确率，阴性结果也应保留。

可复现CPU命令（会重新生成本诊断目录报告；只读取大权重头部）：

```bash
nice -n 15 ionice -c 3 /home/dell/miniconda3/envs/llava_pruner/bin/python /media/disk2/YZX/research/EADP_amp/Qwen_vl/outputs/audit_followup_20261008/next_gap_diagnosis_20261008/input_identity/audit_input_identity.py
```

本次脚本exit=0，`cuda_initialized=false`，16题所有数值身份核验通过。没有查明一个已经证实能补回2.204点的模型/输入错误。
