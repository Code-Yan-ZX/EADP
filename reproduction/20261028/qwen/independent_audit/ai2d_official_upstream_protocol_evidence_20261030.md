# Qwen3 AI2D 的公开 judge 配置：上游与作者仓库的来源核查（2026-10-30）

## 结论与边界

论文对 Qwen 系列称使用 VLMEvalKit 并遵循其官方配置；论文未指明 AI2D 主表的具体 VLMEvalKit commit、judge 模型/API 端点、是否实际调用 judge、原始预测或当时的 fallback。以下上游快照是论文提交前真实存在的候选配置，**不是论文实际执行配置的证据**。作者公开版本和上游快照在 `chatgpt-0125` 的模型映射上不同；据此不能宣称论文使用某个 judge，也不能替换本机正式分数或判断该差异的得分影响。

## 一手来源与可核对内容

1. [EADP 论文 arXiv 版 §5.1、表4](https://arxiv.org/html/2607.02484v1)（2026-07-02）：Qwen 使用 VLMEvalKit，论文写明遵循其官方配置，设置与指标使用默认；Qwen3-VL-8B AI2D K256 为 75.7。AI2D 的 judge 模型、端点、源码版本和运行参数未给出。正式会议版 [ECCV PDF](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/1977.pdf) 第11页同样有该配置说明，第13页有 Qwen3 表。论文对 GPT-4 的具体提及属于 MM-Vet，不能拿来指认 AI2D judge。
2. 上游 VLMEvalKit commit [`401916c00a4769884aeca865a77674abe0c62a26`](https://github.com/open-compass/VLMEvalKit/commit/401916c00a4769884aeca865a77674abe0c62a26)：[官方 commit patch](https://github.com/open-compass/VLMEvalKit/commit/401916c00a4769884aeca865a77674abe0c62a26.patch) 记作者日期 2026-03-25 16:50:53 +0800，即 08:50:53 UTC，早于 arXiv 提交；本项目先前 `qwen/import_overlay/official_helper_history.json` 也记载该日期，但此前是为 MMMU helper 历史调查而收集，不能证明 EADP 当时使用该 commit。
3. 该上游快照的 [run.py](https://raw.githubusercontent.com/open-compass/VLMEvalKit/401916c00a4769884aeca865a77674abe0c62a26/run.py) 行334–360 对 AI2D 所属一般 MCQ 类型、未显式传 `--judge` 时选择别名 `chatgpt-0125`；同快照 [judge_util.py](https://raw.githubusercontent.com/open-compass/VLMEvalKit/401916c00a4769884aeca865a77674abe0c62a26/vlmeval/dataset/utils/judge_util.py) 行11–37 在 `LOCAL_LLM` 未设置时把该别名映射为 `gpt-3.5-turbo-0125`，并使用 `OpenAIWrapper`。`LOCAL_LLM` 可覆盖模型标识，`--judge` 也可改变选择；模型标识不是服务端点、实际服务版本或成功调用的证明。
4. 作者公开 commit [`e1a08801461c181871b9ff1cb0803b8e9966f083` judge_util.py](https://github.com/SJTU-DeepVisionLab/EADP/blob/e1a08801461c181871b9ff1cb0803b8e9966f083/Qwen_vl/VLMEvalKit/vlmeval/dataset/utils/judge_util.py) 在 `LOCAL_LLM` 未设置时将同一别名映射为 `deepseek/deepseek-chat`。其代码注释提到 OpenRouter，但实际端点由外部配置决定，不能只从注释、别名或模型标识推定请求送达何处。作者公开 shell 中没有明确指定 `--judge`，仍不能排除论文历史运行的其他命令或环境。
5. 将上游固定快照 [multiple_choice.py](https://raw.githubusercontent.com/open-compass/VLMEvalKit/401916c00a4769884aeca865a77674abe0c62a26/vlmeval/dataset/utils/multiple_choice.py) 与作者本地固定源码仅 CPU 解析成 AST，忽略行列位置：`build_prompt` 和 `prefetch_answer` 函数 AST 完全相同；`extract_answer_from_item` AST 不同，逐函数 diff 中作者额外一行 `logger = get_logger('Evaluation')`，其余差异为空格。此检查不等于整个评分栈、动态依赖或历史运行完全相同。上游文件 SHA256 `8121d23f388e7766ad239980f381339916a744cffcf429a62d58ba4de5a7a176`；作者文件 SHA256 `460b75c5fee06aa10f4037b2694e74708978820ed9291933b703fca39d7b156a`。

## 已有审计与实际本机结果

本项目 `reproduction/20261028/qwen/independent_audit/author_ai2d_judge_source_audit.json` 已覆盖作者发布源码的别名映射、默认后端与端点边界、工作性测试失败后的精确匹配 fallback、先用 `can_infer`、通用 judge 提示、以及论文未指明 AI2D judge；也已记录论文默认设置。此前该专项记录未比较论文的“官方配置”措辞与提交前上游快照的具体映射；既有上游 commit 日期记录是另一项 helper 调查，不是历史实际 judge 身份。

对于作者发布的代码，judge 建立后若 working() 失败则使用 exact matching；先由 `can_infer` 提取，无法提取且无可用 judge 时返回 Z；可用 judge 的提取经历重试，耗尽后的随机选项路径也记录在既有审计中。不能把这些代码路径的存在当作论文历史上哪条路径被采用的证据。

已完成的本机公开作者 FA2 K256 实验按 API 连接失败后的源码 exact-matching fallback 得到 2296/3088 = 74.352331606%，与论文 75.7 相差 −1.347668394 个百分点。该完整结果、3088 分母和未解析项均保留；以上上游配置差异只是候选协议证据，不授权 API 调用、事后解析或重评分。

此次只读访问公开源代码、论文和已有审计；没有调用 judge API、评分、加载模型、使用 GPU 或读取进行中的实验输出。未改原结果、对照表和评分工件。
