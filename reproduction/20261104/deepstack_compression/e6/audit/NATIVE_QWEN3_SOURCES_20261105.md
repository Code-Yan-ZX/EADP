# Qwen3-VL 原生 DeepStack 对照：远端来源备忘（2026-11-05）

仅依据作者论文、公开源码和运行时官方文档的静态阅读；未下载代码或权重，未运行本地测试。以下均为候选对照，不是本项目的复现、准确率或性能结论。公平比较应固定原始 8B 权重、输入图像、预处理、提示词、生成设置和计分；不改变原方法的位置语义并先验证实际三路 DeepStack 注入与多步生成。

## 优先级 1：ZipRerank 的 QI-EI 选择器

- [作者论文与附录](https://arxiv.org/html/2605.11864)、[选择器源码](https://github.com/dukesun99/ZipRerank/blob/main/models/qi_early_pruner.py)、[Qwen3-VL 封装](https://github.com/dukesun99/ZipRerank/blob/main/models/qwen3vl_with_qi_early.py)。原研究是文档页重排序；论文明确 Qwen3-VL-8B。QI-EI 本身无需训练权重，以图像视觉嵌入与问题文字嵌入的最大余弦选 top-K，随后按原空间顺序排列。在进入 LLM 前对主视觉流和每路 DeepStack 施加相同索引，并保留原三轴位置。
- [作者仓库](https://github.com/dukesun99/ZipRerank) 有实现，[发布的 ZipRerank 权重](https://huggingface.co/mtri-admin/ZipRerank) 经重排序任务训练；用它与原始 Qwen3-VL 问答比较会改变权重和任务，应区分。选择器可见为独立模块，但在本项目模型环境中可直接调用及其许可/版本/实际行为仍需核验。
- 最小门：固定同 K、同图文和原模型；验证作者选择索引、主流及三路 DS 的逐元素选择、文本不丢失、prefill 真实位置与注入；分别检查 K 全保留端点和 K256 的自然输出、固定 32 步 decode 的位置/rope delta、每层 cache。作者封装默认 `all_before_image` 提取首图之前的文字；`query_only` 采用首图前换行启发式，并非真正解析 `Query:` 标记。两种模式都不提取首图之后的问题；需检查实际 prompt 中选中的 token，不能擅改本项目 prompt 或默认称其提取了问题。静态阅读中，封装返回原始 rope delta，尚未观察到相应删减数修正；是否影响其实际多步生成取决于运行路径，需实测，不能仅凭单 token 排序任务宣称通过。

## 优先级 2：PixelPrune

- [作者论文](https://arxiv.org/html/2604.00886)、[官方源码和 Qwen3-VL-8B 文档评测入口](https://github.com/OPPO-Mente-Lab/PixelPrune)。方法在 ViT **之前**按对应空间 merger 单元的像素块做预测式去重，保留原空间坐标；保留后的序列进入 ViT、主 merger 及 DeepStack mergers。论文覆盖原生 Qwen3-VL 8B 文档任务，默认训练免费，无需额外选择器权重；作者提供的环境与本项目环境不同。
- 像素层 τ=0 的可恢复性不表示 ViT 表征、三路 DS 张量或模型输出与完整输入相同，因为 ViT 的上下文已改变。其保留比例随图而变，也改变 ViT 成本；和 decoder-only 固定 K 方法不天然等算力。先实测禁用/全保留路径与本项目 stock 的身份、像素/网格/位置、各路 DS 和 32 步 decode；启用后的张量应验证索引及内部对齐，不要求与全图特征逐字相同。
- 若身份门通过，单独报告逐图保留数、任务质量及包含预处理、视觉编码、选择和生成的端到端成本，在相同样本上给质量—成本曲线；不要将其动态预算结果直接解释成固定 K 净收益。

## 优先级 3：NVIDIA DART 嵌入层变体

- [NVIDIA 官方文档](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/features/visual-token-pruning.html)、[实现](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/cpp/runtime/preprocess/dartPruner.h)、[DART 原作者仓库](https://github.com/ZichenWen1/DART)。NVIDIA 文档说明其实现于 prefill 前压缩嵌入、DeepStack planes 和 mRoPE，保留原位置，无需新训练权重；文档示例是 Qwen3-VL-4B，8B 和本机可运行性待验。
- E5 已记录：这是输入 embedding 版本；DART 论文的选择位置是 decoder 第 2 层 key。两者应明确分别命名。移植选择规则至本项目时须保留作者预算与排序语义，并通过实际三路注入、位置和多步生成门；不得将移植版称作者原样实现。
- 原 NVIDIA runtime 的 engine 编译、精度及执行后端与本项目 Transformers/BF16 不同。若采用它，必须在同一 runtime 分别测关剪枝与启用剪枝，另外检查原模型端点；跨 runtime 时延不能解释为剪枝方法净收益。

## 对照解释边界

PriorTR 官方 [Qwen3-VL 固定源码](https://github.com/CodeChildCZJ/PriorTR/blob/e442a4c/image/Qwen3-VL/visual_token_pruning/model/vtr_qwen3_vl.py) 已纳入本项目：它属于 decoder 层内剪枝，最终相同 K 不代表每层计算相等。其他候选如 [ET-Prune](https://github.com/Labyrinth0419/ET-Prune) 目前仓库仍将代码标为待发布，其 DeepStack 实现无法据此独立核验；[δ-Vision](https://github.com/declare-lab/delta-Vision) 作者说明评测学生关闭 DeepStack，不满足本轮原生 DeepStack 对照条件。
