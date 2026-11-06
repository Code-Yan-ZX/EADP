# E5 固定版本 Qwen3-VL 源码与 CPU 诊断

本次只读上游源码、不修改 clone；没有 GPU、权重加载或 benchmark 生成。不能用下列 CPU 测试声称论文结果或完整运行已复现。

## PriorTR

- 来源：`PriorTR` 固定 commit `e442a4c14af1b333067a6b8310c9111cad922033`。`image/Qwen3-VL/README.md` 指定 transformers Git commit `f8f2834e1a`，安装需 `setup.py develop` 替换 transformers 模型路径；这一独立环境本次未建立。
- `visual_token_pruning/model/prunable_qwen3_vl.py:490-569` 在层内选择后，依相同完整序列索引裁剪 hidden、三轴 position、text position、mask、缓存及 RoPE，并同步后续 DS；`deepstack_handler.py:105-118, 179-190` 以视觉局部 ID 裁剪尚未注入的 DS、以完整 ID 裁剪 visual mask。默认选择器将 ID 排序（`strategy/base.py:74-129`）。这是源码行为描述，不是全路径正确性的证明。
- [priortr_cpu_probe.log](priortr_cpu_probe.log)：从原文件单独载入 DS handler，CPU 合成三路、6 视觉位置，在第 1 层模拟选择 0/2/5，确认已注入 DS 保持、后续两路及 mask 按预期同步；不覆盖 decode/cache/mRoPE。
- [priortr_clean_import.log](priortr_clean_import.log)：本机 qwen3vl_clean（torch 2.10.0、transformers 4.57.6）直接导入包失败，缺少 `BaseModelOutputWithDeepstackFeatures`。未在作者固定环境尝试，因此不判定该方法不可运行。首次错误日志路径导致命令在 Python 启动前失败，见 `priortr_path_failure.log`。

## Hi-Lo Prune

- 来源：`Hi-Lo_Prune` 固定 commit `29265e3b13ef89ecfd2cbd8803f1678a2ce1bdb6`。README 说明这是初始 Qwen3-VL 图像入口，原实验使用 torch 2.6、transformers commit `e2e8dbed13c6a8455fd85c15c9fa91c99d609010`。原始论文、历史实验与本次发布路径须区分。
- 模型公开源码 `modeling_hilo_qwen3_vl.py:1412-1550` 对前三层提供 DS 注入，剪枝时按 keep ID 裁剪文本与三轴位置；公开当前路径在行 1513 引用 `top_index_1`，全仓库没有该名字的赋值。
- [hilo_cpu_probe.log](hilo_cpu_probe.log)：本机 qwen3vl_clean 的 torch 2.10.0 / transformers 4.57.6，随机极小 4 层 text model、6 视觉 token、三路 DS、eager CPU；无选择控制执行成功，启用选择到第 2 层的实际公开 forward 路径触发 `NameError: top_index_1 is not defined`。这是针对该固定提交和此 CPU 运行的失败证据，不是完整大模型验证，也不能断定作者历史论文结果。程序捕获异常保留 traceback，因此脚本退出码 0 不代表选择路径成功。
- [hilo_unit_tests.log](hilo_unit_tests.log)：仓库 5 个 CPU selection/patch 测试通过，但它们不覆盖 model forward。`hilo_author_import.log` 记录 torch 2.8.0 / transformers 4.57.1 导入时 `check_model_inputs` 签名不兼容；非作者指定环境。`hilo_clean_import.log` 在 4.57.6 导入成功。
- 位置裁剪后的 mask、剩余 DS 与 cache 行为尚无成功运行验证；此固定版不应直接作为生成公平对照。若未来修复，应另建可审计补丁与独立版本，先验证剪枝/DS/mRoPE/cache 和输出，再记录偏离公开提交之处。
