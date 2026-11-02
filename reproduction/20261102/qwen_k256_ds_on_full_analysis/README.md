# 全量 DeepStack-on CPU 终端汇总和评分

`analyze_and_score.py` 会先核对完整 `complete.json`、根会话和容器的退出证明及文件哈希，证明齐备后才打开本轮 observations；逐题核对全部 5349 个问题、历史 selection / 输出 ID / 原回答，统计耗时、首轮语言模型 CUDA 事件、PyTorch allocated/reserved 峰值、输出长度、EOS 和截断。计时以自然生成且不同问题可能生成不同长度，不能直接作为统一长度开关成本对比。

确认完整生成和真实退出，且根任务写入 `root_exit_proof.json` 后，使用作者同环境在无 GPU 设备的容器中运行脚本，设置 `CUDA_VISIBLE_DEVICES=''`。参考已有 `gpu_formula_facility_panel_run02/run_panel.py` 的 `container_command(plan, [分析脚本绝对路径], gpu=False, name=...)` 生成容器命令；需要挂载原项目目录和依赖。此脚本不会启动容器或模型。若本机作者环境可直接运行，亦可使用：

```bash
CUDA_VISIBLE_DEVICES='' /media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python /media/disk2/YZX/research/EADP_v2/reproduction/20261102/qwen_k256_ds_on_full_analysis/analyze_and_score.py
```

脚本调用作者 `ImageVQADataset.evaluate` 和独立 ANLS，逐题验证并保留所有样本及 Excel 的 NA 转换；结果写入本目录新建的 `results/`，包括原生 XLSX、逐题 JSONL、统计及哈希。不会修改运行目录或旧结果；`results/` 已存在时拒绝覆盖，失败后的部分结果须保留并由人检查。

无 GPU、无真实运行输出的合成测试：`python -m unittest discover -s reproduction/20261102/qwen_k256_ds_on_full_analysis -p 'test_*.py'`。
