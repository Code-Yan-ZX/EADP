# K256 DeepStack 配对基线：逐题观察归档

这里保存 2026-10-30 真实 DocVQA 全量开关对照的两份有序观察记录。原文件分别约 110 MiB 和 115 MiB，超出普通 GitHub 文件限制；压缩后的两个文件保留了原始字节，`hashes.json` 中记录原始与压缩文件的 SHA-256。已实际解压并校验两份原始 SHA-256 与本机文件相同。

在仓库根目录，可按需还原（如果目标文件已经存在，请先比对其哈希，避免覆盖）：

```bash
mkdir -p reproduction/20261028/qwen/doc_full_k256_ds_pair/execution
gzip -dc reproduction/20261102/qwen_k256_ds_baseline_archive/ds_off.ordered.observations.jsonl.gz > reproduction/20261028/qwen/doc_full_k256_ds_pair/execution/ds_off.ordered.observations.jsonl
gzip -dc reproduction/20261102/qwen_k256_ds_baseline_archive/ds_on.ordered.observations.jsonl.gz > reproduction/20261028/qwen/doc_full_k256_ds_pair/execution/ds_on.ordered.observations.jsonl
```

这是逐题观察数据，不包含视觉塔高维张量缓存。模型权重、数据集、作者外部源码和其它大型缓存不包含在本次 Git 提交内。有关实验设置与结果，请看 `../../20261028/qwen/doc_full_k256_ds_pair/RESULT.md` 和 `../qwen_k256_ds_on_full_perf/RESULT.md`。
