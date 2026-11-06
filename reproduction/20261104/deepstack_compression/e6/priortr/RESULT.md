# PriorTR E6 本机可行性准备

固定公开源码 `e5/PriorTR` 的 HEAD 为 `e442a4c14af1b333067a6b8310c9111cad922033`，Git 工作树只读检查干净。未安装、更改任何现有环境或 clone，未加载 GPU、权重及 benchmark 数据。

## 代码事实和 CPU 证据

- Qwen3-VL 默认 `strategy=priortr`、`prune_layer=3`、`query_aggregation=auto`（解析为 `question`）、`head_aggregation=mean`；默认 `keep_ratio=0.1111` 并非 K256。计划的公平 K256 应明确指定 `keep_tokens=256`（优先于比例），并另外保留原默认配置对照。
- `prune_layer=3` 的打分层索引是 2：第 1–3 个 decoder 层先处理完整输入，随后选择 256 个图像 token，再进入其余层；所以与 EADP 在首层前压缩到 K256 的模型，每层 token 数、KV 路径和总成本不同。需要现场记录各层序列、KV、额外计算、端到端 TTFT 与显存；最终 K 相等不是等成本。
- 策略计算 Q 为 vision_end 位置对视觉 token 的注意力；`question` 的 P 实际取 `img_end:-1` 的后续 token 注意力平均（包括 vision_end 和可能的模板 token，不应未经验证称为纯问题词）。以 P·log(P/Q) 排序，保留原始空间顺序。
- 打分层手工创建并 softmax 完整注意力矩阵，同时常规 attention 仍执行；应计入真实推理成本。作者例子使用 SDPA；E1 使用 FA2，后端或 Transformers 差异必须单列，不能直接归因为算法。应先比较相同输入/权重、预处理、生成和评分，并做两套环境的原生未剪枝端点验证。
- 源码在剪枝后用相同索引截取 hidden、原始三轴位置、mask 和已计算层 KV；DS 第 0、1 路已注入，第 2 路及视觉 mask 在第 3 层的剪枝后、注入前同步。decode 源码以累计删除量修正位置。以上是源码行为，非 full model 已验证。
- `monitor_cpu001/` 是 CUDA 禁用、无模型权重的合成 CPU 探针，实际加载固定版本 config/strategy/DS handler 源码：1024→256 的公式、真实 top-k 空间排序，以及第 2 路 DS 未注入前同步和 mask 核对通过；进程 exit 0、组空。其通过不代表完整模型导入、mRoPE/KV/decode、正确率或性能通过。
- `monitor_import001/` 保留作者环境 Python 3.10、torch 2.8、transformers 4.57.1 下真实完整包导入失败：缺失 `BaseModelOutputWithDeepstackFeatures`，退出 1、组空。E5 在另一个 4.57.6 环境观察到同类失败；不能推断官方指定环境必然失败。

## 独立环境计划（尚未执行）

README 明确要求 transformers Git commit `f8f2834e1a`（标示 5.2.0.dev0）、Python 3.10，以及项目自带 `qwen3/` 对 `transformers.models.qwen3_vl` 的替换。环境记录列出 torch `2.10.0+cu128`、torchvision `0.25.0+cu128`、accelerate `1.13.0`、`qwen-vl-utils==0.0.14`、decord `0.6.0`、spacy `3.8.11`。上游环境记录注明不是一条命令可重建的锁文件；其他依赖需按其完整 `environment.yml` 和实际 resolver 保存冻结结果，不能宣称完全复刻。

拟采用与现有环境互不覆盖的独立 prefix `.../e6/priortr/env`，并先将固定 Qwen3-VL 子树复制到 `.../e6/priortr/source/`，核其文件哈希；避免上游 `setup.py develop` 对现有 transformers 的目录重命名/链接，也避免在原 clone 写 egg-info。建议安装顺序：

1. 独立 prefix 内创建 Python 3.10.20 与 pip；安装固定 torch 2.10.0+cu128 / torchvision 0.25.0+cu128 官方 cu128 wheels。
2. 在独立目录获取 Transformers `f8f2834e1a`，先解析并记录其完整 commit 和源码哈希，再安装该 checkout；同时记录所有 resolved 包的版本和 hash。绝不在 qwen_author 或 clean 环境执行上游 setup。
3. 安装项目要求的固定辅助依赖，复制项目运行 `python setup.py develop` 仅在新 prefix 进行；逐项检查模型实际导入路径和 symlink 指向新 source、无回指旧环境。lmms-eval 属可选基准入口，需单独冻结其提交和依赖；不为了单图本地 smoke 无条件安装整套评测栈。
4. 禁用 CUDA 做真实包导入和随机小 Qwen3 模型 forward、各层长度/DS/mRoPE/cache及 32 步固定 decode 检查，再进行受控 GPU smoke；单独跑官方默认，再做预先冻结的相同输入、最终 K256 对照。训练权重复用已验 SHA 的本地快照，不重复下载。

成本仅为规划：当前机器观测 qwen_author prefix 约 2.0 GiB、clean prefix 约 14 GiB，目标盘可用约 1.1 TiB（均为本机当前 `du`/`df`，硬链接、缓存和新依赖会影响真实空间）；独立环境实际下载量、构建时间、FA2 与 pinned Transformers 的兼容性尚未测。首次建立应记录真实下载量、占用、耗时及失败日志，再决定是否在三天预算内继续。没有使用外部论文数值。
