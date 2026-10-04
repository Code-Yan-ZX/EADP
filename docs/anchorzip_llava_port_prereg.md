# AnchorZip LLaVA 移植预注册（P3，2026-10-04）

> 状态：移植未开始执行。本文档固化已核验事实与映射决策，防止移植过程中临场改动。
> 主方法冻结定义不变：**RTG → 原 Facility → main-only Completion(λ=0.25)**。
> 只移植冻结数学操作；不训练投影、不按测试分数调超参。

## 1. 已核验事实

### 1.1 论文表 1/表 2（arXiv 2607.02484v1，WebFetch 核验 2026-10-04）
- 表 1 = LLaVA-1.5-7B；表 2 = LLaVA-NeXT-7B。列（两表相同）：
  `VQAv2 GQA VizWiz SQA_IMG VQAText(=TextVQA) POPE MME MMBEN MMBCN MMVet Avg.`
- **Avg = 9 benchmark 均值，排除 VizWiz**（caption 原文）。
- LLaVA-1.5：576 token → K=128（论文另有 64/32 档；本轮按用户指令先 K128）。
- LLaVA-NeXT：672×672 输入 → **2880 token/图**；AnyRes 五 crop
  （1 global p=0 + 4 local p=1..4，B=5）→ **总 K=640**（另有 320/160）。
  Dynamic Quota Allocation 在官方 `greed_select` 的 joint 模式 + `token_quotas`。

### 1.2 官方 LLaVA 代码（本仓库 LLaVA/，只读参考）
- 接入点：`llava/model/llava_arch.py::encode_images`（返回 post-projector
  features + bool index_masks），mask 在
  `prepare_inputs_labels_for_multimodal` 中按 `mm_patch_merge_type` 消费
  （v1.5=flat；NeXT=spatial unpad + image_newline，newline token 的 mask 置 1）。
- EADP 评分管线（与 Qwen rtg_importance "verbatim official" 同族）：
  sim_cross(text_mean, text_seq, image_embeds[pre-projector]) → entropy
  filter(T=100, keep 0.2) → local aggregation(negative_entropy, M_temp=0.01)
  → fused = 0.5·global + 0.5·local → minmax → spatial_smoothing(k3,σ1) → β。
- facility：`sim_visual(image_features)`（**post-projector**）+
  `greed_select(importance, sim_matrix, K, strategy='importance_based')`。

### 1.3 环境与权重
- `qwen3vl_clean` 环境 **import llava 成功**（sys.path 插入 EADP_amp/LLaVA），
  无需建 pruner env（待 torch/flash-attn 兼容性 smoke 验证）。
- LLaVA-1.5-7B 权重（原版格式，含 mm_projector.bin）：
  `/media/disk2/YZX/doct/FastV/llava-v1.5-7b` ✓
- LLaVA-NeXT-7B = `liuhaotian/llava-v1.6-vicuna-7b`，下载中（HF cache，
  2026-10-04 启动，~16GB）。
- 评测数据：LLaVA EVAL.md 流程，`playground/data/eval`（VizWiz 本地 val 脚本
  `vizwiz_val.sh` 存在）。MM-Vet judge 不可用时该列标"未完成"；
  VQAv2 testdev 需官方服务器，本地仅留存预测并标注。

## 2. 映射决策（冻结，移植时不得更改）

1. **RTG importance**：仅替换官方步骤 (2.1)+(2.2)+(2.3)——熵过滤与
   local aggregation 换成 RTG 文本权重（`rtg_common.text_weight`，T=100，
   负 cosine 原符号，w=(A*w).sum 后 fused=0.5·g+0.5·local）；输入 A/g 取自
   LLaVA `sim_cross` 的 pre-projector 特征（与 Qwen 用 raw V 一致）；
   (2.4) minmax、(2.5) smoothing、(2.6) β=2 与 facility 全部原样保留。
2. **Facility**：官方 `sim_visual`（post-projector）+ `greed_select`，
   NeXT 用 joint 模式与官方 quota 分配（每 crop 配额），**不重写选择器**。
3. **Completion**：作用于 **post-projector 特征流**（LLaVA 进入 LLM 的
   唯一视觉流；Qwen 的 main-stream 对应物）。逐图（NeXT 逐 crop，
   split_sizes 语义）执行冻结算子：
   `compute_assignment(V_i, keep_i)`（cos-argmax，gid=anchor rank）+
   `merge_stream(kind="uniform", lam=0.25)`：
   y = a + λ(m − a)，m = (group_sum(mem,gid)+a)/(counts+1)，FP32 域计算、
   cast 回模型 dtype。newline/分隔 token 不参与 assignment（mask 之外）。
4. **FULL/E_GATHER 臂**：FULL = visual_token_num<=0 路径（不剪枝）；
   E_GATHER = 官方 EADP（熵过滤原路径，λ=0）。

## 3. 门（先门后全量）

- G0 环境 smoke：LLaVA-1.5 权重加载 + FULL 贪心解码 10 题，输出与
  官方 LLaVA eval 路径一致（prompt/conv vicuna_v1、greedy、max_new 同官方脚本）。
- G1 原路径身份：关闭 RTG/Completion（λ=0、原熵过滤）时，keep mask 与
  EADP 官方路径 **bitwise 一致**（同权重同输入 30 题）。
- G2 N/K：K=128 下每图实际保留 128（NeXT：记录每 crop 配额、取整后实际
  总 K 与额外分隔符数；640 是**五 crop 总预算**不是每 crop 预算）。
- G3 λ=0 ≡ gather：Completion λ=0 输出与纯 gather 逐位一致。
- G4 小样本主方法：TextVQA 100 题 L_R_MAIN025 生成无 NaN/空串，确定性
  （同题重跑两次一致）。
- 过门后才跑十 benchmark 全量；每模型另做 FULL/EADP/AnchorZip 同机在线
  小面板效率（交错配对，同 P2 边界定义）。

## 4. 任务分解

1. [ ] NeXT 权重下载校验（sha/加载）
2. [ ] playground eval 数据准备（10 benchmark；VizWiz 本地 val）
3. [ ] LLaVA-1.5 移植 + G0–G4
4. [ ] LLaVA-1.5 十任务全量（K128）+ 评分器对齐（各 benchmark 官方评分脚本）
5. [ ] LLaVA-NeXT 移植（5-crop quota 记录）+ 门 + 十任务全量（总 K640）
6. [ ] 两模型在线小面板效率
7. [ ] 其他方法行引用 EADP 原表并标 reported
