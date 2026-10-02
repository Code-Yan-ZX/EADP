# Stage 1 Cross-Stream Redundancy Pilot — 协议

**Branch**: `codex/stage1-cross-stream-pilot`(base = `ad08187`,worktree
`/media/disk2/YZX/research/EADP_amp`)· **Date**: 2026-10-02
**本文件在任何本轮准确率/计时数字产生之前写定并冻结。**

## 0. 状态与边界

- **性质**:有界 pilot。检验的假设:主路相似邻居在其他(DeepStack)流中是否
  仍可替代——以此作为 EADP Stage-1 importance 的一个 forward-only 代理信号。
  这是代理信号,**不预先称为因果重要性或已验证的补全损失**。
- 不训练;不扫超参;不扩大到 K=128 或其他模型;不启动 CONFIRM。
- 本轮只运行现有 anchor_merge_pilot manifest 的 **DEV**:TextVQA_VAL /
  DocVQA_VAL / OCRBench 各前 100 题,共 300 题。索引与图像标识直接复制该
  manifest,不重新抽样;原 manifest 文件 SHA256 =
  `8d55e4e491abee2876217c2b10d2d17d47d75b4dbff657f64eefe8dce9c2fc97`。
  **数据已观察声明**:DEV-100×3 已被 E0 / anchor-merge round 1–2 / 多个
  discovery 轮使用过,本轮是 reuse evaluation set,只作探索,不能作独立确认。
- 默认本轮**重跑六组**全部 DEV 预测(含 smoke 最多 1800 条);旧结果
  (`outputs/anchor_merge_pilot/acc/dev/...`)只用于复现核对,不直接用历史
  均值填表。

## 1. 模型与执行路径

- 模型:`Qwen3-VL-8B-Instruct-1024`(VLMEvalKit fixed-res wrapper,1024×1024),
  native E0 engine(`model/native_qwen3.py`):DeepStack 三路注入、3D mRoPE、
  贪心 decode、SDPA,与 anchor-merge round 1–2 完全一致。
- 主设置 N=1024、K=256、贪心、max_new_tokens=2048。
- 环境:qwen3vl_clean(python 3.10.20,torch 2.10.0+cu128,transformers
  4.57.6,cuda 12.8),单卡 A40,无争用窗口计时。
- 代码基线 commit:`ad08187af48cb9218770fdd52f3e4d6d8cbfa1e8`。

## 2. 冻结的公共设置

- Stage 2 始终调用 **official facility-location 原实现**
  (`instrumented.SELECTORS["facility"]` → `model.pruner._greed_select_impl`),
  使用原主特征相似度矩阵(`pruner._similarity`,即 `0.5·(cos+1)` 映射)与原
  greedy / tie-breaking。不改 Top-K、block8、DPP,不换相似度空间。
- 所有方法输出 K 个有序、唯一、升序 anchor 索引;native DeepStack 注入、
  3D mRoPE、原位置与 cache 处理全部保留。
- Completion:在原始主特征上,dropped token 按 cosine argmax 分配给 anchor
  (`amp_common.compute_assignment`,FP32,首最大→更低 anchor 序号);组均值
  含 anchor 自身;y_a = v_a + 0.25·(mean(G_a) − v_a)。**只聚合主路**;三路
  DS 全部保持 DS[s][keep] 原值。
- Stage 2 算法固定不代表 anchors 固定:**每个 scorer 重新选 anchors、重新
  分组**;同一 scorer 的 gather / completion 配对共用同一组 anchors。

## 3. 新 Stage 1:`cross_stream` 精确定义

固定 m=8、eps=1e-8、score_floor=0.1。所有评分 FP32、no_grad、每图独立。

输入:原始主特征 V[N,D] 与 encode 已返回的三路 DS[s][N,D]。不增加
encoder/LLM 前向,不抽取额外 attention,不读取答案;问题文本正常输入 LLM。

1. 按行 L2-normalize V(norm 下限 eps);**复用 official facility 所用的主路
   相似度矩阵** `S = pruner._similarity(feats)`(0.5·(cos+1),单调等价于
   cosine,排序一致)。
2. 对每个 i 取主路相似度最相似的 **min(8, N−1)** 个其他位置 → nbr[i];排除
   自身(mask **独立副本**的对角线,绝不改动传给 facility 的原矩阵);相同
   相似度按**较小原始索引优先**(stable descending sort)。
3. 对每路 DS[s] 单独按行 L2-normalize(H_s,norm 下限 eps);邻居均值
   mu_s[i] = mean(H_s[nbr[i]])(均值不再归一化)。
4. d_s[i] = Σ(H_s[i] − mu_s[i])²;每图每流 z_s = d_s / (mean(d_s) + eps)。
   零残差流自然贡献零;零范数行按 eps 处理并**记录数量**。
5. w_cross[i] = 0.1 + mean_s(z_s[i])。**不混入** EADP importance,无 entropy
   filtering、min-max、Gaussian smoothing、beta polarization 或 query gating。
6. 将 w_cross 按原接口送给 official facility(作为覆盖权重;importance 高
   不保证 token 自己被选中)。只读取 DS,不修改任何 DS 内容。
7. N≤K 时所有 arm 直接完整保留并跳过评分;多图输入**不可跨图**建邻域、
   分组或归一化(逐图循环,与 `_eadp_parts` 相同的 split 结构)。

**对照 scorer**(其余流程与 cross_stream 完全一致):

- `uniform`:w[i] = 1。
- `main_residual`:沿用同一主路 nbr,以 normalize(V) 替代 H_s 得 d_main;
  w_main[i] = 0.1 + d_main[i]/(mean(d_main)+eps)。
- (`eadp`:official EADP scoring 原样,即现役 Stage 1。)

## 4. 六组实验(全部预注册)

| arm | Stage 1 | Completion |
|---|---|---|
| E_GATHER | 原 EADP 评分(official,当前 b1 路径 alpha=0.5, beta=2.0) | 无 |
| E_MAIN025 | 原 EADP 评分 | 仅主路,λ=0.25 |
| X_GATHER | cross_stream | 无 |
| X_MAIN025 | cross_stream | 仅主路,λ=0.25 |
| U_MAIN025 | uniform | 仅主路,λ=0.25 |
| V_MAIN025 | main_residual | 仅主路,λ=0.25 |

注:E_MAIN025 对应 round-2 的旧 MAIN025(不是 U025/BOTH);E_GATHER 对应旧
BASE。原 EADP 参数以当前 b1 路径为准(alpha=0.5、beta=2.0),不得误用类构造
默认 beta。

## 5. 冻结 bank 与正确性门

- 新建 `Qwen_vl/scripts/stage1_cross_stream_pilot/` 与
  `Qwen_vl/outputs/stage1_cross_stream_pilot/`,不覆盖旧 bank、预测或分析。
- 4 个 scorer(eadp / cross_stream / uniform / main_residual)× 3 数据集
  冻结 bank(keep / gid / gsize + importance 与残差诊断),元数据含 scorer、
  全部评分参数、K、scope、λ、manifest SHA、模型与代码版本哈希;续跑先严格
  核验元数据,不能只按题目 ID 跳过。E 组 anchors 同时与 round-1 旧 bank
  (`bank_dev_{ds}.json.gz`)核对。
- 正确性门(全过才允许正式生成):
  - **G1 评分数学**:FP32 向量化 cross_stream / main_residual 与显式逐点
    循环一致(合成 + 真实样本,rel<1e-5);w 有限且非负;确定性。
  - **G2 邻居**:无自身;并列取更小索引;多图不跨图(拼接双图输入下逐图
    结果 = 独立运行结果,索引不越过图边界)。
  - **G3 退化**:重复 token、全零流(贡献零、w 有限、零范数行计数)、N≤K
    直接保留,均有确定行为;DS 三路无一被静默省略。
  - **G4 E 复现**:每数据集 ≥3 题,E_GATHER 与旧 BASE、E_MAIN025 与旧
    MAIN025 的 anchors(bank keep bitwise)与贪心输出(预测文本 bitwise)
    一致;prefill logits 经 λ=0 门间接覆盖(旧 shard 未存 logits,如实记录)。
  - **G5 λ=0 恒等**:E/X 两个 scorer 的 λ=0 路径与各自纯 gather 的主路/DS
    特征与 prefill logits bitwise 一致。
  - **G6 bank 复现**:新 bank 与现场重算 keep/gid 全等;DS 恒等于原
    DS[keep]。
  - **G7 不变量**:native 长度、位置、DeepStack 注入、层调用、cache 不变量
    通过(n_vis_kept=256、无重复、升序、layer_calls/cache ok)。
- smoke:六组 × 每数据集前 10 题(属于 DEV 配额);检查空答案、重复、截断
  及实际 token 数;正确且配置相同的 smoke 预测可续用。失败门修复后重跑,
  不跳过。

## 6. 评分与统计

- 官方 VLMEvalKit evaluator;沿用已修正的 DocVQA ANLS 与 OCRBench D-4 规则;
  OCRBench 用实际题数归一化,保留 math 分支大小写与换行处理。
- 保存逐题原始预测、得分、截断、image ID;逐题键完整匹配 manifest;
  **逐题聚合必须在官方舍入精度内复现 headline**(不用 ±0.5/1 分的宽松自检)。
- 每任务与三任务等权 macro 均 0–100;同题配对,数据集内按 image cluster
  bootstrap **20,000** 次,seed=**20261002**,报告 95% CI。rescue/break 按
  0.5 阈值单独定义并报告。
- **主要对比(预先指定)**:X_MAIN025 − E_MAIN025、X_GATHER − E_GATHER。
  **机制对比**:X_MAIN025 − U_MAIN025、X_MAIN025 − V_MAIN025、
  X_MAIN025 − X_GATHER,以及 completion 的 2×2 交互
  (X_MAIN025 − X_GATHER) − (E_MAIN025 − E_GATHER)。
- **非劣判定**:差值 = 新 − 旧;固定 margin 0.5 macro 点;CI 下界 > −0.5
  → 本 DEV 支持非劣;CI 上界 < −0.5 → 超出容忍损失;其余不确定。该 margin
  是本轮筛选约定;已观察 DEV 不能作独立确认;不显著 ≠ 相同。
- 诊断:每组 importance 分布、三路残差均值/零值比例、与 EADP 的 anchor
  overlap、组大小分布;检查 scorer 变化确实改变 anchors(不只热图)。
- 全部六组结果都报告,不挑任务、不临时改 m / score_floor / 流权重;明显
  失败也完成有界 pilot 后如实总结。

## 7. 真实效率

- 同 DEV 各数据集前 10 题(共 30 题)配对计时 E_GATHER、E_MAIN025、
  X_GATHER、X_MAIN025。
- **现场** encode → scoring → official facility → assignment/completion →
  prefill → decode;不以 bank lookup 替代评分与 selector。主路相似度每次
  请求只算一次并供评分/selector 复用;分段记录 similarity、Stage 1、
  facility、completion,不漏算不双算。
- 15 个预热配对 block;测量交错顺序、CUDA 同步;每 arm 每次新建
  state/cache;reset peak memory;ignore_eos=True 固定 64 tokens 并断言
  实际长度。EADP 路径的 instruction embedding 计入其完整路径(新 scorer
  少了这一步的真实差异保留)。
- 报告含预处理与现场剪枝的 TTFT、总生成时间、峰值显存、原始配对记录;
  明确标注 TTFT 测量边界(从 message 起到首 token,含图像预处理与全部
  现场 scoring/selection);不把纯 prefill 或 bank 时间称为端到端加速。

## 8. 交付

1. 正确性门结果、六组每任务/macro 主表、主要差值与 CI、机制对照、四组
   真实效率表。
2. `docs/stage1_cross_stream_pilot_report.md` + 独立原始预测/逐题评分/
   诊断 JSON + 运行脚本与环境/代码/配置哈希。
3. 结论:Stage 1 是否接近 EADP;整体是否接近当前冻结方法(DCC =
   official EADP + Facility + MAIN025);是否有超出 uniform / 主路残差的
   线索;额外延迟;是否值得进入新图像面板确认。
4. 列出全部失败、未执行项、偏离与实际 GPU 用时;不编造未运行数字。
5. 只提交本轮代码、协议、报告与小型统计 JSON;不提交权重、数据集、大型
   缓存;不合并 main。

## 9. 偏离记录

(均不改变实验定义;详见报告 §7。)

1. **D-1(首次 smoke 评分前)**:X/U/V 四组的 90 条 smoke 预测在 bank
   重建(补全 w 诊断字段;keep/gid 逐样本 bitwise 核验不变)后未通过严格
   bank-sha 续跑核验,删除重生成;E 组不受影响。
2. **D-2/D-3(首次门运行内)**:correctness 脚本三处测试侧缺陷(门键名、
   uniform 参考值、G3 零流构造、G2/G3 合成 gthw 几何)暴露即修复重跑,
   G1–G7 最终全过;生产/评分路径未改动。
3. **D-4(报告前)**:2×2 交互由两次独立 bootstrap 差改为四 arm 同抽样
   联合 bootstrap(分析增强)。
4. **D-5**:bank/shard 元数据 code_commit=`7db444a`;其后仅测试/分析脚本
   与 bank 诊断字段变更,选择/评分代码未变。
