# 七项未复现目标：论文、完整本机证据与公开配置空缺

> 后续进展：本文保留写作时的证据快照。Qwen 两条 FULL 路径及其 DeepStack 开关全量诊断后来已经完成并通过复核，见 [双路径结果](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/qwen/doc_full_two_routes/RESULT.md) 和 [DeepStack 开关结果](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/qwen/doc_full_ds_toggles/RESULT.md)。这些是完整 token 的路径诊断，不替代下表 K256 剪枝结果。后来又完成了 [K256 的 DeepStack 双臂诊断](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/qwen/doc_full_k256_ds_pair/RESULT.md) 和 [MMB 指导内容双臂诊断](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/mmbench/question_only_guidance_full_run02/RESULT.md)；它们修改或启用公开代码中的输入路径，不能证明论文实际配方，也不替代下表原公开配置结果。

核查日期：2026-10-30。七项论文剪枝目标均仍未复现。本文是独立证据快照，不改原 comparison、历史结果或台账，也不把通过源码/来源/终端验证等同于论文复现。

本机新结果仅采用已接受的 [full_run_ledger_v4.json](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/full_run_ledger_v4.json)：12 complete、0 pending、0 verification failed，SHA256 e90aebfce9d1723002f5275c031e9fb283388d1fc4e89c80767df571606fdaf2。该计数描述这份已完成台账，不代表当前没有其他活动实验。正在生成的预测、前缀、分数及尚未通过完整终端门的双 FULL 结果均不在本文。

有来源的历史结果来自原 [comparison.json](/media/disk2/YZX/research/EADP_v2/docs/versions/v2/eadp_comparison_20261028/comparison.json)（SHA256 fe9e3ba86ae5b353699d2b17427df392b11a2ff26ff8a2be9cc3956bd0f5a01d）和 [NeXT 历史小预算完整复核](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/next_textvqa/small_budget_historical_recheck/verification.json)（SHA256 a058bf638281fd159a4f18948dc1e6aabbcc9093091510d7affc594c0e23e108）。文件在本机本身不构成来源证明；这些入口另存原预测路径、实际 SHA、题数、评分及协议。旧运行的未记录环境不以当前安装版本回填。

下表“新”共列出台账中的全部 12 次完整运行，不取最高值；“历史”独立保留，不并入这 12 次。差值统一为本机减论文，单位为百分点，显示到小数点后六位；未舍入值及 Decimal 差值仍以来源工件为准。

| 目标及论文报告 | 可靠本机配置与完整分母 | 已接受本机值；本机−论文 | 现有公开材料未唯一指定的主表信息 |
|---|---|---|---|
| LLaVA-1.5-7B，MMB-EN，K128；循环组准确率 62.7%（表1） | 新：α.5/β1/q.2，旧 EN20230712，4377行/1176组，原代码及仅三 wait 各一轮；同题集另有 α.5/β2/三 wait。历史：V11，4876行，不与新分母混同。 | 新 β1原代码 61.564626；−1.135374。新 β1三 wait 61.564626；−1.135374。新 β2三 wait 61.394558；−1.305442。历史 V11/β2三 wait 60.061920；−2.638080。 | 主表实际 MMB split/题集指纹、μ/β/q、提示与评分快照未唯一指定。公开脚本指定 EN20230712，不等于已证明论文恰用这些行。 |
| LLaVA-1.5-7B，POPE，K32；random/popular/adversarial 三类宏 F1 86.7%（表1） | 新及历史均为规范8910题：2910/3000/3000；新 α1/β1 原代码、α1/β1三 wait、α.5/β1三 wait、α1/β2三 wait 各完整一轮。历史另有 α.5/β2三 wait。q.2、greedy128。 | 新 1/1原代码 77.393686；−9.306314。新 1/1三 wait 82.220237；−4.479763。新 .5/1三 wait 82.706013；−3.993987。新 1/2三 wait 84.278884；−2.421116。历史 .5/2三 wait 84.073600；−2.626400。 | 主表实际 μ/β/q、生成与低层运行快照未唯一指定。α1/β2是公开材料支持的固定单点控制，不是已确认的论文 argv。历史 .5/2 的全运行环境未逐字段闭合，不能把四分数当完整同环境 2×2 因果实验。 |
| LLaVA-NeXT-7B，TextVQA，名义 K640；官方共识准确率 59.2%（表2） | 新 CLI128/α.5/β1/q.2/三 wait；历史 CLI128/α.5/β2/三 wait；均5000题。672×672输入、原2880 tokens、五crop importance配额、完整OCR guidance、greedy128。 | 新 β1 56.944000；−2.256000。历史 β2 57.866000；−1.334000。 | 主表实际 μ/β/q、指导文本的OCR边界及运行快照未唯一指定。CLI128不是总128；名义640不等于每题实际恰好640，原 floor 配额规则保持。 |
| LLaVA-NeXT-7B，TextVQA，名义 K320；官方共识准确率 57.0%（表2） | 新 CLI64/α.5/β1/q.2/三 wait、完整OCR guidance；新同设置仅CLIP指导改为问句；历史 CLI64/α.5/β2/三 wait；均完整5000题。LLM提示在问句指导控制中仍保留全部OCR。 | 新 β1完整OCR 54.900000；−2.100000。新 β1问句指导 54.958000；−2.042000。历史 β2 55.410000；−1.590000。 | 与K640相同。公开 loader 的 inactive split 注释支持问句指导假设，但论文没有唯一规定它；完整对照只增加 .058点，不能称配方恢复。 |
| LLaVA-NeXT-7B，TextVQA，名义 K160；官方共识准确率 54.2%（表2） | 已接受历史 CLI32/α.5/β1及β2、三 wait、q.2、完整OCR guidance、greedy128；两份均完整5000题，已用fresh作者CLI及独立逐题公式重新核验。 | 历史 β1 50.966000；−3.234000。历史 β2 52.086000；−2.114000。 | 与K640相同。这里已有完整公开默认 β1结果，不是待补的预算覆盖；历史复核不是当前环境的新生成。 |
| Qwen3-VL-8B，DocVQA，K256；ANLS 62.8%（表4） | 新 DocVQA_VAL5349，α.5/β2/q.2、固定1024²/原1024视觉tokens、公开EADP legacy inputs_embeds路径、作者FA2、实际BF16、greedy2048；历史同名K256配方另存。新评分保留原生pandas NA转换及全部分母。 | 新 61.166042；−1.633958。历史 61.135578；−1.664422。 | 主表实际 EADP 类/生成快照、Qwen专用的特征与池化适配、所对应FULL生成路径未唯一指定。新公开FA2运行参数已知，不能笼统写成本机 argv 未知；历史精确包版本/后端则未直接记录。 |
| Qwen3-VL-8B，AI2D，K256；准确率 75.7%（表4） | 新 AI2D_TEST3088；与Doc同α.5/β2/q.2、固定1024²、FA2/BF16、greedy2048与legacy路径。原公开API尝试失败后按源码fallback至 can_infer exact matching；未解析60条保留全部分母。历史K256另存。 | 新原生fallback 74.352332；−1.347668。历史 74.546632；−1.153368。 | 除Qwen共享入口信息外，论文实际judge模型/version/endpoint/缓存未明确。公开alias与实际映射不能冒称论文judge；事后解析辅助值不替代正式主分。 |

论文值的一手依据是 [EADP v1正文与表格](https://arxiv.org/html/2607.02484v1)，对应本机新下载官方TeX的 [v15表1](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/pope/arxiv_release_check_20261028/table_source_crosscheck/main_llava-1.5-7b.tex:23)、[NeXT表2](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/pope/arxiv_release_check_20261028/table_source_crosscheck/main_llava-1.6-7b.tex:24) 和 [Qwen表4](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/pope/arxiv_release_check_20261028/table_source_crosscheck/main_qwen3-vl-8b.tex:20)。[逐格核对记录](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/pope/arxiv_release_check_20261028/table_source_crosscheck/crosscheck.json)已核三主表全部99格；本文不重新下载或改写论文值。

公开命令和论文实际命令须分别判断。现有公开源码是提交 e1a08801461c181871b9ff1cb0803b8e9966f083；以下命令参数可以从源文件明确恢复，但现有论文及归档材料没有逐条证明它们就是主表实际 argv。

| 公开入口 | 可明确恢复的设置 | 来源与边界 |
|---|---|---|
| v15 MMB shell | 第1参数K；默认β1、α.5；EN20230712、single-pred、vicuna_v1、temperature0；入口固定max_new_tokens1024。 | [官方mmbench.sh](https://github.com/SJTU-DeepVisionLab/EADP/blob/e1a08801461c181871b9ff1cb0803b8e9966f083/LLaVA/scripts/v1_5/eval/mmbench.sh#L8)。本机另运行三wait及β2，分别标注。 |
| v15 POPE shell | 第1参数K；默认β1、α1；vicuna_v1、temperature0；loader默认max_new_tokens128、beam1。 | [官方pope.sh](https://github.com/SJTU-DeepVisionLab/EADP/blob/e1a08801461c181871b9ff1cb0803b8e9966f083/LLaVA/scripts/v1_5/eval/pope.sh#L10)、[loader](https://github.com/SJTU-DeepVisionLab/EADP/blob/e1a08801461c181871b9ff1cb0803b8e9966f083/LLaVA/llava/eval/model_vqa_loader.py#L147)。三wait是单独声明的同步控制，不叫作者原代码逐字执行。 |
| NeXT TextVQA shell | CLI预算×5作名义预算；默认β1、α.5；v0.5.1 OCR questions、vicuna_v1、temperature0、loader默认128。 | [官方textvqa.sh](https://github.com/SJTU-DeepVisionLab/EADP/blob/e1a08801461c181871b9ff1cb0803b8e9966f083/LLaVA/scripts/v1_6/eval/textvqa.sh#L8)。[附录明确采用importance-based配额](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/geometry_public_source_audit/eadp_source/sections/appendix_or_supplementary.tex:147)，不能为凑整数预算改配额。 |
| Qwen run_eadp及注册 | shell默认α.5/β2；8B注册固定1024²、custom prompt、max2048；实际do_sample=False。模型源码明确FA2，核心剪枝q.2。 | [官方run_eadp.sh](https://github.com/SJTU-DeepVisionLab/EADP/blob/e1a08801461c181871b9ff1cb0803b8e9966f083/Qwen_vl/scripts/run_eadp.sh#L9)、[官方注册](https://github.com/SJTU-DeepVisionLab/EADP/blob/e1a08801461c181871b9ff1cb0803b8e9966f083/Qwen_vl/VLMEvalKit/vlmeval/config.py#L1258)、[FA2加载](https://github.com/SJTU-DeepVisionLab/EADP/blob/e1a08801461c181871b9ff1cb0803b8e9966f083/Qwen_vl/VLMEvalKit/vlmeval/vlm/qwen3_vl/model.py#L151)。注册列出多个参数组合，不证明主表选择了其中哪一条。 |

LLaVA剪枝核心硬编码q.2、熵温度τ.01、聚合γ约.01、Gaussian 3×3/σ1/reflect；源码里的helper默认q.5不等于encode实际q。[源核心](/media/disk2/YZX/research/EADP_v2/_reproduction/author_eadp/LLaVA/llava/model/llava_arch.py:583)与[论文方法](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/pope/q03_fixed_cache_mechanism/public_sources/sections/methodology.tex:114)已逐项审查。论文消融的μ.5、q.3和β各行限定LLaVA-1.5/K128/五任务，不能据其单格最高值推定七项主表argv，见[消融表的范围](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/pope/q03_fixed_cache_mechanism/public_sources/tables/main_ablations.tex:3)。

当前有证据支持的后续判断如下；可执行性不等于论文配置身份，也不预言分数会提高。

| 目标 | 尚有判别力的下一项，或明确没有新机制 |
|---|---|
| MMB K128 | 暂无新的明确公开修复机制。三wait逐题回答不变、单β2更低、作者公开解析和循环评分已闭合。703长提示行的分层不足以解释全部缺口，多数净剪枝损失发生于短提示完整组，故不据此重复截断/分段实验。见[完整FULL配对](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/mmbench/full_vs_pruned_complete/comparison.json)。 |
| POPE K32 | 暂无新的明确公开修复机制。负cross-cos、facility加权轴、非负映射及Gaussian设置与附录匹配；α1下dense有限贡献为零，q、padding和Eq7变体不能解释该分支剩余差距。Eq6/10及q.3固定16已有最终选集不变的阴性证据。β2已完成全8910，不能把未达目标改写为“继续β网格”。见[新1/2完整结果](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/pope/full_alpha1_beta2_single_control_run02/RESULT.md)、[q.3机制结果](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/pope/q03_fixed_cache_mechanism/RESULT.md)。 |
| NeXT K640 | 暂无支持新全量变体的证据。Eq6在固定16的β2/CLI128有一例最终mask改变，已做同缓存native输入回放；限定128新token内输出相同、且未EOS，因此不能推广为全数据无影响，也不能据一例直接安排完整准确率变体。见[原生公式面板](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/next_textvqa/gpu_formula_facility_panel/RESULT.md)、[单例回放](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/next_textvqa/known_mask_case_generation/peer_terminal_review.json)。 |
| NeXT K320 | 问句指导的公开注释假设已完整检验：实际输入改变，完整分数仅+.058点；不把它称为paper recipe或修复，不自动扩到其他预算。见[全5000结果](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/next_textvqa/full_K64_question_only_public_comment/RESULT.md)。 |
| NeXT K160 | 不存在“尚缺β1全5000”的覆盖缺项；β1/β2完整阴性结果均已核。没有公开一手依据支持该预算独有的新修复，故不重复clone/默认入口/已否定padding控制。 |
| Qwen DocVQA K256 | 优先检查同环境两FULL路径：EADP K1024原样返回完整features，但仍走legacy inputs_embeds；plain FixedRes走generate(**inputs)。两者在同FA2/权重/输入下可区分“入口整体差异”与“256选择损失”；需要完整终端证据后才读取/评分，本文不含正在生成的数据。不能单由输出差异归因DeepStack或mRoPE，也不先将native-restored数值替代paper legacy行。 |
| Qwen AI2D K256 | 先用Doc两FULL结果判断共享入口问题；现有公开材料未给唯一paper judge，不能为抬分换judge或改解析。已有固定辅助规则及未解析上界独立保留，正式74.352332及3088分母不变。 |

Qwen两FULL假设直接来自[作者pruner的K≥N无损bypass](https://github.com/SJTU-DeepVisionLab/EADP/blob/e1a08801461c181871b9ff1cb0803b8e9966f083/Qwen_vl/model/pruner.py#L271)、[plain FixedRes原入口](https://github.com/SJTU-DeepVisionLab/EADP/blob/e1a08801461c181871b9ff1cb0803b8e9966f083/Qwen_vl/VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py#L295)和[EADP继承的inputs_embeds入口](https://github.com/SJTU-DeepVisionLab/EADP/blob/e1a08801461c181871b9ff1cb0803b8e9966f083/Qwen_vl/VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py#L495)。已有[固定16 CPU bypass核查](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/qwen/full_route_bypass_cpu_audit_20261030/report.json)只证明选择器不改features，不是GPU生成或准确率结果。

FULL校准不计入上述12条剪枝运行：POPE FULL85.878707按论文精度同为85.9；MMB FULL64.370748对paper64.7；NeXT FULL60.370对paper60.3（舍入为60.4，不能写同舍入）；Qwen历史plain FULL94.483963按论文精度同为94.5，但不是当前同FA2双FULL因果基线。它们用于限定剪枝损失解释，不能替代七项剪枝分数或补齐平均分。

本文没有新增GPU、评分、验证器运行或外部检索；没有读取正在运行的预测。主表未公开身份被逐项记录为证据空缺，不作为停止依据公开材料继续排查的理由；同样，未发现新机制时不制造参数网格、重复阴性控制或用源码一致性宣布复现成功。
