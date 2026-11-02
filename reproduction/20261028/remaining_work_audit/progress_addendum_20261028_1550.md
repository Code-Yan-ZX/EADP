# 2026-10-28 15:50 UTC 进展附录

当前正式台账为 **10完整／0pending／0失败，七个剪枝论文目标均未复现**。该计数不包含诊断面板或保留的失败启动；本轮没有新增benchmark主行。原14:20快照、原审计及旧九行ledger保持，完整NeXT CLI128／64／32现均有合格5000题证据，不再记覆盖缺项。合法本机分数与source gate通过，不等于paper复现。

| 七目标 | 论文 | 已接受本机例值（不选最高） |
|---|---:|---|
| v15 MMB K128 | 62.7 | β1 61.564626；β2 61.394558 |
| v15 POPE K32 | 86.7 | 1/1 wait 82.220237；.5/1 wait 82.706013；历史.5/2 wait 84.073600 |
| NeXT TextVQA CLI128／名义640 | 59.2 | β1 56.944；历史β2 57.866 |
| NeXT TextVQA CLI64／名义320 | 57.0 | β1 54.900；历史β2 55.410 |
| NeXT TextVQA CLI32／名义160 | 54.2 | 历史β1 50.966；β2 52.086 |
| Qwen Doc K256 | 62.8 | 公开FA2 61.166042 |
| Qwen AI2D K256 | 75.7 | 公开exact fallback 74.352332；paper judge未闭合 |

Qwen Eq7仅把聚合分母`.010001`改为`.01`。CPU缓存对照的source importance与原GPU不逐bit相同：CPU顺序1/16变、最终集合0/16变，只是CPU条件参考。随后实际GPU缓存对照的16题source importance／有序256／完整gain trace／最终空间BF16四重锚点全过；variant有序ID、最终集合、实际BF16输入均 **0/16变**。两次结果不能合并成“GPU顺序1变”，也不支持启动无实际输入变化的全量公式变体。CPU exec77094 exit0、worker2804985/ticks1333248456退出；GPU root exec27423 exit0、controller2832239/ticks1333367682与native2832246/ticks1333367725退出。

[CPU报告](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/qwen/eq7_aggregation_denominator_cpu/comparison.json)、[实际GPU独立报告](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/qwen/eq7_aggregation_denominator_gpu_cache/execution/independent_verification.json)及[GPU退出证明](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/qwen/eq7_aggregation_denominator_gpu_cache/root_terminal_process_proof.json)保留数值和进程证据。

原生离线`CLIPTokenizerFast`对全5000的完整guidance IDs／attention SHA与已有两套refs完全匹配。完整guidance有881题超过77 tokens、最长390；仅问句为5–41，5000题均非空且单segment。旧native16缓存确实存在projected／global／dense／visualsim，缺真实CLIP `image_embeds`，所以旧cache加文本塔不足以闭合crosssim；不能从投影特征补造该向量。[精确tokenization](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/next_textvqa/guidance_question_boundary_exact_tokenization.json)及[旧缓存边界](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/next_textvqa/guidance_question_boundary_cache_feasibility.json)均保留。

新NeXT固定16完整32次原生生成，所有source先逐bit锚定旧native特征，并锚定已接受K64完整运行的quota／mask／有效前缀／输出；随后仅改变CLIP guidance的OCR边界。LLM完整OCR prompt／IDs、图像、实际projected／CLIP image_embeds／visualsim与非视觉embeddings保持。16/16 quota、最终mask及有效前缀改变，2/16原生输出改变；两臂各16条均以EOS2结束，新tokens2–18，未达到128上限。source实际视觉token范围317–318，question-only为316–319。新32 NPZ保存真实image_embeds等，合计1,312,849,178 bytes／448有限数组。文本由worker原生tokenizer解码；独立CPU门未另重解码。未读GT、未评分，不判改变回答的对错。[完整结果](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/next_textvqa/question_only_guidance_panel/RESULT.md)及[独立报告](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/next_textvqa/question_only_guidance_panel/execution/independent_verification.json)记录两题和全部门。root PTY76915 exit0；controller2837356/ticks1333382280、native2837368/ticks1333382361均退，[root退出proof](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/next_textvqa/question_only_guidance_panel/root_terminal_process_proof.json)闭合。

POPE新增[wrapper at-run绑定审计](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/pope/historical_wrapper_equivalence_20261028/audit.json)闭合helper／worker／recovery及三wait clip补丁的历史源绑定，且生成与prompt的规范化AST等于fresh。仍不能回填五个core文件的历史at-run身份或历史backend／dtype／package／架构环境；当前未见实际启用其他架构的证据。仅凭wrapper文件名不足以要求再跑8910。公开材料沿用[已保存main历史](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/mmbench/public_history_audit/summary.json)与[额外公开页面审计](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/mmbench/public_history_audit/additional_public_recipe_surfaces.json)：可见仅main、三commit、无新增recipe；本附录未发网络请求。公开默认可执行，不代表唯一paper argv；用户没有额外作者资料，不把等待它当下一步。

**唯一当前下一步是CPU准备**同CLI64／alpha.5／beta1／greedy128／SDPA／FP16／三wait的全5000 question-only guidance假设；LLM完整OCR输入和所有数值计算不改。固定16已证明实际选择与输出可变，提供了直接判别力，但inactive源码注释没有唯一规定paper应如此运行，也不能分离OCR语义与分段影响。目前无该新完整运行GPU分配、队列或分数；未来根授权和实际launch另记。

四个immutable文件本次各只hash一次，全部吻合原SHA：

| 文件 | SHA256 |
|---|---|
| [original_comparison](/media/disk2/YZX/research/EADP_v2/docs/versions/v2/eadp_comparison_20261028/comparison.json) | `fe9e3ba86ae5b353699d2b17427df392b11a2ff26ff8a2be9cc3956bd0f5a01d` |
| [original_evidence](/media/disk2/YZX/research/EADP_v2/docs/versions/v2/evidence.json) | `8f41a119427a78b360a244949c9f8a0e44dd6fe0e6f615390cccd3b3615b2013` |
| [original_nine_run_ledger](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/full_run_ledger.json) | `cb3f29cc5197bebc735f7f99f0b9885c96208f05e960c5eafb0266f3571eed32` |
| [original_ledger_builder](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/build_full_run_ledger.py) | `0cd8df57fb4981c37cb373f95da1645f4c19b2d295d74a6fcfbf730cfc8ceae3` |

本附录仅新增JSON／Markdown，精确路径、各终端SHA和PID身份见[结构化证据](/media/disk2/YZX/research/EADP_v2/reproduction/20261028/remaining_work_audit/progress_addendum_20261028_1550.json)；没有再次verify、评分或启动模型。
