# EADP V2 系统复现核查

日期：2026-10-28，北京时间。目标是实际复现 EADP，定位并验证论文与本机大差值的原因。不能用参数搜索、改指标、挑选历史最高值或只完成静态检查来替代这个目标。

## 授权与保留

用户要求对照作者源码和下载的数据集系统彻查，必要时重新 clone、下载和完整重跑。这覆盖先前版本整理阶段的“不新增 GPU 实验”范围；本轮允许为复现运行 GPU。V1 快照、原 EADP/EADP_amp 文件、预测、评分与已冻结的 V2 初始清单保持原样。新源码、环境与实验均隔离保存。

仅使用 EADP 论文报告值和本机真实实验结果的来源规则继续有效。子集诊断不作为完整 benchmark 分数；来源合格也不自动证明配置与论文一致。

## 对照基准

- 官方仓库：`https://github.com/SJTU-DeepVisionLab/EADP`。
- 重新 clone 至 `_reproduction/author_eadp`，当前提交 `e1a08801461c181871b9ff1cb0803b8e9966f083`，与之前作者版本相同。
- 论文：`https://arxiv.org/html/2607.02484v1`，按原表号、模型、预算、split 和原始单位比较。
- 本机对照起点：`docs/versions/v2/eadp_comparison_20261028/comparison.json`。

每个差值都依次检查：源码与入口 → 依赖及 attention 后端 → 模型和视觉编码器实际权重 → 数据集版本/全量输入及图像 → 问题/提示/预算/生成参数 → 原始回答完整性 → 作者评分器和分母。静态差异必须由受控运行验证，不能直接当成根因。

## 本轮首先处理的差值

| 模型/设置 | 论文 | 原本机 | 差值 | 当前核查方向 |
|---|---:|---:|---:|---|
| v15 MMB-EN K128 | 62.7 | 60.061920 | −2.638080 | 作者脚本的 2023 版题集与旧运行 V11 不同；默认 beta 也不同 |
| v15 POPE K32 | 86.7 | 84.073600 | −2.626400 | 作者 POPE 脚本默认 alpha1/beta1，旧运行 alpha.5/beta2 |
| NeXT TextVQA 名义 K640 | 59.2 | 57.866 | −1.334 | 全量数据身份、发布入口与原包装器、作者依赖与实际参数 |
| NeXT TextVQA 名义 K320 | 57.0 | 55.410 | −1.590 | 同上；保留已完成的 beta1/同步控制负证据 |
| NeXT TextVQA 名义 K160 | 54.2 | 52.086 | −2.114 | 同上 |
| Qwen DocVQA K256 | 62.8 | 61.135578 | −1.664422 | 数据/图像缓存与作者指定的依赖及 FlashAttention 2 路径 |
| Qwen AI2D K256 | 75.7 | 74.546632 | −1.153368 | 同上 |

其余已列出的差值继续纳入共享入口和评分检查；表中顺序是排查优先级，不是缩小最终目标或定义“误差低于 1 就成功”。Qwen HallusionBench 生成时剪枝 bank 身份仍需闭合，不能先填入正式复现表。

## 已取得的实质证据

1. **POPE 数据与评分**：按作者 EVAL.md 固定的 POPE commit `e3e39262c85a6a83f26cf5094022a782cb0df58d` 重新下载标签；作者链接的 eval.zip 也已重新下载，23,220,729 字节、SHA256 `2df36a33e3d3947d3e351e627449ee7cc7725d69ecc8dcbb2103d90172fbe17a`，与旧包逐字一致。题数就是 random2910/popular3000/adversarial3000，共8910；没有擅自补90题。所有图像/问题/顺序与固定标签匹配。作者评分器对旧 EADP32/128 重算与原分仅有浮点末位差，因此评分修正不能解释 K32 缺口。
2. **MMBench 输入**：作者 v15 脚本指定旧 EN20230712；重新下载 137,646,313 字节 TSV，4377 行、1176 循环组；完整图像解码和内容清单已验证。旧 V11 的 4876 行不能冒充这个题集。
3. **Qwen 数据**：DocVQA_VAL 和 AI2D_TEST 的完整 TSV MD5 分别为 `d5ee77e1926ff10690d469c56b73eabf`、`0f593e0d1c7df9a3d69bf1f947e71975`，与作者 vendored 数据集定义精确一致。用作者原始解码函数和固定 Pillow12.0.0 在临时目录重建全部2490张唯一图像，实际缓存逐字节全部相同，覆盖8437题；见 `qwen/dataset_cache_audit.json`。不是只检查 TSV。
4. **Qwen 环境差异**：作者发布配置为 torch2.8.0、transformers4.57.1、flash-attn2.8.3，源码明确指定 `flash_attention_2`。目前旧环境目录的安装为 torch2.10.0、transformers4.57.6；合格旧Doc生成于9月22日，该环境随后升级，不能把现在的版本或SDPA归给当时生成。历史精确包/backend并未直接闭合，见已归档provenance_old_run.md。独立环境已安装上述核心版本和官方 ABI-TRUE FlashAttention wheel，未修改原环境。主机 glibc2.31 无法直接导入此 wheel；复用既有容器的 glibc2.35，仅运行挂载的隔离 Python/包。CPU导入、实际FA2 CUDA微内核和四题完整BF16生成均已通过，缺失发布导入模块的隔离补齐见下文；原公开DocVQA/AI2D脚本全量正在运行。不能先断言旧 SDPA 与作者等价，也不能用四题运行检查代替全量结果。
5. **NeXT 数据与已拒绝解释**：新下载 TextVQA GT/作者 OCR 问题与旧文件一致；远端官方图像 ZIP 的目录 CRC/size 与3166张实际使用图像全部一致，固定64题使用的64张重新下载 JPEG 的 SHA256 也一致。全部3166图像的解码RGB、AnyRes FP32/FP16张量、全部5000题 CLIP/LLM token IDs 在实际环境和作者固定小依赖下均一致。之前完整 beta1 K32 控制更低，不能为匹配论文重复盲跑或挑选参数。已检索历史元数据，未发现 K128 beta1 全量，此公开默认设置仍可作为独立预登记复现任务。
6. **权重身份**：v15 与 CLIP 的17个运行时文件、NeXT三片共14,125,909,456字节、Qwen四片及10个处理器/配置文件，已与各自官方固定 HF revision 的 LFS SHA256 或 Git blob SHA1 核验一致。报告分别位于 `pope/model_identity/`、`next_textvqa/current_weight_identity.json`、`qwen/qwen_weight_identity.json` 与 `qwen/qwen_processor_tokenizer_identity.json`；不是仅凭目录名判断权重正确。
7. **依赖实际输入影响**：v15 POPE 的500图/8910题和 MMB 的1132图/4377题，在当前 Pillow/tokenizers/SentencePiece/protobuf 与作者小依赖固定版本下，全部实际图像张量和提示 token IDs 一致。依赖版本不完全相同，但这几项版本漂移没有改变所测的完整输入。

## 首轮完整新结果（保留全部成功与失败）

| 实验 | 全量本机值 | 论文 | 差值 | 结论边界 |
|---|---:|---:|---:|---|
| v15 MMB EN20230712 K128/.5/1，原代码 stream | 61.5646258503 | 62.7 | −1.1353741497 | 4377行、724/1176循环组，本地 exact-letter 循环评分；未提交外部服务器 |
| v15 MMB EN20230712 K128/.5/1，仅三条 stream wait | 61.5646258503 | 62.7 | −1.1353741497 | 4377条原始回答与原代码逐条完全相同 |
| v15 POPE K32/1/1，原代码 stream | 77.3936863461 | 86.7 | −9.3063136539 | 8910题、作者官方宏F1；重复输入2334组中231组有回答冲突，生成稳定性不成立 |
| v15 POPE K32/1/1，仅三条 stream wait | 82.2202369991 | 86.7 | −4.4797630009 | 同8910题、作者及独立宏F1一致；2334组重复输入回答冲突0 |
| NeXT TextVQA 名义K640（CLI128）/.5/1，三wait | 56.9440000000 | 59.2 | −2.2560000000 | 全5000、捕获token重解码/逐题独立共识分、冻结guards和真实退出均通过 |
| v15 POPE K32/.5/1，单alpha对照，三wait | 82.7060125212 | 86.7 | −3.9939874788 | 全8910官方/独立宏F1一致，重复输入2334组冲突0；比1/1仅高0.4857755221点 |
| v15 MMB EN20230712 K128/.5/2，单beta对照，三wait | 61.3945578231 | 62.7 | −1.3054421769 | 全4377、722/1176组，公开helper与独立评分一致，实际退出；较beta1低0.1700680272点 |

MMB 全部 ID/提示/选项/GT/运行时与 XLSX 已独立验证，无解析失败，见 `mmbench/independent_verification.json`。它和旧 V11/beta2/wait 的60.0619195同时保留，不能把差异单独归因于题集。首次 XLSX 转换缺少 openpyxl 的失败记录保留，隔离安装作者固定评分依赖后仅重做 CPU 转换成功，没有重跑模型。

MMB三wait完整对照于18:40结束，作者XLSX、独立评分、全量配对与真实worker退出门均通过，见 `mmbench/streamwait_control/paired_full_comparison.json`。两轮4377条原始回答、提取字母全部相同；JSONL字节差异来自随机answer_id。原生MMB图像传输是阻塞`.cuda()`，POPE则使用non_blocking传输，因此POPE的同步收益不能推广到MMB。

20:18 MMB单beta1→2完整4377结束，作者转换/独立/真实退出及root公开helper复核均通过：722/1176=61.394557823129254，论文差−1.3054421769。相对同旧EN/beta1低0.1700680272点；137原回答改变，49行错→对/50对→错，循环组14错→对/16对→错，见 `mmbench/beta1_vs_beta2_and_v11_full/comparison.json`。为区分旧V11输入，按同ID3589→完整prompt/options3529→同GT3469逐层筛选，再比较原始图像字节、decoded RGB+size、原生CLIP FP32和实际FP16预处理张量，四层均只28行/7完整组可比，28回答全相同；1777唯一payload重建与既有作者输入指纹一致。见 `mmbench/v11_image_overlap_stage2/report.json`，独立review通过；不能将两个不同全量分母的差直接当作题集因果。

另用 fresh 作者 vendored VLMEvalKit 原始 `can_infer`、`prefetch_answer` 和 `prefetch_circular_group` 函数，对两轮完整4377行分别CPU重算，见 `mmbench/public_mcq_metric_check.json`。两轮每行公开解析与本地解析均完全相同，未解析组0，均为724/1176=61.5646258503。由此关闭“本地first-letter解析/循环公式不同导致缺口”的解释；这仍不是外部服务器或API judge评分。论文残差1.135374点未解决。

POPE 原代码结果及失败证据见 `pope/run01_source_as_is_k32/full_result_summary.json` 和 `same_input_consistency.json`。在同一实际图像 SHA 和完整提示下出现 Yes/No 冲突，尚不能单凭静态流代码认定根因；已登记同一8910题、K32/1/1且只插入三条 caller/image/text CUDA stream wait 的完整受控对照，所有数学操作和源文件不变。

该完整受控对照已结束并通过全部8910题顺序/作者评分/独立重算/源码SHA/worker实际退出门，见 `pope/run02_source_wait_only_k32/full_result_summary.json`。仅三wait的对照比原代码高4.8265506530点，并消除本次观测到的231组重复输入冲突；低层调度和并发负载没有逐事件重放，不能将任何单题变化都断言为同步错误。稳定性问题已获得实验证据，论文86.7仍未复现。旧.5/2/wait的84.0735997也保留，它与新1/1/wait存在参数差异，不能挑选较高值宣布成功。

无剪枝POPE基线的两份既有完整8910题原始回答已用新作者CLI及固定标签重新评分，作者与独立宏F1均为85.87870689695657，按论文一位小数同为85.9；两份逐题回答完全相同、重复输入冲突0。见 `pope/baseline_full_rescore/baseline_full_summary.json`。这支持模型/题集/评分通路的FULL基线对齐，不能替代K32剪枝行的复现。将POPE改成accuracy以抬高结果没有依据，继续固定作者F1口径。

固定单alpha1→.5（beta1/K32/三wait）全8910于19:59结束并通过全部终端门，见 `pope/run03_single_alpha05_beta1_wait_k32/full_result_summary.json` 和 `completion_and_release_proof.json`。宏F1仅82.7060125212074，相对alpha1的82.22023699908897提高0.4857755221点，仍低论文3.9939874788点；全部2334重复输入组冲突0，空回答0。raw SHA `4e9d1e307fec4d5052c6808047544e7cef015b9756cd850f96a4b299b879b879`，summary SHA `77438eefdb2eb2b415949a545ce3f23d565a73674da8dc67567180fa14ae542e`。实际worker2475198/ticks1331864351、controller2475197和监督器2449462均已退出，B按原固定授权接给MMB单beta。alpha单独变化不足以恢复paper，历史.5/2的84.0735997不覆盖任何新结果。

POPE 全部500张本机 COCO 图像的 CRC32/字节数与作者链接的官方 val2014.zip 中央目录一致，固定16张直接重新下载的JPEG SHA256也一致，见 `pope/official_coco_image_identity.json`；没有把只存在本机的SHA冒充上游全量SHA验证。

### 进一步的论文与源码对照

`pope/paper_method_source_audit.json` 及其配套 Markdown 记录了真实 CPU 特征诊断：原生 `device_map=auto` 下独立 CLIP 视觉塔、投影与文本塔实际保持 FP32，16图跨模态相似度和熵也是 FP32、NaN为0。FP16 epsilon 下溢只在合成边界例子成立，不作为实际根因。源码确实将77位中的padding/EOS纳入dense候选，16图240个低熵保留位中185为padding，但padding平均权重仅0.537%，且alpha1融合前后importance逐位不变；这不能解释本次1/1的4.48点差距。已有NeXT fixed64 valid-content masking、bounded chunks及两者组合均未恢复精度，没有重复同一GPU控制。

论文明确的负cos、tau=.01、gamma=.01、3×3/sigma1/reflect和facility增益与源码匹配。Eq10分子epsilon、Eq6 quantile阈值/固定top-k及dense候选域存在描述差异，但未证明能解释完整分数。MMB长CLIP输入会被源码分段（703/4377超过77token，最长582），论文只描述M≤77，分段主表配方也未明确。不能把这些静态差异直接宣告为修复。

作者公开Git非shallow完整历史仅3次提交，后两次仅根README变化，相关9个关键源码/脚本/环境文件自首次提交未改；目前可见公共issue仅验证榜单的#1，没有recipe/stream修复说明。证据 `mmbench/public_history_audit/summary.json`。公开任务脚本与generic/README配置不完全一致，主表完整argv仍未闭合。参数对照只用于分离已有结果中的混杂，不按得分继续调参。

19:22补查当前官方arXiv提交历史与TeX源：仍只有v1，无独立ancillary或新的主表运行脚本/环境配方；官方源归档2,043,245字节，SHA256 `7477093f610f243dbaaf6faf6a40175b34e43d793bdbfa01663436b89fe48e28`。三主表99/99论文格（含9个reported Avg）与存档HTML、paper_table JSON和comparison逐格Decimal一致，任务顺序/预算对应一致，见 `pope/arxiv_release_check_20261028/table_source_crosscheck/`。未用HTTP封装Last-Modified时间冒充v2，参考数值不变。

历史检索未发现v15 K32/.5/1完整对照，也未发现旧EN4377/K128/.5/2完整对照。两个固定单因子计划已CPU冻结：POPE `run03_single_alpha05_beta1_wait_k32/run_plan.json`（SHA256 `bf40e511c27cf920dbd287d61126c405346f97fc37728cff3254af608aef5a70`），MMB `beta2_control/protocol.json`（SHA256 `f4a2eff7485313e8ee1bec7553c4897b60abb48ccd4beb34e165e20a09e49f40`）。保持题集、权重、生成、评分与三个wait相同，仅分别改变alpha或beta；不据得分扩展参数搜索。截至18:54尚未分配启动，CPU监督器准备中。

19:14根完成两监督器及独立审查，追加分配后续slot B，授权文件 `reproduction/20261028/slot_B_causal_controls_authorization.json`（SHA256 `8d74244d4ed67376f2bd950c0629cf37ecd1ee58b8682585c7d9dd7146e4fb21`）。这是初始prepared-only阶段之后的运行授权，原数学/生成计划不改。固定顺序为NeXT full5000→POPE单alpha full8910→MMB单beta full4377，每项均须完整作者/独立评分、源码/输入/权重/图像guards和实际worker退出，不按分数方向或重复一致性结果筛选，不自动重试。

POPE CPU监督器2449462/ticks1331757118/exec79985当前等待NeXT；其SHA `5f4d3953beb75321daf8a90bb7322bbff1c1e8163b51f41b9c5333ff8d4a9bfc`，NeXT独立门SHA `bc00c7d7f4c93f2f8c6be670c453656d65c538a2ef1f8ef360c9301723e41623`，准备记录SHA `12a1e68e218dbdd6904d4f3c7d6de29aa7f2d8ab8393a4fc058f989477b4858e`，20项CPU门检查通过。A槽通过固定容器ID/image/init与完整原生argv+cgroup核验，允许原Doc→AI2D进程切换，不放行未知模型。MMB CPU监督器2449729/ticks1331757871/exec8614当前等待POPE，其SHA `da3c25a2fa63229d905fa1ca261d84605b7208c8282516668156adef34acae3b`，11项CPU门检查通过。原监督器异常退出而状态尚未complete时，两者均重读终端记录后明确失败，避免无限等待；观测超时本身不视作失败。具体句柄和claim见各自目录的supervisor/after_pope_alpha记录，不重复启动。

原16图的有界CPU公式对照也已完成，见 `pope/cpu_formula_real_panel/README.md`：Eq10只去分子epsilon导致importance最大绝对变化1.68085e-5；Eq6改为linear q=.2阈值含ties使每图文本15→16，在.5/2下5/16图有变化，最大6.58631e-6。没有测下游准确率。原生FP16 projector在所用CPU PyTorch不支持，facility保留集明确未测，没有用FP32模拟替代原数值路径；不能把importance顶部32索引相同误称为facility集合不变。

20:30在MMB全4377及终端门完成、B真实释放后，根明确分配一次固定16图原生GPU诊断，计划 `pope/gpu_formula_facility_panel/run_plan.json` SHA355f020bb2bcaf9ac37f4daa061f504ec532e155c0cb61be5aa966fc3e2df4ff，经根完整审读和独立peer review。每图只提取一次原生FP32 CLIP/FP16 projector/FP32视觉sim，source AST锚定native encode importance/ordered IDs/quota/final mask；Eq6只改linear q=.2阈值含ties的成员并保留原top15顺序，Eq10仅去分子epsilon。三个既有recipe1/1、.5/1、.5/2共192case完成，无答案/GT/准确率。

真实GPU中Eq10三recipe均16/16 importance变化，最大9.775e-6/9.656e-6/1.681e-5；Eq6分别0/16、3/16、3/16。虽有真实gain扰动，但全部有序32 ID、最终mask和集合均0/16变化，每组overlap32/32，无首次贪心分歧。独立CPU逐16NPZ/896finite数组/192trace argmax/16原生锚点/48identity/16alpha1负控及所有源/17模型stat/16图SHA核验通过；报告 `independent_verification.json` SHAef6e790c133b9f247130f841ed7b93be6f41fc43844a3e4da8eed539b8ef7d46。worker2559819/ticks1332212734真实退出、root exec49578 exit0，原始缓存99,451,611字节保存。结论只限16图，不排除全8910稀有变化，不支持据此启动全量公式变体；旧CPU5/16与GPU3/16另有特征执行和候选顺序约定差异，均原样保留，见该目录RESULT.md与root_terminal_process_proof.json。

Qwen Hallusion历史bank的生成SHA仍未闭合。对V1归档manifest的独立搜索只找到当前银行字节的备份，没有生成记录中的SHA；见 `qwen/hallusion_bank_archive_search.json`。这既不能证明token选择错误，也不能补定生成时身份，继续保留缺失。

NeXT 作者原入口固定64题诊断已完成，所有64个原始回答和实际保留 token 数与既存完整结果的对应题完全相同（0回答变化、0预算变化）。这是入口等价性证据，没有解决完整5000题的论文差距，不能用面板分数替换主表。原GPU进程退出和释放见 `next_textvqa/panel64_author_wait/release.json`。

新beta1与历史合格beta2两份完整5000题已只CPU逐题配对，见 `next_textvqa/beta1_vs_historical_beta2_full/`。相同有序复合ID、完整prompt、GT与3166张图SHA下，fresh作者评分分别56.944与57.866，净差−0.922点；534个原始回答改变，93题分数提高、138降低，4769不变，总分gain73.1、loss119.2，净−46.1/5000。两份合格token trace也已核实：实际保留总数均636–639，2941题数量不同、2059题相同，均值旧637.5054/新637.5106；没有可比的选中token索引或逐crop quota，不推断视觉内容相同。

20:17另将历史无剪枝FULL完整5000用fresh作者CLI及独立闭式共识分重算，仍60.370000000000225/精确60.37，论文60.3，差+0.070。一位小数舍入为60.4，不能写成与论文同舍入。实际5000题均2880 tokens；完整raw/runtime/control/源/模型config/有序prompt/全部GT及3166实际图SHA通过，逐题作者与独立差最大2.22e-16，独立review通过，见 `next_textvqa/full_token_baseline_recheck/baseline_recheck.json`。相对该FULL，beta1损失3.426、beta2损失2.504点；论文640行仅损失1.1点，分别额外损失2.326/1.404点。完整配对中beta1有256题提高/441降低，beta2有272提高/413降低；这是保留历史运行的复核，不是新FULL生成，也不能证明所有运行因素无关。

公开 `llava_arch.py` 先执行importance的beta幂变换，再将该importance送入importance_based的裁块预算分配和取整，因此beta可同时改变选择及裁块quota/取整后的总数；这些总数并不是CLI128/论文名义640的预算映射错误。这里只报告两次完整执行的观测差异，旧新wrapper/观测路径与低层调度无法逐事件重放，不能据此单独归因beta或宣布paper复现。

NeXT内嵌视觉权重与官方base CLIP的全部391个参数已CPU逐tensor比对，见 `next_textvqa/embedded_vision_vs_official_base.json`。内嵌参数存储为BF16，实际加载转FP16；与base FP32转FP16比较391个tensor均不同，base先转BF16后仍有335个不同。因此内嵌vision body不是原base仅做BF16量化所得，与unfreeze配置相符。当前运行加载的是已与官方LFS匹配的NeXT检查点；不能据这一差异直接认定论文差距根因，也不因此替换视觉权重。CPU mmap比对未构建模型或初始化GPU，活动worker与计划未修改。

论文原HTML的对应视觉骨干/默认评测段落也已核对，见 `next_textvqa/paper_vision_weight_policy_audit.json`：没有明确要求舍弃NeXT内嵌视觉权重或替换为原始冻结base CLIP，未发现与该公开加载路径直接冲突。故不据这一身份差异新增替换视觉权重的GPU变体。

Qwen FlashAttention 的实际 CUDA 微内核已在 A40 通过：torch2.8.0+cu128/flash-attn2.8.3，FP16 causal attention 输出符合预定恒值输入的预期；不是只成功导入 Python 包。随后实际作者完整 BF16 模型在四个预先指定的问题上生成成功，后端 `flash_attention_2`、实际模块路径、参数、全部设备为cuda:0、问题与图像SHA均已保存于 `qwen/author_fa2_smoke/`，源码未改。四题仅作运行检查，不报告准确率。容器仅使用固定镜像系统库及手工只读挂载的当前560.35.05驱动，不改宿主glibc/驱动/Docker配置。

### Qwen 公开入口的发布包装缺失与修复

首次完整 `run_eadp.sh` 尝试于18:09:32北京时间退出1，尚未生成任何预测：作者 `VLMEvalKit/run.py:49` 导入 `vlmeval.utils.result_transfer`，但 fresh Git tree 和原 amp 都没有这个文件。失败保存在 `qwen/public_fa2_full/`，不是 CUDA/模型/数据故障。

为继续原公开入口，在 `qwen/import_overlay/` 添加仅匹配该完整模块名的 Python 导入 hook；对应文件取自官方 VLMEvalKit commit `401916c00a4769884aeca865a77674abe0c62a26`，3496字节，SHA256 `d6225256ec0afa2ac9857b139df2f910b23b693749d2f2eb58cf7e696b9b6974`。原文件未修改；它只提供 MMMU/MMT 提交转换函数，DocVQA/AI2D 不调用。版本选择依据是当前作者 vendored 接口兼容：最新上游 helper 额外导入作者没有的 `warn_if_judge_model_changed`，其失败和官方文件历史同样保留，没有按预测/得分选择版本。作者所有 tracked 文件不改，EADP/生成/评分源码均不替换。CPU 运行原 `run.py --help` 已通过，见 `public_import_repair_cpu_probe_v2.json`。

第二次完整运行使用独立 `qwen/public_fa2_full_run02/`，按原公开脚本顺序运行 DocVQA5349 与 AI2D3088，K256/.5/2/greedy2048。计划 `public_fa2_full_run02_plan.json` 冻结所有运行源码、单模块 overlay、实际包版本、权重/输入证据；完整 transitive 包相同不作声称。未复用旧回答或 bank。原公开 AI2D 评分器可能先尝试缺省 API 匹配，容器没有网络及宿主API凭据；若评分失败，保留原始预测并明确执行 CPU 官方精确匹配，而不是重跑/替换生成。

整个公开序列退出后的CPU终端审计器已由根完整审读并启动，PID2471493/ticks1331849736/exec19609。脚本 `qwen/independent_audit/supervise_terminal_audit.py` SHA256 `7ded4f2e05923b53c10618704b558673f3c287e2a8c84f16443de9b819219bba`，准备记录SHA256 `2a52c673886038022f28ed62d6a8f5a45da82aa9edd41ba6f91375aaf1f9d2b6`；8项CPU控制门通过。它冻结原auditor、11项既有测试、empty policy和原generation plan，等完整controller/docker/容器init/两个native任务实际终端，只调用原CPU核查一次，不因Doc PID先退出就评分AI前缀。不新增模型/API或重跑生成。完整覆盖、合法空回答、原Noempty门、native/API评分状态分别报告；`complete`仅表示审计结束，不宣称paper成功。

### AI2D judge 配置与旧完整回答诊断

只用已合格的旧完整3088回答，released `can_infer`主分仍2302/3088=74.54663212435233，空回答/失败占位/native NA转换均0。482条不是单个有效选项字母，其中422可由原helper解析，60条未解析。若将60条全部判对，极保守上界为76.48963730569949，足以覆盖字面论文差1.153368点，但不能据此认定judge是根因。见 `qwen/independent_audit/old_ai2d_can_infer_judge_upper_bound.json`。论文显示75.7经一位小数四舍五入可对应2337/3088=75.680052（比旧值多35题），达到字面≥75.7才至少需多36题；两个概念分开。

实际源码的MCQ缺省alias为 `chatgpt-0125`，fresh作者HEAD的 `judge_util.py:21`却映射到 `deepseek/deepseek-chat`，文件SHA256 `3765164265540824bcf029a641347ade06d11776a241ecf2ff49afb5835c307e`。注释提到OpenRouter，不能以注释证明作者实际endpoint。`build_judge`后`working()`为假才退到exact matching，构建异常会被run.py捕获；逐项judge重试失败还存在随机选项回退。因此native launcher exit0并不证明评分完成或确定性。论文没有明确指定AI2D judge/endpoint，MM-Vet的GPT4说明不能移用；见 `author_ai2d_judge_source_audit.json`及保存的原matching prompt。没有调用API。

看过旧case之后触发的独立事后诊断发现：57条恰为 `LABEL. literal_option_text`，标签对应的canonical选项内容与后缀逐字相同，其中23条标签等于GT；另外3条是逗号多选。仅按明确标签解释这23条的辅助值为75.29145077720207，在“尊重明示标签及内容”的假设下将剩余3条全判对也只有75.38860103626943。该更紧范围有语义假设，不代替实际author judge，也不撤销一般60全对的宽上界；见 `old_ai2d_explicit_label_diagnostic.py/.json`。它不是预注册实验或新主评分协议，正式主分保持74.546632；所有这些CPU分析都没有读取新FA2未完成输出。

## 当前终端状态与后续核查（20:45更新）

- MMB：原代码、三wait、单beta三次新完整4377均完成。单beta61.394557823129254，作者转换/公开helper/独立评分/真实退出全通过；worker2514846、controller2514842、监督器2449729均退出，根exec8614 exit0。完整beta配对及V11四级实际输入匹配见前文，不能以全量分母不同归因题集。
- POPE：三次新完整8910均结束并独立核查；最新单alpha82.7060125212074仍低论文3.993987点。原worker及控制链均退出。另一次16图/192case原生公式facility诊断也已结束并独立核验，真实32保留集合不变，根exec49578 exit0，未生成答案或全量变体。
- NeXT：公开默认beta1完整5000为56.944，与paper59.2差−2.256，实际worker及监督器均退出。完整token重解码/独立逐题共识分和历史beta2配对均完成。历史FULL5000刚用fresh作者CLI与独立公式复核为60.370，paper60.3差+.070，不同一位小数；不是新FULL生成。所有限定见前文。
- Qwen：整个原生Doc→AI2D序列20:38:26退出0；CPU终端审计20:38:33完成，full_prediction_audit_passed=true，独立peer review通过。Doc5349原生61.166042248329354、独立61.166042248329276，paper差−1.633958；AI3088原生fallback及独立2296/3088=74.35233160621762，paper差−1.347668。两任务空答/失败0、完整有序ID/question/GT/options、原Noempty、全逐题分和38冻结SHA均通过。Doc native NA转换7；AI0、未解析60留全分母。AI日志保留15条API DNS连接失败记录，随后按image_mcq源码退回exact matching，原生acc及3088逐题hit均与独立can_infer一致；不声称论文API judge已复现。

原Qwen Doc2364896、AI2531101、controller2364766/docker2364783/容器init2364864及CPU审计器2471493均已实际退出。根exec43799/19609均返回0，证明在 `qwen/independent_audit/root_terminal_sessions_proof.json`；完整report SHA `57d75bc40e833544ceaf1b94214b559d850fd062e74b10c55717b02d2ae3863b`。原launcher state保持generation_launcher_exited_validation_pending，它不是活跃证明；最终CPU终端report提供正式覆盖/数值/退出门。20:41根统一ledger现9完整/0pending/0failed，goal_complete=false；查看表已同步，全部新旧结果保留，不选最高。

21:08根实查GPU空，Doc全量CPU核查已结束。两套当前真实栈各执行5349次完整processor（重复图重算），1286图和5349提示记录全部指纹相同，源七函数AST相同，独立review通过；没有加载模型/CUDA/FA2。`qwen/processor_input_identity/fullDoc_comparison.json` SHA38e88c560860828dbbf9ed5645fb7561186b1dc1f7cd03f3bad807035c34d03d。两worker2586906/ticks1332325407、2587361/ticks1332326881和比较器2591924/ticks1332348212均真实退出，exec10763/74641/25798均0，见CPU_execution_completion.json。当前qwen3vl_clean为torch2.10/TF4.57.6/Pillow12.2，qwen_author为2.8/4.57.1/Pillow12.0；这不是9月22日历史有效环境的还原，不建立FA2单因果。后续仍最多两个模型，加载计槽，并发运行不作独占效率数据。

官方Releases与Pull requests公开页20:36未给出新增运行recipe，补查记录在 `mmbench/public_history_audit/additional_public_recipe_surfaces.json`。用户明确没有额外作者命令或原始预测/日志，要求“继续依据公开材料排查”；这是当前公开证据识别边界，不是等待许可或结束任务的理由。

21:02完成新AI2D完整3088的固定旧规则辅助诊断，`qwen/independent_audit/new_ai2d_frozen_rule_diagnostic.json`：60未解析中的57个严格label+原文、26正确、另3未归类；主分74.352332不改，辅助75.194301、按标签解释且余3全对75.291451，无条件60全对上限76.295337。旧规则为事后提出，未升级为正式指标或真实judge；不能用条件上限证明任意judge不可能恢复论文75.7。

21:11完成历史Qwen FULL Doc5349源码资格和CPU重评分：94.483963，同论文94.5的一位小数；5349有序ID/question/GT、原log回答、原生detail及fresh作者/独立DP逐题相同，空答/失败0。原driver无剪枝段和唯一Qwen3VLChatFixedRes注册支持完整generate路径，但当时dirty源码/actual argv/权重SHA/token观察未知；见 `qwen/independent_audit/full_doc_baseline_recheck/`，独立review另查。它不是新FA2同环境配对，不代替剪枝62.8目标。

七项目标及剩余证据边界见 `remaining_work_audit/remaining_work_audit.json`。NeXT固定16图AnyRes中介诊断21:25完成全部320case、ragged0：1744数组有限、16native锚点、80identity、每步gain argmax/有效前缀/原source最终mask、12source+5实现+8引用+3额外source/20模型stat/16图SHA均独立核验通过，report SHA2f612bb931fb9874068533b89b6baaf91c2254d09bdbea538520690ec3696907。worker2633433/ticks1332520592已退、根exec34131实际0且GPU空，见 `next_textvqa/gpu_formula_facility_panel/root_terminal_process_proof.json`。

五既有recipe均无quota变化。Eq10五组均最终mask不变；beta2/CLI128一次内部maxT序列变化发生于丢弃后缀，不算模型输入变化。Eq6五组均6/16 importance变化，只有beta2/CLI128在1/16最终mask变：full position3632、crop3移除330/新增115，quota始终[146,162,106,130,93]总637。首次共享前缀后的第16次选择source144、variant142；另几组有顺序变化但最终mask相同。实际dense三段，新增成员跨三段，不单独归因某一段。该真实中介个案不证明完整5000精度或paper配方，未自动启动全量变体；16缓存526,948,467字节保存，详见该目录RESULT.md。plan6ea96586及root allocation b7935341保持原样。

21:47 MMB原生CLI0 FULL完整4377/1176已结束，757/1176=64.37074829931973，paper64.7差−.32925170068027；每题576全True mask、有限feature、完整input SHA/output重解码、原XLSX与两circular评分均通过，public解析变化/未解析0。根exec31413实际exit0，native2645294/ticks1332572280和controller2645284/ticks1332572209已退/GPU空，root proof d6de5ea7ecce86c90676dbea6962aacfcc146360c9d7d025dca1600d2ae4c869，独立report b4227aaec954a0b9aeaa640b334518d85b32fe4d2785f5a30f2ed9080a2d2252及peer0c4464d958d717da687a830429f1952271cf39e624b3c87a27204ab603291685。plan70911f00/50source/17stat、allocation556dc2e2不改。该新FULL用于校准，不覆盖EADP主分。

完整配对 `mmbench/full_vs_pruned_complete/comparison.json`：beta1→FULL恢复61/降低28组，净33，loss2.80612245点；beta2恢复60/降低25，净35，loss2.97619048。论文报告loss2.0，额外loss.80612245/.97619048；这是参照报告精度的数值分解，不是论文逐题因果。CLIP长度分层703行>77/最长582、198长组/978短组；beta1长净loss4/短29，beta2长6/短29。不能把多数短组损失归到长guidance处理；也不据此自动挑新变体。

21:35另完成NeXT三个历史小预算的15000逐题fresh CLI/独立闭式评分：beta2 CLI64=55.410、beta2 CLI32=52.086、beta1 CLI32=50.966，均保持，最大逐题差2.22e-16。完整有序ID/原prompt/GT/runtime/参数/source及raw SHA闭合，3166图manifest只与既有已核canonical SHA map比较，未声称再次全量上游下载/hash。新补门 `small_budget_historical_recheck/verification.json` SHAa058bf638281fd159a4f18948dc1e6aabbcc9093091510d7affc594c0e23e108；原始实验未改，不新增GPU或回填当前runtime。

21:53 Qwen Doc固定16/64case的第一次容器启动在torch/模型导入前失败，0/16：宿主driver原路径guard与容器别名mount不一致，FileNotFoundError。plan7701cfd2、claim/state/failure/log均保留于 `qwen/gpu_formula_facility_panel/`。根exec67790实际exit1，controller2679018/ticks1332712856/docker2679026/ticks1332712915均退、容器absent、GPU空，证明root_failed_terminal_process_proof.json。根已授权独立run02只处理路径/mount与跨namespace stat语义，先无GPU设备/无torch CPU容器preflight，再冻结/独立review；不改原公式/recipe/fixed16、不自动GPU重试。

21:59查看表新增独立四行FULL上下文及来源CSV，`docs/versions/v2/eadp_comparison_20261028/full_path_calibrations.{json,csv}`。原99格、29历史配对与9新EADP运行保留，FULL不凑入剪枝分数或平均分。

22:10新的Qwen固定16 run02已在B实际启动。plan d752c876367803c5728ffa84a819fd63b4c129cfd0e2eb78931f7d94dae5d4ad；原native/4公式主体AST与run01相同，仅修driver只读原路径和model resolved命名空间。真实CPU容器55SHA/4严格stat/16图/12bind来源及RW、无GPU设备/无torch全部通过，preflight e2fcb5f129483bfe86e7be74cda3eca0dd66970625f5092d23f4850f1b791a1a，独立review通过。根allocation d02442866a7ca42c4a873d79765e619e4a263cbd7b24353c95b13f13863e5e76；controller2701769/ticks1332811192、docker2701777/ticks1332811253、host init/GPU2701838/ticks1332811264，CID92466abf4d218029d0806ded3861e9cbda40665fe8806a4e1795e15ec209b8a8，rootexec5027。完整结果尚待，不改变主分或goal完成状态。

22:12 Qwen run02完整16题/64case结束，根exec5027实际exit0；controller2701769、docker2701777、host init2701838及容器均已退出/删除。根终端证明SHA03516e429818c7dc39a1db27e396b921ff8462ce4628dc64f7add23b4f6794ea；独立CPU终端核验一次exit0，report SHA3b9a0a0d582d972bfeccb4f765cd2754f2512c7ea43c1c5a985b419f2db61c46。16原生锚点/identity、64trace与最终空间mask、464有限数组、55SHA/4model stats/16图/12bind、官方BF16输入embedding实际token行及进程退出门均通过。实际FA2/BF16、generate调用0；16缓存202,125,255字节。Eq6在10/16题各新增1候选，但importance/有序256/最终集合16/16逐位不变；Eq10所有题importance微扰，最终集合仍全部不变。CPU参考新增权重8.0781e-19至3.6953e-9均为正常正FP32，不归因下溢/NaN；不能从CPU镜像断言GPU中间量首次抵消位置。Eq6精度边界报告e61c485f825486c5a8029840e15b974fb4fb3c1e02f4642335eeb5cbb9c519bf。该固定面板不代替全5349准确率，不支持自动全量公式变体。

22:20继续依据公开材料，独立准备NeXT CLI64（名义320）/.5/beta1/三wait完整5000，补齐已发布默认参数的中间预算覆盖。现有结构化元数据未找到同配方完整结果；已有beta2/CLI64=55.410、beta1/CLI32=50.966及beta1/CLI128=56.944均保留。新目录 `next_textvqa/full_default_K64_public/`，须根/独立审查及实际GPU分配后运行，不能将配额floor语义改成每题恰好320，也不能按得分挑配方。当前GPU已空，尚未启动新模型。

22:32 NeXT名义320/CLI64/.5/beta1公开默认完整5000已在空A槽启动，plan9ef9b4c69727d7baf324c5e3b3d93fd8c7d23a899d466f35e8c2389f05405d0d，allocation77602cb7ef1ffba87aee01bebbd2dbe54c5e02dcf7f3b890f81200a07c5fe204。原生成/评分体AST逆变换、7CPU门和独立peer通过；prelaunch两次验证器草稿及diff均保留，不是失败GPU运行。native2734206/ticks1332948051、controller2734202/ticks1332948026，根exec63618；实际argv/boot/cgroup见root_launch_process_snapshot.json。保存逐crop quota、实际mask indices和有效ordered prefix，所有新tensor读取在原generate返回后；不强制每题320，保留source min1可超名义的合法边界。全5000完成后才独立评分及CLI核验，控制器真实返回需根另证；不按分数重试。新full_run_ledger_v2承接旧九行逐项不改，当前9完整+1pending，原ledger/builder及99格comparison不动。

23:00 NeXT公开默认中间预算CLI64/.5/beta1完整5000已结束：作者54.90000000000014、独立精确54.9，paper57.0差−2.1点。四项5000门（输入SHA/实际quota-mask有效prefix/输出token重解码/逐题共识分）、原12source及新增11引用、model/image guards和fresh CLI54.90%全通过；空答/失败占位0，实际316/317/318/319分别211/2291/2274/224题。native2734206与controller2734202均真实退出，根exec63618实际exit0；rootproof100454d855cd853632a389693312cff5de9702fd5124a512a292e3ef3b16eae4，independent da9bc500921add4500a7d52063431b3df558e3d27a603c7016a60cf462762ede，raw27ac3f1a2665b29262db2903961bff1f0fb4c29c206594b83d55dfe078de937c。ledger-v2及查看表现10完整/0pending/0验证失败，原九行ledger/builder与99格comparison不改。完整配对β1−历史β2为−.51点：728原回答改变、147题分数提高/173降低；β1−历史FULL为−5.47点：307提高/601降低。按paper FULL60.3→57.0的loss3.3，本机60.37→54.9损失5.47，额外2.17只是数值分解，历史runtime差异仍不能称单beta因果，见K64_completed_pairing/comparison_report.json。公开默认三个预算现均有完整结果，七剪枝目标仍未复现。

23:01已在K64全终端通过且GPU空后分配A做一次窄单例缓存回放：known_mask_case_generation/，plan34b815cc26db836ad6bc9f7ea69eb08956680d350a86b8e7a697c904545a84c3，allocation7b9fecdb1a33d8a74c3e27704f800886fd5163c2ccbf8fa608a54a3b1fd0f078。只固定fullpos3632的beta2/CLI128 source-helper cache mask与已有Eq6 mask各一次nativegreedy128；同FP16投影缓存、原生prepare/generate body、637视觉行及非视觉区域逐byte/输出重解码门；原native特征锚点属beta1，不冒称beta2另有原生锚点。native2774847/ticks1333118719、controller2774786/ticks1333118631，根exec89061。该诊断无GT/历史答案/accuracy，缓存mask330→115导致原空间排序83行位移；结果不能外推5000分数，尚待真实退出和完整核验。

23:03单例缓存回放完整2次结束并通过终端CPU核验，rootexec89061实际exit0，controller2774786/native2774847均已退出/GPU空；rootproof77dca7df064b943cd61b309943800827c9dd1622362e7a2c126a69968c86a068，independent637a1adf82a3757dbe8b14f6e927dc5c5d4b4a608facf0ebce14cbe638f8ef7a。两臂实际原生prepare形成的637视觉行分别逐bit等于FP16缓存按原空间顺序重建的source/Eq6，视觉83行位移、非视觉区逐bit相同；每臂encode1/fullprepare1、CLIP/projector forward0。两条输出ID序列及解码文本全部相同，均129返回tokens（BOS1+128新token）且未含EOS2，故只覆盖所声明greedy128的长度范围，不声称自然终止或排除logit差异/更后续变化；无GT、准确率或新benchmark主行。独立peer终端另查。K64完整终端独立peer亦通过，报告SHA b0bb54e254fa5661d9e46f09deedfd955f1e90748194b689acd0d4ec0d959ad9；台账仍10完整/0pending/0failed、goal_complete=false。当前所有GPU任务结束，无自动重试/队列；公开默认预算覆盖齐，但七项主要剪枝差距仍未复现，不能以这些阴性结果宣称目标完成。

23:09单例回放的独立终端CPU审查通过，peer_terminal_review.json SHAa5fe64f95e337624bb48c1f8115f5ac1293c38d1905689aaeb88046b2fe5a1f6：独立显式坐标遍历重建2×2空间序列，逐bit核验两臂637视觉行、35+212非视觉行，129输出IDs完全相同、BOS1/EOS2由既有模型config与generation_config身份确认，均未EOS且仅128新token范围；12source/9冻结证据/20model stat/root终端链及实际PID/GPU退出全通过，未加载模型、调用原verify/评分或读取GT/历史答案。至此所有本轮已分配完整运行及诊断的终端门闭合；论文剪枝目标仍未达到，不能据无待运行任务标goal完成。

## 23:33 公开材料继续排查：新熵分母与问句指导边界

Qwen源码pruner.py:89的negative_entropy聚合分母为M_temp+1e−6，传入M_temp=.01；论文Eq7及附录默认gamma=.01，先前Eq6/Eq10并未测该项。新纯CPU固定16题真实缓存单项对照完整结束，原始缓存/source/guards保持。source CPU与原GPU importance并非逐bit相同（0/16，最大差2.2172928e−5），故该实验只给条件CPU参考：两CPU臂最终256空间集合全部相同，importance改变11/16，有序选择改变1/16但空间排序消去该变化。共同历史有效gain最大微扰.0026397705；整trace最大1e9是分歧后selected-mask sentinel位置差，不是数值溢出。worker2804985/ticks1333248456实际退出、exec77094实际exit0，comparison SHA feb116dec1014cd71fbcd74287d291b342d20b78ac14c6e74269aad5373d6afe，boundary aea067f8f0fea1f0dc0f6b57dfc135f9c10c82aeaa820ec881a47a684631fe37，CPU completion3ce0b01d1b669b83434dddccd80940ca73423d3d8fb6d9b500925d164080e4af。

后续GPU候选仅复用原16真实BF16features和FP32 cross/visual similarities，无8B/FA2/encoder/GEMM/LLM；每题source importance、有序256、整256步gain及最终空间BF16序列必须先逐bit重现，未通过就保存并明确variant未测，不调flag重试、不用CPU补数。原GPU未记录全部cuDNN flags，新默认完整披露；严格source锚点约束这一缺失。当前仅准备 `qwen/eq7_aggregation_denominator_gpu_cache/`，根及独立审查后才分配GPU。

TextVQA公开model_vqa_loader.py:113保留question.split("\nReference OCR token")[0]的inactive注释。先前64题valid-dense/bounded-chunk四臂都没有测此问句边界。真实native CLIPTokenizerFast离线全5000分词闭合原full指导IDs/attention哈希，881题>77、最长390；question-only全部非空、5–41 tokens、均单段。报告 `next_textvqa/guidance_question_boundary_exact_tokenization.json` SHA fcab501744bb1cbda1fc46aaae288f59f98d8d49881b3be27255573b21af43b6；不改LLM完整OCR提示，不读GT/评分/模型。旧16缓存全部schema没有CLIP image_embeds，不能用投影后features造出新crosssim，也不能替换baseCLIP vision。只准备fresh native固定原16/K64/.5/1两臂窄诊断，source先锚定旧nativefeatures、新完整K64实际mask/配额/prefix与source原回答一致性后再variant。该输入假设有源码注释依据，论文不唯一规定，不宣称bug、paper recipe或全量精度。

POPE历史wrapper专门核查进一步找到运行时绑定链：旧completed protocol/BASE_PROTOCOL source map绑定helper24d712b8…，旧worker在runpy前校验，recovery在Popen前校验worker，实际stdout记录三wait AST d7dece035…。其数据/提示/生成主体与fresh作者AST一致，区别只有明确禁用AnchorZip和观测代码，故无证据仅为wrapper不同而重跑8910。其余五个核心文件只确认当前字节一致，旧时实际dtype/backend/包/继承环境未闭合；不把未知当错误，不宣称整历史runtime已重建。新audit SHA899ac2f330d7d0a91e13f81fbd5816a32cc31c74c49072504610dbeb8598dce8，未读预测/评分/加载模型。

上述诊断均独立于10条完整benchmark主台账；未增加论文配对结果，七剪枝目标仍未复现。23:33实查GPU空，两个新候选尚无GPU分配、无队列/自动重试。

23:43 Qwen Eq7GPU缓存回放完整结束：final plan4a2bbbc8b484cf7660ef1bfdcf38107262175425567e5eed3f22eeee783d811d，根allocationf0fa25abf761b9f15003812f515b85fc0c8748d0ecba59f74f2532cd2266da4a。5CPU门与独立peer通过，source四门采用dtype/shape+原始byte（拒绝signed-zero），不是仅torch.equal。16source四门全部匹配旧nativeGPU，16variant全部实测，0未测；weights16/local13/importance12/gain12有微扰，最大importance9.4771385e−6、共同历史有效gain最大.002593994140625，有序256/最终集合/最终BF16序列均全部不变。CPU有序1题变化不能移作GPU事实。独立NumPy终端报告193469c9afae8e2a087b7344e8d961850c943cd8700c18e407f9ca8b43a5a343，288有限数组、16NPZ共57,728,997字节；独立peer另外重核51SHA及固定第0/7/15题source四byte锚点和最终两臂序列。根exec27423实际exit0、controller2832239/ticks1333367682与worker2832246/ticks1333367725均退，rootproof5e4af0e00f4f906bf0f8ae7895762a6ca5aab22b68fb545e602e76c26a212db1；无LLM/GT/评分/新benchmark行，不据面板全量跑Eq7。

23:45在Qwen实际结束后，空A槽单次启动NeXT固定原16/K64/.5/1/3wait问句指导两臂，finalplan812c39f98bb238c0f8bf513c0b730fb85a41c5ee3603510f82b2db16592dfe52，allocation5c220e8675182c3442cfb85292cf599cbd8267d66d761e9ab4333953c4da7ec3。9CPU门及根/peer完整review通过；postgen锚点前先保存NPZ与未验证attempt，失败侧车引用后停止，不丢已测数据。根exec76915，controller2837356/ticks1333382280、native2837368/ticks1333382361；所有source16先通过才测variant。完整32与CPU门/真实退出仍待，不读GT/算精度。并行仅准备相同K64公开注释指导边界的全5000候选，须当前16终端成功且输入确实改变后另审/另分配，不自动队列、多预算或按分数选配方。

23:48 NeXT问句指导32case完整结束，完整CPU门通过：source16旧features/importance及fullK64实际selection/output全部闭合；两臂真实native projected/image_embeds/visualsim/pixels/完整LLM IDs及非视觉FP16部分逐byte一致。问句指导的配额/最终mask/有效prefix各16/16改变，输出IDs和原文本各2/16改变；两臂各16/16均有EOS。所有32保存真实输入embedding并按2×2空间序列重建核对；文本是worker原生tokenizer解码，独立CPU门只核ID/记录，没有另行tokenizer重解码。根exec76915实际exit0、controller2837356/ticks1333382280与native2837368/ticks1333382361均退/GPU空；rootproof272ae626ed2bc36b03ea69d38bb21c9f11a6cb9a82df4b0f87f297693a23e356，reportf9d9076891883ea98d8569d9907e43e056a34303211339fe8fd6728c8ee8479b，release0bae00bcf12b649101e6528ac7e706f2b7ac7f5ae73f6a8898d732b070ca66f6。无GT/准确率，不据两题文本判断增益。现仅准备 `full_K64_question_only_public_comment/` 全5000同参数单项指导对照，公开注释支持的假设而非已知paper argv；必须独立terminal peer与新候选根/peer审读后另分配，不自动启动、多预算或score-based调参。

10-29 00:02北京时间（10-28 16:02UTC）唯一全5000问句指导假设已在空A槽实际启动，目录 `next_textvqa/full_K64_question_only_public_comment/`，finalplan3a1bfff5c78c47d7a0676f471eadcbff89aa3b98e9407132ffeaece3dbdce7e2，root allocationbdd8872140761e1bc42d7e2fda233a122461914aee2d6373837c3808561f7ebc。根完整读8实现/原K64与BASE、5CPU门与独立peer通过。只启用source inactive split，逆AST恢复全部原入口，运行时actual eval codeSHA/patched AST与原entry sourceSHA分开；每题实际texts SHA对5000离线refs，完整LLM OCR及IDs、images、参数与数值算法/评分保持。新verifier的GT字段限定本verifier的额外指导门；worker每题指导门在5000生成及首轮官方评分之前已逐题执行，不声称全pipeline在独立复核前从未读GT。

controller2861700/ticks1333486787、native2861704/ticks1333486819（boot1b0bf851-085c-43c5-9855-503a2909543b），root exec60045活跃。启动snapshot已保存；00:04实查正常生成runtime382/5000，GPU21562MiB、仅此一模型。当前真实全量状态10complete+1pending；原v2 ledger仍是此前10行快照，另备v3承接，不拿旧0pending掩盖活动实验。无前缀评分/新指标/自动重试/多预算，官方及独立全5000/CLI/输入/真实worker与controller退出门完整后才入新数值。已有32case独立peer已读448数组、逐byte显式坐标重建32真实输入并闭合所有source/非视觉/身份门，不能将2题输出变化当精度收益。

新研究汇总 `remaining_work_audit/progress_addendum_20261028_1550.{json,md}` 是15:50UTC历史快照，四immutable已各一次核验未变；后续实际运行以root_launch_process_snapshot和实时进程为准。七目标仍未复现，目标保持active。

00:10 v3接入已实际执行pending导出，不读未完成新raw/score；根完整代码审读与独立peer通过。`build_full_run_ledger_v3.py` SHA166d3ee00a5ad910d75771377048d52bb2e7cef15209c0600f2bce2fd63d8ec7，ledger-v3 JSON7ab990ebbe9d6b083bd99b1c1b91c83678cd4410dd89656baaaa4522fe77474b，10complete/1pending/0failed，原10deepcopy/序列化及六immutable SHA保持。renderer最小增v3优先和input-hypothesis条件标签，当前表29历史+10完整+1pending、JSONf4ccafa836c64da620834e1088530412d20b16f57845311a7a44ef630fb70583；原comparison不改，差值/行数/非缺失计数检查及git diff --check通过。00:11 full1761/5000 runtime/guidance一致，无失败，GPU21654MiB。根PTY60045仍活跃，当前无新分数。

下一步必须等完整controller/内置终端门、根PTY60045实际exit0与原controller2861700/native2861704退出，然后rootproof绑定全部身份和plan/allocation/release/independent SHA，再独立terminal peer、重建v3和current表。双侧complete-only配对候选 `question_only_completed_pairing/pair_completed.py` a6c1b82bba29f384640be2a5762d0337c75f0d64096d420b84d324e9ebe2e384 已有6CPU门与真实旧侧metadata-only一次通过；根读过，末peer另查。未运行真实配对、未读新前缀或重新评测，所有评分正负均保留。


10-29 00:41（UTC10-28 16:41）全5000问句指导已完成并正式登记第11条：独立精确54.958%，原完整OCR指导54.900%，变化+.058点；paper57.0，差−2.042点，未复现。全部5000实际guidance SHA/完整LLM IDs/预算mask有效prefix/输出重解码/逐题官方与独立分数，以及源码/模型/输入门通过；原生54.95800000000012，freshCLI54.96%，空答/失败占位0。4986条含EOS2，14条仅达到128新token范围且未EOS，全部保留分母。根PTY60045实际exit0，原controller2861700/ticks1333486787、native2861704/ticks1333486819均退/GPU空；rootproof a0e9c852029c96099021328d69d1cf8ab2f62120ef6b50268344c0b150c46925，independent1f66d83e5e213db974d297d2bb402dc7283130dd17bbb3ad1583d81124af05ed。独立终端peer另核全5000记录和已接受分数通过，无重生成/重评分。

complete-only配对脚本已实际exit0：两端5000有序ID/fullOCR prompt/image/LLM IDs及记录runtime一致；684原回答/689输出IDs改变，151题提高/148降低/4701不变，quota4999、总保留数2901、最终mask/有效prefix5000改变。既有逐题分整数tenths净29，重构+.058点；预定881长题贡献+.048、4119短题贡献+.010，覆盖全部，无结果选样；不单独识别OCR语义与分段暴露。配对报告419a3c91bc10e7d348547e14b69ca49fcc1bd8f250fec5c107c751adae682c6b。v3 complete分支及renderer已实际执行，11complete/0pending/0failed；ledger1744fb69f5a42660918f12eb49ed1daf342ef2fa226ef5d10db189ed6313ff03，current表1d4698fc3c80f39610477d7436810c7ff58213ecc10158c1f1ad447d6337f7dd，99paper cells/29historical pairs/11new complete，原10行及两代旧ledger/builder不改。详见full_K64_question_only_public_comment/RESULT.md。

10-29 00:40另完成唯一POPE固定原16缓存q=.2→.3机制对照，alpha.5/beta1/K32保持。公开消融LLaVA1.5-7B/K128支持此单点机制假设，不确认主表q；不同于先前Eq6的15→16边界，本次原floor/topk实际15→23。finalplan668b9a5f644eec4019cb24e7b368a076b558421678c9109ad7dc60da9b492a4d，allocation2a1a15c9df81ee636b235ce11a9ec5a26f3de786d8a624f97dbd995884c647e4；5CPU门、根与独立peer实现审查通过后单次A启动。source importance/有序32/完整32gain/mask/派生FP16空间序列16/16逐byte通过，16variant实测0未测。local4/importance3/gain3变化，最大importance4.7236681e−5、共同历史gain.00408935546875；有序选择/最终空间集合/派生序列均0/16变化。原native锚点是1/1，目标.5/1是同次真实GPU helper；序列仅cache无损推导，非实际LLM prepare捕获。没有GT/答案/LLM/模型/精度；320有限数组、16NPZ共9,996,121字节，独立NumPy报告3c092c8a26fd71cd75d49fb1ffd3728c3e456968216f6f1361efc4f817ba6a07。根PTY76513实际exit0，controller2913858/ticks1333710845、worker2913862/ticks1333710874均退/GPU空，rootproof6803dea9866edcf463449372834e44db00467ee45d06145521fc83c7b1e6e52d。该面板不足以支持全量q=.3准确率运行，不据此声称全数据无效或paper配方已恢复；peer终审另查，详见pope/q03_fixed_cache_mechanism/RESULT.md。

截至本次更新无活动GPU、无pending完整实验、无自动队列；七主要剪枝目标仍未复现，goal保持active。新的剩余研究判断只考虑尚未检验且有公开实质依据的假设，不重复已否定的微小epsilon或按得分调参。上述完成事实取代00:11仍活跃的历史快照，latest_root_process_snapshot.json已同步。

00:44 q03独立peer终审完成，另读全部16新cache与80项源byte锚点、320有限数组及真实退出链，PASS；peer_terminal_review.json SHA0acf481de241cea9765d50bcd06edeb82a07eaeb45ceb1abc18279ec80c26456，结果保持local4/importance3/gain3变化、最终token集合0/16变化。对已记录两臂importance/gain进行完整独立重算统计，不执行selector/模型/评分；未保存聚合softmax权重，不能据此指定首次舍入抵消位置。

00:54 新一手CDPruner核查完成，官方repo9541616c40fcd5625de1cdb8ea6c33c129eb7864在mm_projector之后计算DPP余弦，原论文§3.1定义Hv=g(fv(X))且§3.2沿用；CLIP visual_projection是独立的crossmodal路线。其§3.3还为无配对text encoder的Qwen2.5明确使用projector输出与LLM instruction embedding mean，提供前作依据但不等于Qwen3主表身份。故CD引用不支持把EADP post-projector geometry叫错误或声称缺adapter；EADP自己的符号/图仍可提出LLaVA1024/4096单项假设，却未唯一指定层与维度。目前不优先跑几何变体。新证据geometry_public_source_audit/geometry_source_evidence_20261029.json SHA510a9ba8a8e371b7b3583591a987408da16670e169be61860cf3c273f53df8f8。

POPE有界查96相关JSON、149个显式alpha/beta对象和83段argv，未找到α1/β2/K32完整8910或cachecase；原16只有1/1、.5/1、.5/2。该范围之外不宣称绝对不存在。报告pope/alpha1_beta2_existing_metadata_audit_20261029.json SHA6707e7783c5bd3bf0418000ecadb3748c69341deb82449559594ffcdf25a2da5。作者POPE shell保持α1，paper method严格β>1且已发表消融含β2，为一个固定β1→2全局重要度极化对照提供依据；这不是主表参数推定，也不把历史.5/2未闭合runtime与三分数当完整2×2因果。

根已授权仅CPU准备pope/alpha1_beta2_native_panel/，固定原16、sourceα1β1与variantα1β2，其余q.2/K32/三wait/greedy128不改，直接原生prepare/generate32次，先全source16通过再测variant。锚定旧native16真实特征/importance/IDs/gains/mask与完整1/1 benchmark原文本；历史8910未保存输出token IDs，不造旧IDs门。新两臂完整实际input/output IDs与LLM embeddings/空间序列都记录，采集后先保留工件再判断锚点；无GT/评分/挑样/自动重试。当前仍无该候选GPU分配、模型/队列或新结果，最终冻结/根与独立peer审查后才另分配。此项比已否定的微小数值改动有更明确的非恒等数学机制，仍不得预言它能恢复论文值。

01:05 已在空A槽实际启动POPE固定原16 native32的α1/β1→β2单项对照。finalplan404ee2ad79241f91b8c392e69b3958fe096727e84fe9a1eaa6a2cd1a9d3c6605，rootallocation49408e6a4449448e00941c5d20c7f7b825cc823ff47e6b423b76809c2324903d；7CPU门、根全文8实现和独立peer均通过。根PTY98439活跃，controller2949970/ticks1333861741、native2949974/ticks1333861772，同boot1b0bf851-085c-43c5-9855-503a2909543b；实际argv/cgroup启动snapshot897e3c306900e664040d8dae23ba08791a795be5a4458e2cc93ea943412d2fd9。原generate/prepare/encode/cross数学体保持；greedy只在原selected-mask赋值后、argmax前append total_gain引用，逆AST恢复原方法，全部tensor内容读取在generate后。实际self alpha/beta/K逐次记录；每次先存NPZ+rawattempt再验锚，source16全过才测beta2。历史原文本仅作源一致性anchor，明确不是GT或历史token IDs；所有输出保留，不看前缀精度。真实native和controller退出及独立CPU门仍待。该诊断不新增主台账行，完整实验仍11complete/0pending/0failed。

01:08 native32已完整通过终端核验并真正退出：source16旧native/cache9项及历史原文本门全过，随后variant16全测。beta2导致importance/gain/有效prefix/mask/实际送入LLM的FP16视觉序列各16/16改变，quota保持32，完整LLM IDs/图像与文本features/similarities/pixels和非视觉embedding逐byte相同；输出IDs/原文本1/16改变（原固定index6/qid2167），没有GT/精度方向。全部32返回3IDs=BOS1+2新token且均EOS2，不是截断后恰好相同。32NPZ269,020,894B/480有限数组，独立CPU报告2342e6b9cae6cb19f2056ace0f24fe618ea1254f93413a78f5f51c93c114069f，release849b8422c5ae842c036ecda58d2457f26de6a05fb2928de94dce668d8b38adc7。根PTY98439实际exit0，原controller2949970/native2949974均退/GPU空，rootproof2c5c74592cdd58e44c754f994936033ce6315d56f10cb5c4c8f3766eb9d8a1f9；独立peer终审另查。该机制有真实下游传播但不能据1题输出判断增益；不加benchmark行。

并行仅CPU准备pope/full_alpha1_beta2_single_control/，唯一同α1/β2/K32/q.2/三wait/greedy128全8910单点验证，不能称paperargv/repair或完整2×2。须native32完整终端与peer及新候选根/peer审通过，另行根分配后才可实际运行；尚无GPU/队列/新分数。不按16题对错/子集准确率挑β。完整实验仍11complete/0pending/0failed，七主要目标仍未复现。latest snapshot已将本次diagnostic从active转complete，详见alpha1_beta2_native_panel/RESULT.md。

01:12 native32独立全32终审另通过，peer_terminal_review.json SHA0a7356219219b4d126cfc92f0e79cc4c1902a5b6f5b8658d4a643415253a487b：144项source byte锚、16历史原文本唯一键、480有限数组、两臂全部实际32视觉/非视觉与LLM IDs、根PTY98439及原boot身份退出均重核。唯一qid2167的原回答Yes→No只记传播，不作GT/增益结论。新全8910固定α1β2候选的前序终端门已满足，目前仍CPU准备、无新GPU/队列。


2026-10-30恢复核查：10-29 01:21 POPE全8910 α1/β2首次启动在worker的eval_model marshal SHA身份断言处失败；没有进入模型加载，0预测/0运行记录/未评分。根PTY48822实际exit1，controller2972267/ticks1333959385和worker2972278/ticks1333959402均已退出；当前GPU空。原计划c9ef04605a42f523ea0c4b48eaed2a56bc297aa89206525590e09729bd5b275b、失败/日志/claim原样保留，root_failed_terminal_process_proof.json SHA780c3f08b17e9728f7f334830f891834b7654351c622b8c2b4b698d75576f986。旧v3台账仍11完整、没有新benchmark分数；其0验证失败不等于没有此次失败启动。

随后已有CPU probe实际用runpy载入入口并比较代码对象，actual与同进程重新compile后的对象相等、字段无差异，但原冻结plan的marshal SHA仍不同；probe文件pope/full_alpha1_beta2_single_control_code_identity_CPU_probe.json SHA07c72febb79236d29f204a65a987bf383b9ab5e374efd03ce97f8dc9bc414e9a。这表明需查清身份校验在不同上下文的稳定性，不能据失败推断作者数学代码错误。现仅准备独立run02，修身份校验并做实际无模型CPU预检；保持α1/β2/K32、源代码、题集、评分和原失败记录，未重新分配GPU或启动生成。


2026-10-30 13:48北京时间：POPE α1/β2完整8910的独立run02已在空GPU单次启动，原先失败的run01完整保留。独立CPU核查发现相同eval_model代码对象仅因父代码对象生命周期改变，marshal v4哈希即可变化；原失败实际哈希未记录，不能确定其具体引用状态。run02用已锁定源码SHA、同进程runpy与源码重新编译的code结构相等、精确路径及稳定递归字段fingerprint校验，保留实际marshal值作元数据。两个独立CPU进程的真实runpy检查通过并拒绝代码、常量及路径篡改；6项CPU检查与独立源码review通过；原作者生成、α1/β2/K32/q.2/三wait、题集和评分不变。

run02 plan 4704a777360b042b4d1e63dd9be4458f50d37a0234bfe09d4169e675e268ea5b；allocation a9f6d6d3f87dc86dbf530388f110fa72c2529381b1543fb910f75ebab946cf0f；root PTY23602，controller1790922/ticks1347080289、native1790926/ticks1347080307，同boot1b0bf851-085c-43c5-9855-503a2909543b，启动snapshot2284f1a80bb97a1bbe5d96e0555c4c04f9680abe28514b5de10b257c0e7262d1。全部8910题实际生成和native进程退出之后才运行完整输入/输出验证与原POPE评分，未将前缀当分数。当前实际状态11完整+1运行，旧v3仍为先前11完整快照。完整退出证据后再入主台账；该单点配置未确认是论文主表参数。

2026-10-30 13:50实际生成中，predictions与runtime均644/8910，未评分前缀。独立v4 builder已在根完整审读和metadata tamper测试后执行pending分支，11完整/1pending/0验证失败，原v3完整11行及旧ledger不变。v4 JSON SHA3af27f0b8fedaf604aef85891a0eef9a02903bae405a5e4af9bd409a42cc5136；current_comparison已同步11完整+1pending，JSON SHA66ca6649e7b18eedffd1b602c0ea7396271319da26f9cb90b4f78ab9f4e9c96e。renderer仅增加v4优先和固定β假设标签，不提前登记新数值。

## 完成边界

当前目标未完成。每项必须有全量新生成/评分、明确的来源与运行配置，并对大差距给出实验证据支持的解释或解决；仅“与源码一致”“审计通过”“分数更接近”都不足以宣布整体复现成功。不得用 native 修复后的 Qwen 数字代替论文的 legacy 剪枝路径，或把代码修复增益当作方法创新。

能解释一项差距但尚未实际复现的项目仍记为未完成。若最终需要作者未公开的主表参数/预测/环境信息，应列出具体缺失，不用解释或阴性实验改写实际复现目标。

2026-10-30 14:06（北京时间）：POPE α1/β2/K32/q.2 全8910单点已真正完整结束，root PTY23602实际exit0，原controller1790922/ticks1347080289与native1790926/ticks1347080307均退、GPU空；rootproof `bb41d5a3c87a417ac7c316659a8d914ac477e47565e94d6b53fb1d861a2695da`，独立验证 `98b499f25acc86fb91c6c670962446e890c8480c9afc91b31a8adb3a9deef342`。8910条actual guidance/input IDs/output重解码/mask与预算均通过，16个预定native β2锚点全过；全部EOS、空答0、重复输入yes/no冲突0。完整生成后原POPE官方和独立评分一致，macro F1 84.278884105%，论文86.7%，本机−论文 -2.421115895个百分点；α1/β1完整基线 82.220236999%，本次相对基线 +2.058647106个百分点，依然未复现论文值。作者主表argv未确认，单点对照不证明论文配置。

完成后配对8910条，442条yes/no变化（Yes→No 109，No→Yes 333），旧全量runtime不具逐字段闭合，不能单凭全量配对宣称β因果；原native固定16面板已证实β变化可传到实际视觉输入。配对JSON `40cbe8c22d06efffa3a1cdbf3400ea2d2820ab9cefeffe699def625ecc48c4e1`；先前配对脚本对旧摘要字段来源的问题已修正，7项测试通过。v4完整台账12 complete/0 pending/0 verification failed，SHA `e90aebfce9d1723002f5275c031e9fb283388d1fc4e89c80767df571606fdaf2`；当前对照99论文格、29历史配对、12完整新运行，JSON SHA `df1e628d6972116a3b61ee3c790346eccb04079d281506848e225f4fc44c0797`。旧run01失败和旧台账不覆盖，GPU无活动、七主要剪枝目标仍未复现。详见 `reproduction/20261028/pope/full_alpha1_beta2_single_control_run02/RESULT.md`。

2026-10-30继续Qwen完整路径调查：公开注册仅K128/256/512；K1024须显式构造诊断实例，不是公开主表命令。原pruner在K≥N直接返回特征，而EADP类仍走inputs_embeds生成，旧plain FixedRes FULL走原生多模态generate；历史94.483963不能当同环境同wrapper FULL。CPU已对原固定16真实BF16特征调用原pruner.forward(K1024)，16/16全部1024行按原顺序逐字节保留，选择相关调用0；无模型/生成/GT/评分，不能证明完整模型路径可运行。脚本 `reproduction/20261028/qwen/full_route_bypass_cpu_audit_20261030/audit_bypass.py` SHA `a709474c64c8be5b92b9ede146e992fa54a84af74485a86c1f40d8e76d8f6857`，报告SHA `94873718e16369563ee3f436a81aa3ff66a24355171493a89780188ff33060c7`，worker1830061/ticks1347240946真实退出、exec12270 exit0。正在仅CPU准备同实例的原生FULL vs EADP K1024固定16路径诊断；尚未分配GPU或产生该诊断的生成结果。其observer/终端和容器预检必须审完，当前草稿含待修的输出保全问题，不能启动。

2026-10-30 14时（北京时间）：Qwen固定16双FULL路径诊断在8项CPU检查、无GPU设备容器实际预检及独立源码review后单次启动。plan5fbf6c3c4355d55a26fe2f2ff8395a47bb6d39e14ed35046214f6a0459060e0c；allocation54b08f53e612112d893b60a214d4336f610e417540ffaa066356a1e8aa842e3b；root exec69304，controller1841880/ticks1347285362、docker1841884/ticks1347285421、容器GPU host init1841951/ticks1347285434；根启动snapshot c26d91eade55c0635059fae8ffa836f29ee1a1f09b10a1e7fe7f1137bf2d3e1c。仅一个模型、同题同FA2、plain原始多模态vs显式EADP K1024 bypass各16次；不读取GT、不评分，不是主表benchmark或作者注册配置。当前尚无完整诊断结果；32次运行、原进程及实际根退出和CPU终端核验需闭合后才能报告结果。observer clone有显存/时序开销，路径差异不可单独归因DeepStack或mRoPE。

2026-10-30 Qwen固定16两完整视觉路径诊断已完整32次生成并终端核验：root PTY69304实际exit0，controller1841880、docker1841884、容器native1841951原身份均退/GPU空；rootproof `400669950aa064a2ce2f6de80c21af9cd0bc5451e75a867a002b9ab1a1632a25`，CPU terminal `83b2e2fabbf8c946ab3ccbde0dc65294bc72d760c1b6f2acaf655f9f790501e7`。同FA2/模型/processor、无GT评分：两臂visual features及prefill embedding/attention mask均16/16逐byte相同、旧真实feature锚16/16；原生与EADP1024 position IDs 16/16不同、原生每题有visual mask和三组DeepStack而EADP均无，3/16输出IDs及原native回答改变（其余13相同）。说明无token裁剪时路径差异能传到输出；不推断精度方向或单独归因DeepStack/mRoPE；输出未独立重解码，固定16不能外推全数据。K1024不是作者注册配置。详见 `reproduction/20261028/qwen/full1024_two_path_panel/RESULT.md`。七个剪枝目标仍未复现，完整5349同环境校准尚未运行。

Qwen固定16双路径独立只读终审通过：独立逐一解压32份NPZ并核672数组、真实generate输入/输出suffix、30冻结SHA、根退出链，旧visual BF16锚16/16一致。视觉塔自身三层DeepStack输出两臂16/16一致；差别是实际LLM prefill仅原生臂接收DeepStack及visual mask。两臂共32条均含EOS、未达2048上限；回答只核原记录，未独立tokenizer重解码。无GT/评分，不归因单个机制。

2026-10-30 14时（北京时间）：Qwen DocVQA完整5349双FULL路径校准在9项CPU检查、合成作者评分验证、无设备容器预检及独立源码审查后单次启动。plan5e810f360b2b5169f7f010b8be9b436211e9c07b1f3d3ea3f2ce10cc33585bd0，allocationaf60d1fa30b766614d1c5f356095fd0c0c195531df8353c682be05b6c4b67aee，root exec16759；controller1868790/ticks1347400567、docker1868815/ticks1347400665、容器native1868875/ticks1347400677，启动snapshot467b880f80bcccb16888fd1eaf45bf99dcf108bd638e7efd305d2f4b48ce1f5c。一个同FA2模型每题两次原生生成：plain多模态FULL与显式非注册EADP K1024全部token路径；共10698次，全部结束/独立重解码/native与根exec实际退出后才单独CPU评分，原官方与独立ANLS均保留。准备时读取含答案canonical但未复制到清单或用于选择；生成/评分前不读GT。当前尚无新分数、无自动队列/重试。须持续保留根exec并用write_stdin等待actual exit，终端证明要绑定三角色原身份及launch/state/plan/allocation/release/verification SHA；单独评分gate还需原始预测与观察SHA。七项主要剪枝目标仍未复现。

2026-10-30 16:56北京时间：Qwen Doc双FULL路径校准5349题/10698次生成完整结束。根PTY16759实际exit0，原controller1868790、docker1868815、native1868875原身份均退/GPU空；10698实际processor身份和输出token独立重解码全部通过，rootproof `c6f7a6ab10f27e2d085ec1a7e80cbd5a31da620be65209cf5146e95e50851571`。全部生成与根退出后，CPU作者官方与独立ANLS评分exec22774实际exit0，scoreproof `1b26e648136fe0862d5a0d84c28efc07b4126d0f7bc9416d6f47a3b6de4f8bbc`，report `0b7843b902bc0fd50b05abc75791796373da5b3f4330387af5697ff18d0d37e0`：plain FULL 94.481578011（paper FULL94.5差-0.018421989），EADP非注册K1024全部视觉token 87.825405943，后者−plain -6.656172068点。1189/5349回答改变，逐题ANLS升132/降728/不变4489，空答0，pandas NA转换6/5；全分母保留。两路线综合差异已真实影响完整分数，但不单独归因position/DeepStack。只固定16捕获到真实位置与视觉注入差异，不外推为每一全量题实测细节。K1024是路径校准，不登记成论文EADP剪枝行、不代表paper argv；K256新61.166042对论文62.8仍−1.633958，七项目标仍未复现。详见 `reproduction/20261028/qwen/doc_full_two_routes/RESULT.md`。

Qwen Doc5349双FULL独立终审通过：只读逐行核10698输入与保存预测、作者XLSX与独立逐题ANLS、5349配对及22冻结SHA/全部退出链；两臂各5349输出均含EOS、0截断及0空答，作者Excel NA转换6/5均保留分母。本复核不重跑模型、tokenizer或评分，独立tokenizer重解码由既有完整CPU验证完成。

2026-10-30旧Qwen消融出处复核补正：旧 `audit_base_gap_20261003/archive_recovery/20261003_r2/ablation_inventory.md` 将E0 a1/a2/b2称clean 2×2并写反部分轴；原 `EADP/Qwen_vl/scripts/e0/e0_accuracy.py` 实际只有(1D,DSoff)/(1D,DSon)/(3D,DSon)，缺(3D,DSoff)，a2−a1为固定1D的DS差、b2−a2为固定DSon的位置差。旧D1 verdict与对应headline方向不一致，不作为定量因果结论。旧M13在3D下有FULL K1024的DS单因素DEV实验，及N2固定16 logit差，因旧自定义引擎/环境与已知记录限制，不能量化当前同FA2全5349路径差；旧E0 DSoff的n_vis_kept=0是统计依赖DS mask的计数问题，不能解读成无视觉输入。新双FULL校准差−6.656172是两整条路径共同变化，仍需当前环境固定原16分角机制诊断；不冒称此前未研究DS或将实现修复当方法创新。

2026-10-30同FA2固定16 DeepStack两路径机制诊断单次启动：先32个原端点逐byte锚定全部通过再测32个prefill DS开关；plan 255e5a7dc9d97de0c597e8220d2cd7e1c1babe54aa1f1b26d90f413d30c91e76，allocation 732d2293318d6924016fe889be696f5f80d07c7153230f909db020c0943b9f14，root exec71385，controller2081967/ticks1348317746、docker2081974/ticks1348317999、native2082035/ticks1348318009，launch snapshot 3d7086be4dc195b166764e364cdac0613075b45db2504922de4882e31b0207b2。6合同+7终端测试、作者torch2.8真实CPU hooks及无设备容器预检、独立review通过；无GT/评分，不是paper主表，双原生API不同，不能称完全隔离位置因果。真实64次/端点/后续同前缀比较及根进程退出仍待核验。

2026-10-30固定16 Qwen同路径DeepStack开关诊断已完成64次：先32端点全与此前原生/legacy记录逐byte吻合，再32个真实prefill干预；CPU核3378数组、实际kwargs/DS/mask与同生成前缀续接。plain FULL关闭DS使2/16输出IDs及原回答改变；EADP K1024开启DS亦2/16改变。共享前缀内分别87/89续接步骤的position/cache/rope均0差，分叉后不当同条件；64条均EOS，无GT/评分，输出文本未独立重新解码。rootPTY71385实际exit0，controller2081967/docker2081974/native2082035原身份均退/GPU空；rootproof `fac478dc842c870ae9f2f360441d5880146b425984c2f07d82a1e04087a851ee`，CPU report `115e4828cae49329f7dc7f83e7f7d5d882bd20c7b38b036191fcade73622209a`。两条API还含位置与其他差异，不称完整位置单因素因果；16例不估算全量6.656分归因，不替代论文剪枝结果。详见 `reproduction/20261028/qwen/fixed16_ds_factorial/RESULT.md`。

固定16 DeepStack开关独立只读终审通过：另一CPU进程exec81888 exit0，另解压64个新NPZ校验3378数组、32旧端点与真实mask/三层视觉DS、84冻结SHA及根退出链。plain关闭DS改变position2507、3510两题；EADP1024开启DS改变1504、2507两题。共享输出前缀内累计176个续接步骤position/cache/rope一致；3个变化案例首token即分歧，没有可比后续位置。全部EOS且未到2048上限；无GT、评分、模型、独立重解码。

2026-10-30全5349题双路径DeepStack开关单次实际启动：plan 1eeeaa1f4a2cdbf12cdd1d6a17af4a7931ad221ba85021ec919596502323c482，allocation fcb1a78e54c23d441afd3c5a3326668ebd64ab5424f6c6f201516f33e0d6447e，root exec16152，controller2139616/ticks1348555672、docker2139898/ticks1348556411、native2139964/ticks1348556421，root launch 60c7ea2367759bfed84b878f0e319489d9028e1a530c2198b4f4b7efac4eaead。14CPU测试、无设备容器、作者torch2.8钩子及独立只读review通过。先32固定端点锚定已接受16和全量，再对5349题的plain关闭DS与EADP1024开启DS共10698次生成；复用已完成原两臂评分基线，必须全部生成与token独立重解码、三进程和root exec实际退出证明后才单独原官方/独立ANLS评分。新高维tensor仅留实际运行摘要，无法事后逐byte重建；旧全量未保存所有内部状态，跨运行仅有固定16端点锚。不是作者注册paper主表配置，不替代任何剪枝行；目前无新分数、无自动重试队列。

并行只CPU准备 `reproduction/20261028/qwen/fixed16_k256_ds_toggle/`：旧真实16 K256 greedy选中空间排序ID、输入features/processor及旧完整回答文本可锚定；DS三层须按同一真实排序ID取256行，并在压缩后实际embedding中构建256视觉位置mask。作者torch2.8无GPU容器10测试通过；旧全量没有输出IDs/prefill，不能造历史token锚。该目录仅草稿，缺完整落盘/控制/独立终端与冻结，未加载模型/启动GPU，等待当前K1024全量结果后再决定推进。详情见README。

2026-10-30全5349题Qwen双FULL路径DeepStack开关已完成：先32既有原端点、32固定16变体锚，再10698新生成；原容器/native2139964、docker2139898、controller2139616实际退出，根PTY16152 exit0/rootproof `e1d5d1003e96aca181797018da09bdadbe9714e9ef35aabfec55217624e50724`；10698 processor、实际prefill开关摘要、后续状态及token独立重解码通过。之后作者官方+独立逐题ANLS的CPU评分exec11840实际exit0/scoreproof `ed8f8ae84380c92f88a7083c219d74bec8cd2b15257a3c09c0311c8898377d42`，report `f0a4d86377d946bbb44620bb6bbb6553af001e10e7f01d2905d50f25c54a4b44`。plain原基线 94.481578011→关闭DS 92.385504368（-2.096073643点，666回答改变，逐题升110/降268）；EADP1024原基线 87.825405943→开启DS 91.310070544（+3.484664601点，1155回答改变，逐题升574/降250）。两新臂空答0、Excel NA转换各5，完整分母；原生/独立逐题一致。旧全量逐题内部及全部数值标志缺失，仅固定16锚；新高维张量仅运行时摘要，跨运行因果范围受限。该完整1024开关是实现路径诊断，不是论文K256结果，不能单独分摊−6.656路径差或外推至K256。详见 `reproduction/20261028/qwen/doc_full_ds_toggles/RESULT.md`。七剪枝目标仍未复现。


2026-10-30 补充：Qwen DocVQA 全量 DeepStack 同路径开关已完成独立只读终审（复核进程实际退出 0）。核对 10698 条新记录、32 个既有端点及 32 个固定题开关锚、保存的逐题评分和配对、生成与评分退出链；终审未重跑模型或评分。新全量高维张量仅存摘要，旧全量内部状态未逐题闭合，仍是有条件的跨运行对照，不能称为论文 K256 主表复现。详见 `reproduction/20261028/qwen/doc_full_ds_toggles/RESULT.md`。


2026-10-30：Qwen DocVQA K256 固定16题同选择 DeepStack 开关已完成真实32次生成、独立CPU验证及根PTY实际退出0。原生16历史选择与文本锚点全部通过；两臂用同一真实256位置和输入，开启DS后11/16回答改变，32/32 EOS。未读取GT、未评分；回答变化不证明准确率方向，不能称论文复现。详见 `reproduction/20261028/qwen/fixed16_k256_ds_toggle/RESULT.md`。

2026-10-30：K256 固定16 DS 对照进一步通过独立只读终审，逐字节复核32份数组及退出链；16题的11条回答改变并非准确率结果。


2026-10-30：Qwen DocVQA K256 同实例 DS 双臂全量诊断已通过最终冻结、25项无设备CPU测试及独立只读审查，并由根PTY 86476启动；计划先核16题双臂锚，再执行其余题。运行时状态以 `reproduction/20261028/qwen/doc_full_k256_ds_pair/state.json` 与根PTY为准。完整生成、独立验证、实际退出前无全量分数；此为修改公开实现的条件诊断，不覆盖作者原K256主行。


2026-10-30 MMB新增纯CPU输入证据：作者 `LLaVA/llava/eval/model_vqa_mmbench.py:118–119` 保留 inactive `question=row["question"]`，当前实际CLIP指导仍为hint+选项全文（LLM完整prompt已在此前构造）。离线native CLIPTokenizerFast核4377条全文IDs/mask指纹全部吻合既有参考；仅问句使4377/4377指导改变，长度>77由703降4、最长582→79，且322行同时去hint。全部1176循环组内question/图像相同，当前全文指导1176/1176随选项旋转变化、问句指导1176/1176组内恒定。保存 `mmbench/question_only_guidance_cpu_audit_20261030/`，report SHA a1ce95c88c11abba589eb38835761cd1e6408a995e1519833503228342160759。未加载模型/运行selector/读取GT或预测/评分，无GPU；只证明输入变化，尚未测importance、实际mask、LLM输入或准确率。该注释不证明paper argv或已知bug。当前MMB元数据及旧协议诊断的有界检索未见此MMB对照，NeXT OCR问句控制不能替代；现查MMB树无可用真实pruner数值cache，只有输入指纹和运行记录。最小后续建议是按1176组排序等距固定16完整组（60旋转行），同K128/.5/1/q.2/三wait/greedy1024，source全文与问句指导两臂各60次native生成，source先锚已接受原文本，完整LLM提示不变，记录真实选择/视觉与非视觉区/输出；仅提方案，未写GPU plan、未排队或启动。它同时改变hint/选项语义与分段，不能据面板变化推断收益或paper配方。本次公开材料未发现新的POPE判别项；既有wait、alpha/beta、q及公式阴性不重跑。

2026-10-30：MMB question-only 指导固定16组/60行候选已完成冻结及无GPU容器26项CPU测试，并经独立只读审查；`reproduction/20261028/mmbench/question_only_guidance_panel/plan.json` SHA `73bc827e45aaa12c61b6585f9f92f3e1071fec5c5ca5b6becb3ed1c1b38f9fa1`。首次容器路径别名预检失败并留档，修正为文件身份检查后通过。尚未启动或排队；当前Qwen全量实验单模型占用GPU。该方案仅为源码注释支持的指导内容对照，候选不证明论文配方。


2026-10-30：Qwen DocVQA K256 同实例 DeepStack 双臂完成全部5349题、独立CPU验证、根生成PTY86476实际exit0及评分PTY20783实际exit0。原生DS-off独立ANLS 61.166042，DS-on 67.903106，差+6.737063点；2446回答变化，1129逐题改善、480降低；各有4条到2048且无EOS，全部分母保留。DS-on修改公开实现，非论文配方，原论文62.8与原K256 61.166042的对比不变；非固定16高维只保存摘要。独立终审尚在进行。详见 `reproduction/20261028/qwen/doc_full_k256_ds_pair/RESULT.md`。


2026-10-30：MMB固定16组60行CLIP指导对照已完成真实120次生成和独立CPU验证，根PTY68264退出0。60/60实际最终mask改变，但仅1/60回答改变；source16组mask组内均变化、仅问题指导16组组内mask均相同。120/120 EOS。没有评分，不能外推全量准确率或认定论文配方。旧GPU未保存mask/输出ID，输入字节来自CPU重建。独立终审待完成，详见 `reproduction/20261028/mmbench/question_only_guidance_panel/RESULT.md`。首次shell路径错误未启动controller/模型并留档。

2026-10-30：Qwen K256 DS 同实例全量配对已通过独立只读终审，覆盖全部10698条记录、32锚、全部官方和独立逐题分数及生成/评分退出链；未重跑模型或评分。非锚高维仅摘要与论文实际配方未知的限制不变。

2026-10-30：MMB固定16组CLIP指导对照已通过独立终审，复核120份NPZ/2160数组、120次独立重解码、历史文字与CPU输入参照、完整退出链；无GT评分，仍不可外推全量效果。

2026-10-30：MMB 4377行CLIP指导双臂全量候选 plan 523434132597cf923e6428094ddd8056ebe748797c58633fe9bdaa207abf6865 经32项无GPU容器测试与独立只读审查后，根PTY16635启动，先核固定60真实数组锚，再运行其余4317逐题配对。生成/验收不读GT，所有完整结果及根实际退出后才在无设备容器分开进行公开源码exact matching和本地letter循环组评分；不可称外部官方server。非锚高维仅摘要，非论文实际配方，原K128主行保留。

2026-10-30：MMB 全量指导对照首次执行在第一条source锚的 image_sizes 断言失败，根PTY16635实际exit1，原controller/worker已退出、GPU释放，保全 `question_only_guidance_full/root_failure_exit_proof.json` 与worker log；没有得到全量结果或评分。正在只读定位格式或实际输入差异，不能将此失败当成功。

2026-10-31：MMB全量首次失败确认为图片尺寸元组/JSON列表的检查不兼容；首题实际生成及18数组与旧面板逐字节一致，但原作者输出文件为空，不能算通过。原失败证据保留。独立 sibling run02 只改尺寸检查与捕获，不改模型输入；35项无GPU测试及独立审查通过，plan 09404c6685b0e148f3ea6373b3a79fd3842997ef14f5be2891d16d5a8784ea99，根PTY86075启动。执行情况看state及根PTY，不按部分预测计算分数。

2026-10-31：MMB 4377行/1176组同实例指导双臂run02完成真实8754次生成、CPU验收、根PTY86075 exit0、无设备评分PTY57729 exit0。公开exact-matching+本地循环组及本地独立字母解析，两种均source 61.564626%、question-only 62.159864%，提升0.595238点（724→731组，15改善/8退步）；95行回答变化，4372行mask变化，全部输出EOS。未运行外部官方服务器，question-only未证论文argv、非锚高维仅摘要，原paper/主行保留。首次全量断言失败及评分容器PID start_ticks未记录均留档；独立终审待完成。详见 `reproduction/20261028/mmbench/question_only_guidance_full_run02/RESULT.md`。

2026-10-31：MMB 4377行指导双臂 run02 已通过独立只读终审：8754条记录、120固定数组锚、两种解析及全部1176组评分、生成/评分退出链均核对。评分容器PID未保存start_ticks，根PTY和容器ID退出已确认；外部官方服务器未运行，paper配方未确认。

2026-11-02：Qwen3 K256 同模型DeepStack开关性能实测：单A40、16个固定DocVQA问题、每题每臂1次、两种生成方式各16对/64次实际测量，每臂各模式预热1次。固定32输出总耗时均值关1.873364秒/开1.877397秒（+4.033毫秒，单题波动大于均值差）；首轮LLM CUDA事件关78.890/开80.474毫秒（+1.584毫秒，含开侧选行注入）；整次PyTorch allocated峰值均16932.291 MiB（局部额外内存可能被其他峰值掩盖，并非所有设备总显存）。自然生成长度10/16题不同，不能据总耗时判断加速。全部选择与32自然输出ID锚定既有真实结果，实验退出0，未评分；只适用于本机小样本，见 `reproduction/20261102/qwen_k256_ds_performance/RESULT.md`。
作者公开Qwen3代码中DivPrune、CDPruner、EADP共用不传DeepStack的剪枝生成路径，FULL走原生路径；paper表4仅DivPrune/CDPruner作为剪枝比较，公开代码状态不证明论文实际运行版本。两篇原方法论文未报告Qwen3，作者公开仓库有适配代码；不能断言论文所有比较数均由作者亲自重跑。证据 `reproduction/20261102/qwen_k256_ds_performance/baseline_sources.md`。
