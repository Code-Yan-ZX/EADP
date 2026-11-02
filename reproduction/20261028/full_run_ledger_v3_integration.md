# v3完整台账接入候选

仅新增 `build_full_run_ledger_v3.py` 及其未来 `full_run_ledger_v3.{json,csv}`，不改原九行ledger/builder或v2十行ledger/builder。构建器尚未执行；须根审读后手动调用，无调度器、模型、评分或verifier调用。

v3固定承接v2 SHA `5b19d5a62d4b05dff0c430063766649023f4f123633814352f5da9f574d9ac50`，原10行逐项及其序列化保持。每条原始预测和独立报告SHA继续核对。原comparison/evidence/两代ledger及builder均有硬SHA门。

本次唯一新登记为 `NeXT_full_K64_question_only_public_comment`：CLI64／名义320，alpha.5／beta1／三wait／greedy128／SDPA FP16。公开inactive split注释支持这一CLIP指导输入假设；完整LLM OCR输入及原剪枝数学保持，paper argv仍未确认，paper参考57.0。

根实际分配：plan `3a1bfff5…`，allocation `bdd88721…`；PTY60045，controller2861700/ticks1333486787、native2861704/ticks1333486819。构建器固定launch snapshot SHA和两进程身份。**没有root_terminal_process_proof时仅输出10complete＋1pending**，不打开本次execution中的answers、score、runtime或逐题文件；pending的value／delta／n为null，CSV留空。

接纳第11行须root proof明确PTY60045实际exit0、原controller/native已退出，并与claim／snapshot／release及当前proc/GPU名单交叉核对；controller complete/native exit0；全5000官方／独立逐题分、输出重解码、实际guidance和完整LLM IDs、crop配额／mask／有效前缀门；计划／源码／输入／引用SHA、模型严格stat及3166图stat；raw/runtime/两份逐题/实际guidance全部SHA和5000行数；fresh作者CLI退出0。只消费原已接受分数，不计算metric；分数提高／降低／是否接近57均不作gate。失败侧车进入失败列表，无自动重试；不以结果创建paper复现成功标记。

根最终proof须提供 `root_owned_exec_handle=60045`、`root_owned_exec_actual_exit_code=0`、`all_original_processes_exited=true`、与launch一致的完整controller/native身份、`plan_sha256`、allocation SHA（`allocation_note_sha256`或`allocation_sha256`）、release及independent verification SHA。这些是根持有实际PTY的证据，builder不自行伪造。

查看表尚未修改。建议根审查后最小加入优先级：

```diff
     if (RUNS / 'full_run_ledger_v2.json').exists():
         ledger_path = RUNS / 'full_run_ledger_v2.json'
+    if (RUNS / 'full_run_ledger_v3.json').exists():
+        ledger_path = RUNS / 'full_run_ledger_v3.json'
```

`conditions(row)`另对`input_hypothesis_only`附加“仅问句CLIP指导假设；LLM完整OCR不变；paper argv未确认”。原10行缺少该字段，显示保持。现renderer只消费完整行分数并单列pending数量，故不得为pending填值；若新增pending逐行显示，只用该条的paper57／模型／预算／条件和空白value／delta，不能读取raw。v3 CSV已显式包含pending状态和空值。不要把新结果替代原K64完整OCR行，不取最高或补平均分。

本候选仅做stdlib语法／源码比较，未运行builder、评测、verifier或模型。完成接入后仍须按实际结果独立判断论文目标，合法source值不是paper复现。
