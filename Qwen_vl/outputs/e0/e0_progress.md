# E0 progress note

Branch: `e0-native-baselines`. Spec: `docs/e0_native_baselines_prereg.md` (frozen).

## Environment (recorded 2026-09-29, M0)

| item | value |
|---|---|
| GPU | NVIDIA A40, 46 GB (1 card, index 0) |
| driver | 560.35.05 |
| conda env | `qwen3vl_clean` (python 3.10.20) |
| torch | 2.10.0+cu128 |
| CUDA (torch build) | 12.8 |
| transformers | 4.57.6 |
| VLMEvalKit | 0.2rc1 (editable, `Qwen_vl/VLMEvalKit`) |
| attention | SDPA (`QWEN3_VLM_ATTN_IMPL=sdpa`; no flash-attn wheel for this combo) |
| env conventions | `HF_HUB_OFFLINE=1`, `LMUData=/media/disk2/YZX/LMUData`, `HF_DATASETS_CACHE=/media/disk2/YZX/hf-datasets-cache` (from `Qwen_vl/scripts/discovery/common.py`) |

transformers 4.57.6 `modeling_qwen3_vl.py` was read before implementation:
`get_rope_index`, `_deepstack_process` (inject after layers 0–2 via
`visual_pos_masks`), the decode branch (`position_ids=None` → `arange +
cache_position + self.rope_deltas`) and `prepare_inputs_for_generation`
(forcing `position_ids=None`) all behave exactly as prereg §3.2 assumes.
No amendment needed on this account.

## M0 — data partition (2026-09-29)

Script `Qwen_vl/scripts/e0/e0_plan.py` → `Qwen_vl/outputs/e0/e0_plan.json`.
Seed `E0_SEED=20260929`; exclusions at image level; sources exactly the frozen
list (all `*_plan.json`, SAGE fit/val labels, `sample_indices(n,150)` bank,
S2-A causal cases).

| dataset | total rows | excluded imgs (rows touched) | pool rows / imgs | DEV | CONFIRM |
|---|---:|---:|---:|---:|---:|
| TextVQA_VAL | 5000 | 608 | 3958 / 2558 | 300 | 1949 |
| DocVQA_VAL | 5349 | 490 | 2767 / 796 | 300 | 1362 |
| OCRBench | 1000 | 606 | 328 / 324 | **164** | 164 |
| ChartQA_TEST | 2500 | 0 | 2500 / 1509 | 300 | 1245 |
| MMBench_DEV_EN_V11 | 4876 | 0 | 4876 / 1310 | 300 | 2472 |
| MMStar | 1500 | 0 | 1500 / 1430 | 300 | 749 |
| RealWorldQA | 765 | 0 | 765 / 762 | 300 | 381 |
| POPE | 5127 | 0 | 5127 / 5127 | 300 | 2563 |

- OCRBench is short exactly as prereg §2.5 predicted: 164 DEV / 164 CONFIRM
  (606 of 1000 images touched by history). Reported as-is; nothing borrowed
  from CONFIRM.
- DEV caps at 300; surplus images were dropped, not moved to CONFIRM
  (per-dataset `dev_dropped_for_cap` in the plan JSON).
- Text-only MMBench rows (no image) are each their own group.
- Split hashes: global DEV `35214185be3a73814fbb335d0dc61c8227ba9ae99dcac86e83be9096794783c7`,
  global CONFIRM `b81f9135e8cec3cd571c1c2818b3a74251164a07e69673b1f75a2bee3cb8a7ec`.
  Per-dataset hashes inside `e0_plan.json`. CONFIRM answers are NOT read; only
  indices + sha256 are archived.

## Host resource note (2026-09-29)

The GPU host runs an **Ollama service** (`/usr/local/bin/ollama`, systemd
managed) which intermittently loads a `qwen3.6:35b` chat model (~27 GB of the
46 GB A40) and unloads it after ~5 min idle. It is not ours to kill. All E0
GPU scripts carry OOM-retry loops (150 s backoff) so foreign loads cannot
fail a gate or corrupt an accuracy run; wall-clock timings measured while the
foreign model is resident are discarded/retried.

## CIVIC code search (prereg §4.2, 2026-09-29)

Queries (web search): "CIVIC visual token pruning compressed KV anchor
aggregation Qwen3-VL code github arXiv 2605.28115"; follow-ups on the arXiv
page and Semantic Scholar. Result: paper identified ([arXiv:2605.28115],
"CIVIC: End-to-End Sequence Compactness for Efficient Vision-Language
Models", Yang et al., Univ. of Utah, May 2026) — **no public code found** on
GitHub/project page/author pages. Per prereg §4.2 "no code" branch: E0 does
not reimplement it; it is listed as an un-reproduced near neighbour that
(a) requires training and (b) uses compressed-KV anchor attention; its own
evaluation (Qwen3-VL-2B, MMMU/MathVision/ODinW-13/RealWorldQA/VideoMME)
does not overlap this project's setting. A second search is run before the
report is finalised.

## M1 — native repair + gates (2026-09-29): ALL GATES PASSED

`Qwen_vl/scripts/e0/e0_gates.py` → `e0_gates.json`, `all_passed=True`:

| gate | result |
|---|---|
| N1 identity (16 DEV samples, K=1024, 32 forced steps) | max\|Δlogits\| = **0.00e+00** prefill and every decode step, 0/512 token mismatches (bit-exact) |
| N2 DeepStack live | min \|Δ\| = 2.88 ≫ 0.1 — injection is real |
| N3 3-D positions (K=256 random keep) | kept tokens carry full-sequence (t,h,w); text monotone; decode step n = prefill_max+1+n |
| N4 invariants (12 samples × 8 local arms) | 0 bad: exact K, no dup/OOB, DS lengths, cache len, layer_calls = 36·(1+decode_steps) |
| N5 legacy B2 vs archived SAGE-confirmation B2 | **32/32 prediction-exact** (100 % ≥ 95 %) — the new engine's B2 selection matches the code that produced the archived predictions |

The N1 lm_head note (amendment A2): with the all-positions lm_head the
identity is bit-exact; the earlier 6.25e-02 was pure GEMM-shape rounding on
the last-position-only call.

## M2 — baseline ports (2026-09-29)

Cloned upstream repos (`_upstream/`), HEADs pinned and cited in each port:
FastV d1659729, PyramidDrop 6444f304, VisionZip 8f86b55c, SparseVLMs
a9e71427, PACE 240b2206. Ports live in `Qwen_vl/model/baselines/`; the local
pruners (cdpruner/hipruner) got index-exposing wrappers (`select_indices`).
N4 on 12 DEV samples: visionzip/fastv/pdrop/sparsevlm all 0 bad (sparsevlm
counted with the amendment-A4 recycling bound; realized visual count =
K + n_recycled). Implementation bugs found and fixed during the port are in
git history (DS compaction alignment, layer-call double count, GQA repeat_kv
in PDrop scoring, SparseVLM recycling pre/post-compaction split).



## CIVIC runtime re-search (§4.2, 2026-09-29)

Second search round ("CIVIC End-to-End Sequence Compactness code release
github ..."): again **no code repository found** — only the arXiv entry,
Semantic Scholar (no code link), and third-party summaries. Conclusion from
§4.2 unchanged.

## M3 plan (frozen reading of the prereg)

M4's question count: 9 pruned arms x 3 budgets on the full 8-dataset DEV
panel (27 x 2264) + B0 (2264) + R-res variants (4 x 2264) + PACE (3 x 2264) +
A1/A2 (2 x 764) ~= 82k generations. With the prereg §8 upper bound of 25
GPU-hours, the 1.5x line is 37.5 h, i.e. an observed s/q of ~1.65 s/q on the
host. The chain script (/tmp/e0_chain.sh, also mirrored in git) measures the
observed s/q on the first ~120 b2/K=256/TextVQA questions, projects the total
with the exact run inventory, and stops with `outputs/e0/M3_BUDGET_STOP` for
a human decision if the projection exceeds 37.5 h. Nothing else in the chain
depends on the estimate, so a stop costs nothing.

## Smoke results (M2 close-out, 2026-09-29)

`e0_smoke.json` — 30-question TextVQA DEV smoke, K=256:
visionzip / fastv / pdrop **0/30 truncated, 0 empty**; sparsevlm (full
recycling) **20/30 truncated, 1 empty** → amendment A6: main grid runs
`sparsevlm_norecycle` behind a smoke gate in `run_m4_all.sh` (skip + flag if
its truncation rate exceeds the prereg's 2 % bar); the full-recycling port is
labelled 移植存疑.

## Chain status

`/tmp/e0_chain.sh` running (2026-09-29 20:14): PACE Qwen2.5-VL reproduction
(≈4765 requests, ~2.1 s/it) → M4 full grid with the A6 smoke gate and the M3
budget guard. Logs: `/tmp/pace_repro4.log`, `/tmp/m4_all.log`,
`/tmp/e0_chain.log`.

## ⚠️ 2026-09-30 01:45 — D1 computed but BLOCKED by an unexplained anomaly

The D1 inputs (a1/a2/b2 K=256 OCR panel) completed and the mechanical
verdict says D1 does NOT fire (pooled B2−A1 per-question delta +7.5 pts,
CI [3.5, 8.4]). But the absolute numbers fail every sanity check:

| arm (native unless noted) | TextVQA | OCRBench norm | historical reference |
|---|---:|---:|---|
| b0 (unpruned) | **85.03** | **14.3** | 73.6 / ~62-68 |
| a1 (legacy B2) | (empty official) | 10.3 | 71.04 / 62.3 |
| b2 (native B2) | (empty official) | 13.1 | ~71 / ~62 |

TextVQA +11 over the historical B0 and OCRBench −48: both directions are
implausible as a repair effect, and the same engine reproduced the archived
SAGE legacy predictions 32/32 (gate N5) and stock generate bit-exactly
(gate N1) on DEV samples. So the anomaly is in the accuracy-run pipeline,
not obviously in the engine. Suspects, in order:
1. the DEV-row message/scoring path differs from the SAGE-conf one
   (build_prompt / use_custom_prompt interplay);
2. OCRBench per-question scoring mismatch on the reduced DEV subset;
3. Text/DocVQA official dicts came back EMPTY ({}) — the evaluate call for
   them silently produced nothing, so their macros are currently
   CONTRIBUTING NOTHING (macros over 1-2 datasets only).
D1/D3 conclusions are ON HOLD until this is root-caused. Do not cite the
numbers above.

## ✅ 2026-10-02 — BLOCKED 销案（S0/S1/S2 完成）

**结论：09-30 的"异常"没有一个来自预测生成；全部是评分/参照层的问题。
预测 shard 有效，无需重跑。**

### S0 — 代码 diff（无 GPU）

`discovery/common.py`（build_message）自 M1 (796d7f9) 起零改动；e0 与存档
SAGE-conf 管线用同一个 `common.build_message`（sage_deploy.py:70），且 N5
已端到端 32/32 复现 → 嫌疑 #1（消息路径不同）不成立。

补丁（commit 4e01e0c）：`e0_accuracy.run_score` 空 official / evaluate 异常
= 硬失败（写 `score_failures.jsonl`，非零退出，不再静默写空 dict）；
`e0_analyze.macros` 显式记录 `missing_ocr/missing_general`，决策规则经
`require_panel` 拒读不完整面板；`official_main` 支持嵌套 dict。

### S2 — OCRBench 子集公式（嫌疑 #2，证实）

official Final Score 的类别聚合在 164 行子集上崩溃（s2
_ocrbench_subset_check.json，逐题 scorer 为 round-1 验证版 27/27）：

| arm | official-on-subset | 逐题重评 |
|---|---:|---:|
| a1 (legacy B2) | 10.3 | **62.80**（历史带：sage_conf_B2 62.08） |
| b0 | 14.3 | 87.20 |
| b2 | 13.1 | 79.88 |

→ "OCRBench −48" 为子集公式假象。**规定：official OCRBench 分只在全量集
上报；DEV/子集一律用逐题分。**

### S1 — TextVQA 双配置归因（嫌疑 #1 的行为学验证）

同一 DEV-300 行，b0 (identity K=1024) legacy 配置（DS off + 1D pos，
即历史 harness 配置，research_reset M2 §1.4）：**75.33**；native 配置：
85.03。配对差 **+10.0 点 [cluster-bootstrap CI +5, +14]**（38/10，
s1_dualpath.json）。75.33 落历史带（73.6 / sage_conf_B2 74.21）→
"+11" 完全归因于 native 修复（DeepStack + 3D mRoPE），两侧数字各自有效。

### D1 判定修复（原实现两处 bug）

原 D1 段 (1) 对两臂 per_q `dict.values()` 直接逐元素相减（键集不对齐时
静默错配，出现 mean 与 CI 矛盾）；(2) 阈值写成 2.0（per_q 为 0/1，2 点
应为 0.02）。修复后（键对齐 + image-cluster bootstrap）：
**D1 fires=True：native B2 − legacy A1 macro = +7.52 点 [CI +4.89,
+10.15]**（TV +9.83 / DV **−3.73** / OCR +16.46）。TextVQA 与 S1 的 b0
归因（+10）交叉一致。DV 的 −3.73 如实记录。

### 遗留（S3，未销案）

FastV（per-q OCR 34.15）/ PDrop（54.88）仍显著低于其论文自报水平：查截断
率与 in-LLM pruning × DeepStack 交互，或主表脚注 + 附录 sanity。其余端口
（visionzip/divprune/cdpruner）逐题水平正常。D2/D3 仍 INCOMPLETE（perf
文件不在），不影响本销案。

**引用规则**：`e0_verdict.json` 的 official-on-subset OCRBench 数字作废；
一切 DEV 对比用逐题分（本文件及 s1/s2 JSON）。
