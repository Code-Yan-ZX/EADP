# E0 pre-registration amendments

Frozen **before** any accuracy number of the affected arm exists. Each entry
states the deviation/clarification, the reason, and the replacement rule.
Nothing here modifies gates, arms, datasets or decision rules; the prereg
(`docs/e0_native_baselines_prereg.md`) stands.

## A1 (2026-09-29, before M2 arm implementation): budget alignment for variable-ratio arms

The prereg fixes the budget K = visual tokens entering the LLM, and forbids
hyper-parameter tuning. For arms whose official hyper-parameters do not land
on K by themselves, the following alignment rules are frozen now, before any
of these arms produces a number. All rules keep the official *structure*
unchanged and scale only the overall retention so the arm meets the budget
(the same treatment the prereg §4 prescribes for PDrop and PACE).

1. **FastV** (official: prune before layer 3 using layer-2 attention, keep the
   top `round(N·(1−R))` visual tokens by the last-token row of the head-mean
   attention; README settings R ∈ {25%, 50%, 75%} removal).
   Rule: R := 1 − K/N, i.e. the removal ratio is the budget. K=256 ⇒ R=75%
   (the official README's own setting); K=128 ⇒ R=87.5%; K=64 ⇒ R=93.75%.
   The prune layer follows the prereg table (after LLM layer 2, scores from
   layer 2), which is the official K=2.

2. **PDrop / PyramidDrop** (official: rank&drop after layers [8, 16, 24],
   `image_token_ratio_list` geometric, default [0.5, 0.25, 0.125] of the
   original visual tokens).
   Rule: keep the official layer list [8, 16, 24] and the geometric stage
   structure; per-stage retention r := (K/N)^(1/3) so the three stages land at
   K after layer 24 (int() truncation per stage exactly as the official
   implementation). K=256 ⇒ r≈0.6300 ⇒ stages [645, 406, 256] of 1024.
   The per-stage token counts are reported with the results (prereg §4).

3. **SparseVLM** (official: text-guided top-k at pruning layers [2, 6, 15],
   per-stage retention schedules keyed to a final token count, token
   recycling; default schedules e.g. 576-token LLaVA: 192 ⇒ [300, 200, 110]).
   Rule: keep pruning layers [2, 6, 15] and the official schedule *shape*;
   scale the 192 schedule by K/192 (its final retained count), i.e. stage
   retention [400·K/256, 267·K/256, 147·K/256]... expressed exactly:
   [1.5625, 1.0417, 0.5729]·K. K=256 ⇒ [400, 267, 147]; K=128 ⇒ [200, 134, 74];
   K=64 ⇒ [100, 67, 37]. Full token recycling is attempted first; if it
   cannot be ported faithfully to Qwen3-VL, the prereg's downgrade applies
   (recycling-free variant, explicitly labelled).

4. **VisionZip** (official Qwen2.5-VL adaptation: dominant = top 65% by
   last-block received attention, contextual = 5% anchors aggregated by
   last-block key similarity, `contextual = target + mean(assignees)`).
   Rule: keep the official dominant:contextual ratio 65:5; scale both by K
   (dominant = round(13K/14), contextual = K − dominant, computed per image
   with the official int() semantics). K=256 ⇒ dominant 238, contextual 18.
   Merged tokens keep the anchor's 3-D position; DeepStack streams are
   averaged with the same assignment (prereg §4 VisionZip row).

## A2 (2026-09-29, before M1 implementation): N1 logit comparison detail

The stock `Qwen3VLForConditionalGeneration.forward` computes logits with
`lm_head` over ALL prefill positions, while the engine's generation path
computes only the last position. The GEMM shape changes bf16 accumulation
order (~1 bf16 ulp on large-magnitude logits, measured 6.25e-02 on TextVQA
DEV samples; every decode step was bit-exact at 0.00e+00). For the N1/N2
gates the engine therefore computes the prefill logits with the same
all-positions lm_head call as stock (`full_lm_head=True`); generation arms
keep the last-position call (first-token logits are identical in argmax and
the TTFT window stays honest). Source of the difference is recorded here per
the prereg's "若非零,必须解释来源" rule.

## A4 (2026-09-29, before any SparseVLM run): recycling positions and bookkeeping

SparseVLM's token recycling appends synthetic merged tokens that do not exist
in the full sequence, so prereg §3.2.5's "same-index shrink" cannot apply to
them literally. Frozen rules: a recycled token takes its cluster CENTRE's 3-D
coordinate and cos/sin; its hidden state, DeepStack rows and every cache
layer's K/V are the official uniform mean of its cluster members' rows; it
counts as a visual token for later pruning stages (as in the official
implementation). The realized visual count is therefore K + n_recycled and is
reported per run.

## A5 (2026-09-29, before any PACE run): Qwen3-VL port decisions

1. APC preview runs on the harness's 1024x1024 expanded image (patch grid
   64x64); the compressed resolution targets merged-token geometry with patch
   unit 32; the full ViT pass (and DeepStack) runs on the COMPRESSED grid --
   this is the method's encoder-side mechanism and is reported as the
   realized ViT patch count (prereg §4.1.3).
2. The vision attention score for DDAE fusion is the compressed pass's
   last-block received attention (head-mean, query-sum); if the preview and
   merged token counts disagree it is nearest-interpolated to the merged
   count (alignment detail, frozen before any number).
3. Budget alignment: DDAE keeps exactly K tokens per image (K = the E0
   budget); APC's retention stays data-adaptive. The official token_budget
   ratio semantics are replaced by this fixed-K rule so the arm sits on the
   E0 budget curves; the realized retention/patch distribution is reported.
4. The generation config and everything LLM-side is the shared native engine.

## A3 (2026-09-29, before M1 gates): engine/greedy-loop choice

Prereg §3.2 step 4 offers "自写 greedy 循环" or hooking
`prepare_inputs_for_generation`. Chosen: **self-written greedy loop** through
`Qwen3VLTextModel.forward` (the exact module stock generate drives),
`position_ids` explicit in prefill, decode position = `cache_position +
rope_deltas` recomputed per step, `do_sample=False` semantics reproduced by
raw argmax (the `-1024` config's top_k=1 / top_p=0.001 / temperature=0.01 are
inert under `do_sample=False`, and repetition_penalty=1.0 / presence_penalty=0.0
install no processors). A1/A2 run in the same loop with `pos='1d'`
(`position_ids=None` reproduces the legacy 1-D running index in both prefill
and decode) and, for A1, no DeepStack injection.

## A6 (2026-09-29, before any SparseVLM accuracy number): recycling downgrade

The full SparseVLM port (with token recycling) passes the N4 mechanical
invariants on 12 samples, but the 30-question smoke shows degenerate
generation: 20/30 generations hit the 2048-token cap and 1/30 is empty,
vs 0/30 for every other ported arm — a semantic defect in the merged-token
insertion path (KV-cache rows / positions of synthetic tokens), not in the
pruning logic. Two fixes were attempted (member-index remapping; pre/post-
compaction split of the merge); the second made the machinery consistent but
did not repair generation quality.

Per prereg §4 ("若 Qwen3 上无法完整移植,降级为不含 recycling 的版本,并明确
标注"), the E0 main grid runs **sparsevlm_norecycle** — identical pruning
layers, schedule and text-guided scoring, minus recycling. The full-
recycling port is labelled "移植存疑" and excluded from Best24/25 unless a
later fix passes the same smoke bar. No SparseVLM accuracy number existed
when this was written.

## A7 (2026-09-29, user-authorized scope reduction, before any accuracy number): trend-first M4

The user authorized reducing the M4 grid to reach the D1-D4 verdicts fast
("只要看到走向"). The full grid (82k generations, 23-37 h) is replaced by a
prioritized queue; everything not in the queue is DEFERRED, not cancelled,
and can be appended later with no protocol change. Cuts:

1. **Budget curve**: K=128 is dropped everywhere; K in {256, 64}.
2. **OCR panel first**: TextVQA_VAL, DocVQA_VAL, OCRBench, ChartQA_TEST
   (1064 questions) carry D1/D2/D4; the general panel is reduced to a
   K=256 trend pass for b0, b2, rres512, pace and the two strongest 24/25
   arms on the full 4 general datasets.
3. **Deferred arms**: b1, cdpruner, hiprune (all outside the prereg's
   Best24/25 set) and all K=64/128 combinations beyond the trend endpoint
   (K=64: b2, rres256, fastv, visionzip, pdrop, pace).
4. **M5 (paired perf + resolution sweep) moves BEFORE M4**: it is
   accuracy-independent and unlocks ViTshare and every TTFT contrast in
   D2/D3 tonight.

Cost estimate: ~25k generations ≈ 7 h at the host's observed ~1 s/q, i.e.
verdicts by tomorrow midday; D1 alone readable ~1 h after M5. The reduced
grid changes no arm, hyper-parameter, prompt, or scoring rule; macros are
reported over the panels that actually ran, with the reduction stated in
the report.

A7 revision (same day, still before any accuracy number): after the user
asked to keep more than the OCR panel, the general-panel K=256 pass is
extended from 5 arms to the same main-arm set as the OCR panel (b2, rres,
fastv, pdrop, visionzip, divprune, sparsevlm_norecycle, pace + b0). All 8
datasets now carry the main arms at K=256; only K=128, the K=64 general
pass, and the deferred arms (b1, cdpruner, hiprune) are cut.
