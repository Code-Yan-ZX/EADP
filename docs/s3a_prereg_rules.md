# S3-A decision rules (fixed 2026-09-24, before any Δ number was computed)

The measurement protocol and all case/context/candidate constructions were
frozen in `s3a_cases.py` before the first GPU forward (2026-09-24 01:10).
The rules below were fixed on the same date after the harness gates passed
and BEFORE the first instance of the main grid was analyzed. No rule was
touched after seeing results.

Numeric scale convention: every Δ is a difference of mean gold-answer NLL in
nats. The harness re-measures, per instance, a `noise|floor` probe: the same
set scored batched vs per-gold; the maximum absolute drift is this instance's
bf16 resolution floor (~0.08 nats at these logit scales).

## Definitions

* `share(ds)`  — mean two-way interaction variance share of the
  12-candidate × 5-context Δ matrix, per benchmark (Part A).
* `ratio(ds)`  — mean within-token Δ range ÷ mean |Δ(base)|, per benchmark.
* `rho(ds)`    — mean Spearman ρ of the 12-candidate Δ ranking between the
  base context and the weak context.
* `resc(cond_k, ds)` vs `resc(unary_k, ds)` — answer-rescue rates on the
  base-wrong rescuable instances (Part D); paired bootstrap per instance.
* 3×floor      — per-instance threshold for calling a Δ (or difference of
  Δs) resolved rather than noise.

## Rules

1. **REFUTED** iff for every benchmark: the within-token Δ range lies below
   3× the instance noise floor for > 80 % of candidates (i.e. nothing
   resolves), AND every cross-context `rho ≥ 0.9`, AND the
   `resc(cond_k16) − resc(unary_k16)` paired CI includes 0 with point
   estimate < +1 rescue on every benchmark. If REFUTED, the
   conditional-utility route stops; nothing else is looked at.
2. **CONDITIONAL-UTILITY-SUPPORTED** iff ALL of:
   (a) some benchmark has `ratio(ds) ≥ 0.5` (context variation is at least
       half the utility's own scale) with bootstrap CI above the noise floor;
   (b) some benchmark has `rho(base,weak) ≤ 0.8` or top-3 Jaccard ≤ 0.6
       (the identity of the best tokens actually changes with S);
   (c) on the rescue experiment `dL(cond_k8) > dL(unary_k8)` with paired CI
       excluding 0 (the conditional greedy, despite its oracle-like
       objective, converts into better loss) OR
       `resc(cond_k) − resc(unary_k) ≥ 0` at every benchmark with at least
       one strict win.
3. **WEAK / TASK-SPECIFIC** — everything else, including: conditionality
   visible on one benchmark but not another (e.g. OCRBench yes, TextVQA no),
   or visible in NLL but not converting into rescue wins.

## Benchmark-difference test (Part B)

TextVQA-vs-OCRBench contrast on `share` and on `ratio` uses independent
instance-level bootstrap CIs on the difference; "strongly more conditional"
requires the CI to exclude 0 AND the point difference ≥ 0.15 (share units).

## Honest-giving-up clause

If the rescue arm shows cond ≤ unary AND all sign flips / rank changes are
confined to the noise band, the report must say CONDITIONAL-UTILITY-REFUTED
even though this kills the S3 direction; the same clause that made S2-C0/C4/C5
report negatives stands.
