# VisionZip / PPE content–position binding oracle pilot

Status: registered 2026-11-08 before downstream predictions. Branch
`codex/visionzip-binding-oracle`, base `a724122d`; previous CF artifacts retained.

## Question and limits

Does forbidding cross-region aggregation repair numeric field binding beyond
preserving multiple positions, at the same decoder token budget? This is an
oracle diagnostic on real DocVQA pages, not yet a deployable new method.
No new compressor, model training, iterative mining or test-time teacher is used.
The available regions are QA-grounded numeric value boxes and OCR line boxes,
**not gold semantic field polygons**. A null result cannot rule out every
possible field-aware compressor.

## Data

Target 60 distinct pages, chosen without compressed-model outcomes, with two
existing annotated numeric questions and uniquely OCR-localized distinct values.
Exclude all UCSF parent source documents used by the prior CF manifest where
identifiable. Render both numbers in both pair members with the same jointly
fitted font conditions; one member preserves values and the other swaps them.
Numeric multiset is unchanged. Save source pages and exact edit boxes.
Accept a pair for primary analysis only if full native model answers both
questions on both members correctly (4/4). Keep all teacher rejections and
their predictions. Do not select pairs using any compressed arm.

Pre-generation curation amendment: montage inspection identified extreme font
shrink and incompatible value types before any new-data model prediction.
Exclude rotated short digit boxes, digit-count ratios above 2, fitted glyph
heights below 60% of either original box, and mismatched explicit value types
(e.g. year versus percentage), using annotation/image rules. Preserve v1 data
and rejection reasons; do not relax these rules to force a sample count. The
curated v2 prediction input has 22 independent pages. Additional strictly
eligible pages may be collected without reference to compressed outcomes.

Geometry amendment after seven teacher screens and before any compressed-arm
prediction: two non-overlapping pixel boxes may touch the same visual token.
Assign a shared token to the box with larger intersection area (ties: lower
field index); save collision counts. This is a model-independent partition,
applied to every arm. A field with no distinct token is explicitly ineligible
for this oracle, not a failed model answer. No source annotation is changed.

Implementation correction: the fixed wrapper first calls qwen-vl-utils
`smart_resize` (32-pixel factors, min=max1024^2), then square-pads and resizes.
Direct source-to-square boxes were an approximation. Compute oracle masks from
raw source boxes through this exact two-step transform, record both mappings.
The first 300 compressed answers were archived and invalidated wholesale;
full-teacher answers and image encodings remain valid and are reused. No
algorithm parameter was selected from those invalidated results.

Sample-size amendment: before inspecting compressed results, request up to 40
additional independent pages from a locally cached DocVQA TRAIN dataset with
provided OCR coordinates. Exclude all VAL parent documents and previous CF
parent documents; apply the same unit/font/orientation/geometry filters. Keep
v2's original 22 records immutable. Report VAL and fresh TRAIN strata separately
as well as the combined pilot; this is not an official test-set evaluation.

Evaluate unedited source questions separately for natural QA damage. No claim
of official DocVQA train/test performance or unseen-layout generalization.

## Execution and baseline audit

Native Qwen3-VL-8B, fixed square 1024 input (1024 visual tokens), K=256,
all three DeepStack streams and native 3-D interleaved mRoPE. Greedy generation,
24 answer tokens maximum, identical value-only prompt. Numeric exact correctness
uses Decimal canonicalization; report ANLS secondarily when useful.

VisionZip audit on first 12 preselected pairs separates:

* historical D238/C18, main anchor+mean versus DS mean-only;
* D238/C18 with the same anchor+mean operator in all four feature streams;
* D205/C51, reserving approximately 5% of original N for contextual tokens.

Generation comparisons use the teacher-valid subset of those 12; rejected
pages remain in the teacher-yield table, not the primary oracle comparison.

The latter is a declared fixed configuration, not selected by pilot accuracy.
Its native port is not an official Qwen3 implementation. The vision forward
stays stock SDPA; a final-block read-only hook extracts received attention and
post-RoPE keys. No wall-clock speedup is claimed from this diagnostic.

PPE is a **paper-operator adaptation**, not reproduction of the author's SFT
model or official benchmark results. Preserve Qwen3 frequency/axis mapping
[24,20,20]; split each axis into four position slots. Choose positions of cluster
members closest to their feature centroid, repeat the closest if undersized.
Change only image rotary cos/sin during prefill. Text/decode positions and
features remain unchanged. Author released code has a different position-choice
implementation; do not silently call this code-identical to it.

Required gates: singleton PPE equals native rotary and first-token logits;
text cos/sin unchanged; PPE modifies intended image slots only; decoder starts
at native full-prompt max position+1; K=256 unique sorted positions; all V/DS
streams use exactly the same clusters; each non-dominant token is assigned
exactly once; no silent unassigned tokens or per-arm sample removal.

## Paired allocation and aggregation controls

All primary variants use the same D205 dominant tokens per image, C51 contextual
tokens and total K256. Region labels come from box/token overlap (positive
area); record counts and collisions. Value regions are primary; OCR lines are
a predeclared sensitivity check, with overlapping-line cases flagged.

An original uniform anchor set may miss a labeled region. Repair coverage with
the fewest deterministic replacements: add its highest-attention contextual
token; remove an anchor from a region with at least two. Never add tokens or
discard uncovered source members. Explicitly record repair counts.

Primary causal contrast:

* `field_anchor_control_ppe`: region-covering anchors, unrestricted assignments;
* `field_oracle_ppe`: **same anchors**, assignments limited to the same region.

Also compare ordinary VisionZip, ordinary VisionZip+PPE, region oracle without
PPE, 4x4 equal-area spatial-region+PPE, and fixed-seed random-region+PPE.
Random labels preserve the exact value-region label counts. Spatial and random
controls repair coverage by the same algorithm. Their own unrestricted-anchor
controls isolate allocation effects if requested by the observed diagnostics.
Report anchor-only gain separately from restricted-aggregation gain.

Keep native 512-square keep-all (256 input and decoder tokens) as a strong
resolution reference. Equal decoder K does not imply equal computation.

## Endpoints and decisions

Primary: 4/4 correct per teacher-valid page; report member/question accuracy,
numeric-set recovery versus correct field assignment, rescued/broken pages,
and document-clustered paired bootstrap 95% CIs (10,000 draws, seed20261108).
Show all individual predictions, full-model acceptance rate and truncations.
No hyperparameter/arm selection based on confirmation outcomes.

Record PPE representative IDs and their target-region coverage per arm. Four
position slots do not preserve every source position: an oracle gain may
reflect both content aggregation and improved position coverage. It supports
the joint oracle; a content-only binding attribution would require an additional
crossed position/content control on common feasible slots.

ADVANCE only if at least 20 teacher-valid independent pages are available,
restricted aggregation with PPE yields at least three net 4/4 page repairs over
its identical-anchor unrestricted PPE control, its paired CI excludes zero,
and it also beats ordinary PPE plus simple spatial/random controls. Inspect
unedited QA damage before proceeding. Any advancement is a pilot signal only.
If n>=20 and the oracle produces no net repairs or matches simple controls,
stop developing this particular mechanism. Intermediate evidence is HOLD.
If teacher yield<20 or the PPE adaptation itself fails interface/quality gates,
report an underpowered/inadequate-carrier result rather than rejecting binding
preservation generally. Do not respond to a null by adding another ranker/loss.

Primary sources:
[VisionZip](https://github.com/JIA-Lab-research/VisionZip),
[PPE paper](https://arxiv.org/html/2510.22936v2),
[PPE code](https://github.com/MouxiaoHuang/PPE),
[S2Prune](https://github.com/yuanyuanjia71-spec/S2Prune).
