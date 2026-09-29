# Research reset after SAGE — 2026-09-29

Status: evidence review and research proposal preparation, not a successful new method. No new GPU experiment was run for this review.

## Current evidence and revised objective

Checkout inspected: `main`, HEAD `cb5821a`. The complete `project_handoff.md` was read, but its 2026-09-23 M1-era recommendations are superseded by M2–M9 and SAGE. Existing untracked proposals and IdeaSpark runs were preserved.

The user's current objective is to outperform strong 2024/2025 top-conference visual-token compression/pruning methods on downstream quality OR measured inference efficiency. This does not require every experiment to beat EADP B2 at 1024→256 on Qwen3-VL OCR tasks. EADP itself is a 2026 method. Compression/merging and offline training are within the current brief; online inference remains forward-only. An eventual publication still needs comparison with directly relevant recent methods, particularly any 2026 method used as a starting point.

Borrowing a published mechanism is legitimate with attribution. A renamed reproduction is not an independent contribution. Separate (1) reproducible baseline capability, (2) a proposed modification, and (3) evidence that the modification matters.

## What failed, and what remains open

| Evidence | Supported conclusion | Unsupported extension |
|---|---|---|
| SAGE fresh 720: B2 63.62, SAGE 62.32; paired delta −1.30 points, CI [−2.3, −0.4] | This trained equal-budget exchange critic worsens B2 in this setting | All token compression is exhausted |
| SAGE unary 62.57; sampled hindsight 68.83 | Set conditioning does not help this model; beneficial exchanges exist but the tested predictor cannot identify them reliably | Oracle improvement is deployable |
| M2 corrected paired TTFT: B2 278.6 ms; L4 learned TopK 287.8–288.3 ms | Full-token early decoder processing is too expensive for the tested benefit | Every possible in-LLM method is impossible |
| M4 fixed residual capsules failed, including content controls | Untrained residual-weighted spatial pooling did not improve the incumbent | A learned compression map or trained correction is disproven |
| M2 preprocessing roughly 84 ms, vision roughly 111 ms | A large part of measured TTFT is outside the decoder; reducing encoder work merits investigation | A particular encoder-pruning method already achieves a speedup here |
| M2 batch-1 A40 decode roughly 40 ms/token with little change across arms | Decode throughput and total model memory are weak target metrics in this regime | Token pruning cannot help serving or longer contexts |

Sources: `refine-logs/EXPERIMENT_RESULTS_20260929.md`, `docs/scoring_search_m2_system.md`, its amendment, and `docs/m4_rec_v0.md`. Numbers from different evaluation panels must not be compared as if they used the same samples.

## Mandatory comparison repair

M2 §1.4 explicitly states that the historical harness disables DeepStack and uses ordinary one-dimensional positions. These are internally shared conventions, so they do not erase the historical controlled comparisons, but they restrict those conclusions to the altered model.

Before claiming native Qwen3-VL performance:

1. Restore the original multimodal position construction and DeepStack injection for every arm.
2. Verify the uncompressed identity against the stock model through both prefill and autoregressive decode.
3. For hard pruning, use the same ordered retained indices for all relevant feature streams, positions and cache bookkeeping.
4. For learned merging, define anchor coordinates and the mapping of each feature stream explicitly; mixed features do not automatically have a valid native spatial coordinate.
5. Keep architecture repair separate from the claimed new method. A gain from re-enabling DeepStack is not a compression-method contribution.

Author architecture source: [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631).

## Prior-art anchors verified in this review

| Work | Status verified here | Mechanism relevant to the decision |
|---|---|---|
| [EADP](https://github.com/SJTU-DeepVisionLab/EADP) | Author repository identifies ECCV 2026 | Entropy-filtered dense relevance and facility-location selection; retain as strong reference |
| [VisionZip](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_VisionZip_Longer_is_Better_but_Not_Necessary_in_Vision_Language_CVPR_2025_paper.html) | CVPR 2025 proceedings | Strong pre-LLM compression comparison |
| [DivPrune](https://openaccess.thecvf.com/content/CVPR2025/html/Alvar_DivPrune_Diversity-based_Visual_Token_Pruning_for_Large_Multimodal_Models_CVPR_2025_paper.html) | CVPR 2025 proceedings | Diversity selection without fine-tuning |
| [SparseVLM](https://proceedings.mlr.press/v267/zhang25s.html) | ICML 2025 proceedings | Progressive text-guided sparsification with recycling |
| [MetaCompress](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_Rethinking_Token_Reduction_for_Large_Vision-Language_Models_CVPR_2026_paper.pdf) | CVPR 2026 proceedings; method also read in arXiv HTML | Learns a prompt-agnostic image-conditioned compression matrix before the LLM, using prediction discrepancy and regularization; generic learned merging is occupied |
| [DocPrune](https://openaccess.thecvf.com/content/CVPR2026/papers/Choi_DocPrune_Efficient_Document_Question_Answering_via_Background_Question_and_Comprehension-aware_CVPR_2026_paper.pdf) | CVPR 2026 proceedings | Background and question-aware pre-encoder pruning plus comprehension-aware decoder pruning in document RAG; generic document-background removal is occupied |
| [Information Horizon](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_When_Token_Pruning_is_Worse_than_Random_Understanding_Visual_Token_CVPR_2026_paper.html) | CVPR 2026 proceedings | Deep-layer redundancy depends on task/model; late random pruning is already a published strategy |
| [PixelPrune](https://arxiv.org/abs/2604.00886) | 2026 preprint; no conference acceptance established here | Predictive-coding pixel redundancy pruning before ViT; generic duplicate-patch removal is occupied |
| [VISOR](https://arxiv.org/abs/2603.23495) | Author paper examined; CVF supplemental found | Sparse image–text interactions and selected visual refinement layers; simply skipping visual updates is occupied |
| [Reroute](https://arxiv.org/abs/2606.12412) | 2026 preprint lead | Deferred visual tokens can bypass stages and be reconsidered; recoverability alone is not a new claim |

Search date and cutoff: 2026-09-29. Queries covered visual token pruning/compression, encoder cost, native DeepStack, learned compression, document pruning and relevant conference proceedings. Inclusion favored original papers, official proceedings and author repositories. The `nature-academic-search` MCP methods were not callable in the active tool inventory; this review used web retrieval instead and is not an exhaustive five-database search. The separate fresh IdeaSpark run records its actual connector outcomes. No citation counts were used.

## Experimental decision rules for the next proposal

- Stop using another small equal-budget rescue/scoring modification as the default next step.
- First reproduce a relevant 2026 starting point and the strongest compatible 2024/2025 baselines in the same native execution graph. Do not compare the proposed method on Qwen3 against an old method's published LLaVA number.
- Keep a general visual benchmark panel as well as OCR tasks; declare the target workload before testing. OCR specialization is allowed, but evidence then supports that narrower scope.
- Compare full accuracy–cost curves at multiple budgets. For variable-depth or encoder-side methods, an equal final decoder-token count is not equal computation. Report actual encoder/decoder work, selector overhead, TTFT and fixed-length generation time.
- Suggested engineering target, not a demonstrated result: at matched quality, at least 15% lower median end-to-end TTFT than the strongest applicable baseline. Also inspect tail latency. Non-inferiority must be assessed with a declared margin and paired confidence interval, not merely a nonsignificant accuracy difference.
- Pilot for large failures first. Determine confirmatory sample size from paired pilot variance and the declared accuracy margin; 150 or 300 examples do not automatically resolve a 1-point claim.
- Use new image-disjoint fit/validation/confirmation partitions. Prior SAGE confirmation data is now known and should not be reused to tune the successor while still calling it untouched confirmation.
- Keep a simple baseline that can erase the proposed contribution: plain learned pooling for a compressor, original DocPrune/PixelPrune for pre-encoder work, and matched-cost fixed schedules for an adaptive scheduler.
- A first native-model timing audit can change the ranking of directions. Historical A40 timings are motivation, not a promised speedup on another backend or workload.

The fresh IdeaSpark proposal and its audit status are stored separately under `ideaspark_run/eadp-compute-frontier/`; this evidence note does not certify its outcome.
