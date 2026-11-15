# Full DocVQA validation: frozen evaluation protocol

2026-11-09. Prepared before any outcomes from this full run. This is a full local DocVQA_VAL evaluation of the current magnitude-V teacher student family, not a paper-wide or unseen-pretraining claim. The 32-document TRAIN pilot used numeric exact and a different prompt; its scores cannot be compared directly to official DocVQA ANLS.

## Scope and data

Use **all 5,349** rows of the local DocVQA_VAL TSV with its original annotation aliases. Preserve the official index, full question, parent document and source image, source TSV hash, deterministic mapping and row count. Do not filter based on full-model correctness, numeric answers, prompt length, truncation or a method's result. Rows are clustered by original UCSF parent document for paired intervals. Validate exact data bytes and original image bytes, and compare parent IDs and exact image SHA256 with all 120 fit, 16 val, 32 test and one dev records used in the previous student experiment. Report training/validation image+parent overlap (must be zero for those records). Other historical project experiments inspected some DocVQA_VAL questions, so the benchmark is not wholly project-unseen even if no student training/selection used these validation parents. Hash checks cannot exclude re-encoding or pretraining overlap.

## Frozen methods

All methods use the same Qwen3-VL-8B cached revision and bf16/SDPA native graph. Preselect the three magnitude-V teacher students from the previous experiment (seeds 11, 23, 37), all reported separately; the student checkpoint for each was selected on its own independent TRAIN validation split. These student models use all V and three DeepStack features plus the image-independent question embedding; the V designation refers only to the label. Their forward scorer accepts no target answer or teacher score. The seven methods on every question are:

- `full`: N=1024 native visual tokens, DS on;
- `b1`: native Qwen3 adapter of EADP scoring plus facility-location, K256;
- `b2`: same native scoring plus block8 facility approximation, K256;
- `rres512`: unpruned native 512-square input with 256 tokens;
- `magnitude_v__seed_11`, `magnitude_v__seed_23`, `magnitude_v__seed_37`: forward-only student Top256.

B1 and B2 are this project's native Qwen3 adapters, not a claim to reproduce every detail of the EADP authors' original Qwen setup. All selected visual positions carry their original V and three DS rows and unmodified native 3-D mRoPE. Each question gets a fresh causal prefill/decode cache and its own original prompt, question embedding and selector. No gold label, answer gradient, or gold-derived model selection enters any inference arm. No additional model is trained.

## Prompt, generation and cache

For image inputs, validate the source TSV image bytes against the manifest, then reproduce the checked-in VLMEvalKit `decode_base64_to_image_file` PIL decode/save behavior in a new experiment-owned image cache, with source and derived image hashes. Its JPEG save can alter RGB pixels; do not silently reuse existing shared image caches. Record PIL/library versions. All seven arms use the same derived image.

Construct every message via the same checked-in VLMEvalKit `common.build_message`/Qwen3 custom VQA prompt. The fixed default adds `\nPlease answer concisely with short words or phrases when possible.` This differs from the student training and previous pilot's `Answer with the value only.`; treat this as a real prompt distribution change, not the same protocol or an isolated dataset effect. Greedy batch-one generation, normal EOS, `max_new_tokens=128` for every arm. Record all outputs/IDs and truncation; preserve capped/empty generations as completed predictions, run a truncation sensitivity only as secondary, never remove them from the primary denominator. This finite cap and native adapter details must accompany comparisons with other work.

The vision encoder takes only image pixels/grid and model weights. For efficiency, group questions from the same source page and keep at most one page's N1024 and 512-square vision outputs in memory. Reuse only when actual preprocessed pixel tensor, grid, resolution, dtype and model revision match exactly. Question embedding, stable K256 selection, original positions, LLM prefill and language KV are always computed anew for each question/arm. Native model has three aligned DS injections. A preregistered independent TRAIN development page and two fixed questions on that same image are used to check cached versus fresh V/DS, selections, positions and generated IDs for all seven arms, plus correct count/cache and single-question behavior. This gate never uses an official validation answer. Save all failed attempts and hashes.

## Primary outcomes and tests

Main endpoint is VLMEvalKit's DocVQA ANLS, not the pilot's numeric exact: for each question use best original alias, official normalization/Levenshtein and threshold semantics. In particular the raw `eval_score` written by the local evaluator is a *distance* and must not be reported as ANLS; use the official `process_line`/`hit_calculate` score path. Report overall question-macro ANLS ×100 for each of seven arms; literal strict exact and truncation are auxiliary. All 5,349 questions stay in primary denominator.

Report each seed and its mean/range without selecting a seed or checkpoint on validation outcomes. Predeclare each of the three student seeds vs `b1`, `b2`, `rres512`, `full`, and the matched comparison `b2` vs `b1`, in before→after order; use 10,000 paired bootstrap draws resampling original document clusters with replacement, shared across arms, seed 20261109, reporting question-weighted ANLS differences and 95% percentile intervals. Because the 5,349 questions cluster into 938 documents, the bootstrap resamples parents and retains all their question rows; report number of positive, negative, tied per-question score changes, full predictions and score differences. Multiple comparisons and prior small-pilot development constrain inference. Do not compare a partial-run score to a complete-run score as if paired; an operationally missing arm remains pending, and no errors are silently scored wrong or replaced.

## Execution and cost

Freeze the complete row/parent/image manifest, original TSV SHA, predeclared arms, checkpoint hashes, train/eval manifests, code, model revision, prompt construction, feature-cache gates, library versions and generation cap before any full-set predictions. Incremental transactional per-(question,arm) storage is resumable; retain statuses, prompts, token IDs, output, truncation, physical token counts, score/provenance and errors. Validate coverage with exactly 5,349 completed answers per arm before declaring full results. Development gates and speed measurements run on the independent prior dev image, and may not pick a favorable main subset. Record sampled per-question inference time and total wall/GPU run time. A separately benchmarked end-to-end TTFT must include fresh preprocessing and vision if made; cached multi-question throughput is a distinct measurement, not deployable first-query latency. Run all predeclared arms/seeds to completion without adapting to early results.
