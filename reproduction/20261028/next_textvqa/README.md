# NeXT TextVQA reproduction audit, 2026-10-28

Existing complete stable EADP scores are 57.866 / 55.410 / 52.086 at nominal
640 / 320 / 160 tokens, versus paper 59.2 / 57.0 / 54.2. The preserved full-token
baseline freshly re-scores to60.370 versus reported60.3 (+0.070pp). It rounds
to60.4, so exact one-decimal agreement is not claimed. The pruning discrepancy
remains unresolved; see full_token_baseline_recheck/baseline_recheck.json.

The new author checkout is `e1a08801461c181871b9ff1cb0803b8e9966f083`.
Twelve relevant files are byte-identical to the old workspace, including the
native loader, pruning architecture, CLIP, evaluator and environment files.
The native published launcher defaults to alpha .5 / beta 1, while the old
primary run used beta 2. The already completed full K32 beta1 control was
50.966; repeating it is not a new reproduction fix.

`source_equivalence_plan.json` registers the same 64 fixed midpoint samples
used in the earlier diagnostic. `repro_panel.py` runs the native fresh-author
entry with matched alpha .5 / beta 2 / K32 / greedy128 / FP16 / SDPA and the
same three caller-stream dependencies as the frozen reference. It preserves
author files and the scoring/pruning mathematics. CPU preparation confirms
that imports resolve to the fresh checkout and does not initialize CUDA.
The diagnostic compares answers to complete, previously run frozen-reference
predictions; its accuracy is not a replacement for the full 5000-question test.

The completed source-equivalence execution used:

```sh
/home/dell/miniconda3/envs/llava_pruner/bin/python \
  reproduction/20261028/next_textvqa/repro_panel.py run --entry author --gpu-authorized
```

All 64 generated answers and actual retained-token counts exactly matched the
frozen reference; both panel scores are 56.40625%. Independent CPU review also
passed. See `panel64_author_wait/result.json`. This fixed panel does not replace
the full benchmark or resolve the paper residual. The process exited and slot A
was released; a redundant prior-entry generation is unnecessary.

`full_default_K128.plan.json` pre-registers the previously unrun public shell
default alpha .5 / beta 1 / CLI K128 (nominal640), greedy128, FP16/SDPA, using
the same three disclosed caller-stream waits. After all MMB completion/validation
and exit gates passed, its worker started in slot B at 10:40:33 UTC, PID2399640.
The first answer is recorded; complete5000 scoring is pending. The completed K32 beta1 score 50.966 remains negative
evidence. This recipe is not asserted to be the paper argv; no parameters are
selected or retried according to the resulting accuracy.

`repro_full_default.py run --gpu-authorized` requires a separately allocated
slot. Its CPU `verify` command requires all5000 ordered composite question keys,
aligned runtime rows, exact decoding of captured generated IDs to raw answers,
the actual configuration, frozen source/input identity and the fresh official
TextVQA scorer. Partial results cannot pass the full denominator. The prepared
image manifest contains all3166 local JPEG SHA256 values plus the prior fresh
official ZIP CRC/size proof; model payload hashes are reused only with unchanged
recorded stats. Eight CPU acceptance tests pass, including token decoding of the
completed real64 panel and rejection of alternative pruning-arch switches.
The full worker also observes actual vision body/projection/text/projector
parameter dtypes, torch default dtype and the first actual input image dtype;
these observations add no casts or numerical changes.

`embedded_vision_vs_official_base.json` compares all391 embedded vision-body
parameter tensors on CPU without constructing a model. The released NeXT
checkpoint stores them as BF16; its runtime FP16 values differ from base CLIP
FP32 cast to FP16 in all391 tensors. Even comparing base CLIP rounded to BF16
with the stored checkpoint leaves335 tensors different (56 equal), so simple
release quantization does not explain all weight differences. The earlier
loaded-runtime audit established equality with the NeXT checkpoint itself,
not equality with base CLIP. This does not establish the cause of the accuracy
gap or justify replacing the checkpoint vision body. The script uses mmap and
keeps CUDA uninitialized; the active generation worker and plan are untouched.

Root has conditionally allocated slot B after the full MMB wait control. The
CPU-only `supervise_after_mmbench.py` waits for complete4377/1176-group scoring,
author XLSX, independent verification, paired comparison and finalizer success,
actual previous-worker exit, unchanged frozen inputs and fewer than two GPU
models. Its registration/state/log preserve the exact authorization and gates.
Any terminal failure stops without retry. The passed gates, authorization and
source/plan SHA identities are preserved in the registration and launch records.
The separate CPU-only `finalize_release.py` watches the full terminal state and
automatically writes the run directory's `release.json` after frozen guards,
all score/raw hashes and actual process/GPU-PID exit pass. It cannot start or
retry generation. Third-party full-score rechecking is recorded separately in
`../pope/run03_single_alpha05_beta1_wait_k32/predecessor_verification.json`;
neither release nor independent verification is claimed while the run is pending.

`audit_official_images.py` attempts fresh downloads of the official annotation
and selected official ZIP members, checking the archive central-directory CRC
and size for all 3166 used images. Its final JSON distinguishes CRC/size checks
from cryptographic checks of downloaded members. A successful HTTP HEAD alone
does not establish successful downloading or image identity.

The fresh download audit completed successfully: the 5000 OCR questions from
the newly downloaded author-linked eval.zip and the official annotation are
byte-identical to the previous files. All 3166 used JPEGs match the fresh ZIP
central-directory CRC32 and size; 64 downloaded members (64 unique JPEGs) also
match SHA256. Only 87,733,234 ZIP range bytes were transferred.
All three current NeXT weight payloads (14,125,909,456 bytes) independently
match the official HF revision LFS hashes; metadata/tokenizer files also match
the earlier verified bytes. The local `model_type=llava_llama` adaptation is
explicit and is the sole recorded semantic config difference from HF.

Exact author dependency pins were installed into an isolated vendor directory,
leaving the shared environment intact. Across the full corpus, Pillow 10.4.0
versus 12.3.0 produces identical decoded RGB and AnyRes FP32/FP16 tensors on
all 3166 images. Tokenizers 0.15.1 versus 0.15.2 produces identical CLIP IDs and
masks on all 5000 guidance prompts. SentencePiece 0.1.99 / protobuf 3.20.3 versus
0.2.2 / 7.36.2 produces identical native LLaVA prompt IDs on all 5000 questions.
See `author_package_full_corpus_comparison.json` and
`author_sentencepiece_causal_cpu_comparison.json`. These results rule out those
input changes; they do not prove the entire runtime or paper argv identical.

Already completed controls are not repeated: the stream wait changes the full
K32 EADP score by only +.090; official weights, image preparation, corpus prompt
keys, crop-budget rounding, facility selection and answer completeness were
audited earlier. Valid-content CLIP masking and bounded CLIP chunks did not
recover accuracy in the previous fixed64 panel and are not promoted as fixes.
