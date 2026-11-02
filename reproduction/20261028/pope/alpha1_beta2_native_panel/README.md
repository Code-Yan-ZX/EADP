# Native fixed16 alpha1, beta1 to beta2

CPU preparation only. No model/GPU, automatic queue or launch is authorized by
this directory. This is one fixed polarization mechanism hypothesis: the public
POPE shell uses alpha1/beta1, whereas the paper defines beta>1 and includes a
beta2 ablation. The ablation has LLaVA-v1.5/K128/five-task scope; it does not
identify the paper K32 argv. No known-bug/repair, complete2x2 or comparable
historical alpha.5/beta2-runtime claim is made. No new budget or beta sweep.

Exactly the prior native16 image/question panel remains, in its original order.
The complete native LLaVA-v1.5-7B model, original processor, full question/answer
instruction prompt, original CLIP guidance suffix removal, q.2, K32, three
caller-stream waits, SDPA/FP16, vicuna_v1 and greedy max128 are retained.
Source alpha1/beta1 runs all16 first. Only after every source has passed its
anchors may the same model's beta scalar change to2 for all16 variants. The
native encode function still reads its original `getattr(self,'beta')`; the
actual alpha/beta/K observed at each encode is recorded. Weights and original
prepare/generate/encode/cross arithmetic bodies are unchanged.

The only additional native greedy code is
`gain_refs.append(total_gain)` after the selected-mask assignment and before
argmax. Every iteration allocates total_gain anew and does not later mutate it.
This retains a reference without reading/copying/synchronizing its content.
The inverse transformation restores the entire original greedy AST, and actual
compiled observer-code SHA is recorded. Gain stacking and every captured tensor
content read/copy occur after native generate returns. This is instrumentation,
not a latency/efficiency measurement or a claim of fully unobserved scheduling.

Every completed generation first saves its raw NPZ and attempt JSON. Only then
are anchors and structural assertions applied. Failure adds a sidecar and stops
with all measured output retained; no later source/variant is generated, no
flags or equality gates are relaxed, and no automatic retry occurs.

The source must match old actual native1/1 projected FP16/pixels and FP32 global,
dense, visual similarity, importance, ordered32 and effective mask. Its complete
gain trace also matches the old same-source-AST GPU helper trace; the old cache
did not capture native internal gains. The historical complete8910 source1/1
prediction text is matched only at exact unique(questionID,fullprompt) keys,
with the canonical question/image mapping. No historical token IDs or prepared
embeddings exist; none are invented. Those existing generated source answers
are read solely for the consistency anchor, never GT, scoring or sample choice.

Both new arms save actual output IDs and worker-native tokenizer decoding,
complete LLM IDs, projected/image/text features, similarities, importance,
ordered32/quotas/full gains/mask and actual native prepared FP16 embeddings.
For this B1 image, the actual32 visual rows must equal projected features in
the boolean mask's spatial order at the single image marker. All corresponding
image/text features, similarities, pixels, full LLM IDs and nonvisual embedding
prefix/suffix must match across arms; only beta and its downstream selection
and generation may change. Outputs retain BOS/EOS/length: without EOS they
describe only the declared128-new-token horizon, not later output or logits.

The independent CPU terminal verifier uses NumPy/stdlib only after the real
native worker exits. It audits all32 raw/case artifacts, source anchors,
cross-arm feature/nonvisual bytes, spatial inputs, gain argmax/history, actual
beta metadata, output-ID ranges and horizon, frozen source/model/image/cache
guards. It does not run a selector/model, read labels, normalize/score answers
or independently tokenize/re-decode; native decoding is disclosed as such.

Default `run.py check` is CPU-only. A future explicit root note must bind the
plan and CPU acceptance SHA, run ID, GPU slot/device0, boot ID and exact allowed
other GPU PID/starttick/boot identities, with at most2 loaded models including
loading. A single O_EXCL claim allows one attempt. The controller waits for real
native exit, then one CPU verifier, and saves release. Root separately proves
the actual controller/PTY return. A controller failure alone never proves its
worker/GPU was released. No GT, accuracy, ledger update or automatic next run.
