# Fixed MMB input preparation

Preparation completed on CPU only. `inputs/panel.tsv` contains60 unique
pre-expanded rows from16 fixed complete circular groups, in the original TSV
order; `all_rounds=false`, `round_id=0`. It has exactly these columns:
`index, question, hint, A, B, C, D, image`. There is no GT `answer` column.
The16 image payloads are also saved without transformation in `inputs/images/`.

The preparation process did read and hash bytes of the original TSV, which
contains GT. Its parser explicitly selected only the eight input columns; GT
values were neither parsed nor used. Historical model-response texts were read
as source-replay anchors and are explicitly marked as non-GT. This distinction
must not be shortened to “the source bytes contained no GT.”

The immutable manifest is `inputs/manifest.json`:

- SHA256: `920a465f64e220e519b41e13664a5959abb5cf55c1b4c5ab2410c539f721ebfb`.
- Panel TSV SHA256: `66dc079872510098d12ec4e2fdab79eedd5df462ce6f9cd6ba6b0cd42b14ee83`.
- `rows` order exactly matches the TSV. Each row has `index`, `group_id`,
  `position`, `original_position`, `round_id`, `full_guidance`,
  `question_only_guidance`, `llm_prompt`, `llm_input_ids`, `llm_n_tokens`,
  `llm_input_ids_sha256`, `source_clip`, `question_only_clip`, `image` and
  `accepted_source`.
- `llm_input_ids` is one-dimensional `[N]`, int64; native generation unsqueezes it
  to `[1,N]`. The raw int64 byte SHA is unchanged by that unsqueeze.
- Each CLIP entry has `input_ids` and `attention_mask` shaped `[segments,77]`,
  plus their `input_ids_sha256`/`attention_mask_sha256`, `text_sha256`,
  `n_tokens` and `n_segments`. Arrays follow the native zero-padding and
  consecutive-segment transform, with int64 little-endian byte hashes.
- `image.pixel_float16_sha256` and `image.pixel_float32_sha256` bind the actual
  native CPU preprocessing result to both existing current/author-pin
  references. Shape is `[1,3,336,336]`; runtime image dtype is `torch.float16`.
  `image.path` points to the original encoded payload; `payload_sha256`,
  `decoded_rgb_sha256`, `size` and `mode` are also recorded.
- `accepted_source.text` is the completed prior source model response, not a GT
  answer. The source raw line/file SHA and original runtime call are retained.
  Historical masks, importance, embeddings and output token IDs were not saved
  and are not claimed as anchors here.

All60 complete source guidance fingerprints, LLM input fingerprints, model
response records, original runtime observations and actual processed-image
fingerprints were checked. CPU image preprocessing ran once per unique payload;
there are16 images. A TSV save/reload roundtrip checked all60 prompts and payloads.
No vision/text model, selector, generation or scoring was run.

## Distinct test entry points

`prepare_inputs.py` is a preparation tool, never a worker or container-preflight
entry. It refuses to overwrite `inputs/`. Its preparation-side tests are:

```bash
python -B reproduction/20261028/mmbench/question_only_guidance_panel/test_prepare_inputs.py
```

Seven tests passed. This preparation-only test imports preparation contracts and
hashes original source TSV bytes; it parses no GT. Its report
`input_CPU_tests.json` explicitly marks that boundary. Do not put it into a
runtime/container-preflight test suite.

An independent stdlib-only entry is available for runtime/preflight:

```bash
python -B reproduction/20261028/mmbench/question_only_guidance_panel/test_sanitized_inputs.py
```

Six tests passed with no preparer or torch import, no model or scoring, no file
writes and zero original-GT-TSV open attempts. A path guard rejects opening that
source. This entry reads only the sealed manifest, sanitized panel, copied image
payloads and label-free CLIP/LLM fingerprint references. It checks IDs and complete
rotations, GT-column rejection, full prompt construction, actual saved int64
arrays, image fingerprints and the sealed historical-binding metadata.

The manifest `sources` map includes the original TSV for provenance. Runtime
code must not indiscriminately open/hash every map entry: that would read the
original GT-bearing file again. Consume the sealed manifest, sanitized panel and
images, and explicitly approved label-free source/model references instead.

These inputs do not prove that question-only guidance changes selected tokens,
answers or accuracy. The inactive source comment is not proof of paper argv.
Option rotations need not return the same answer letter; any consistency check
must refer to guidance/selection or correctly account for option relabeling.
