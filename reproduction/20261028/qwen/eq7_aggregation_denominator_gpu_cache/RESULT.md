# Eq7 denominator: completed GPU cache replay

All 16 fixed questions were measured once on A40 with torch 2.8.0+cu128.
Every source replay matched the original GPU cache **byte for byte** in all
four gates: importance, ordered 256 IDs, the full gain trace and the final
spatial BF16 feature sequence. No question was left unmeasured.

The sole change was aggregation denominator `.010001` to `.01`, retaining
q=.2, alpha=.5, beta=2 and K256. Both arms started at the same saved native
FP32 global/dense/visual similarities and real BF16 projected features.

| Source versus literal Eq7 | Questions changed | Maximum absolute change |
|---|---:|---:|
| Entropy weights |16/16|2.1338463e-5|
| Local aggregate |13/16|1.5581027e-6|
| Importance |12/16|9.4771385e-6|
| Gain trace |12/16|0.002593994140625|
| Ordered 256 IDs |0/16|—|
| Final spatial IDs / BF16 sequence |0/16|—|

All selection histories remained identical, so the gain difference above
compares valid candidates under the same history and excludes selected-mask
sentinels. The earlier CPU reference's one ordering change did not occur in
this GPU measurement.

The existing independent NumPy terminal audit passed: **288 finite output
arrays**, 16 compressed NPZ files totaling **57,728,997 bytes**, all array hashes,
256-step argmax/masking, source byte anchors and spatial reconstruction.
Controller 2832239 and worker 2832246 exited; root execution 27423 returned 0
and the root confirmed the GPU empty. No verifier or GPU run was repeated.

The plan SHA is `4a2bbbc8b484cf7660ef1bfdcf38107262175425567e5eed3f22eeee783d811d`.
The terminal audit SHA is `193469c9afae8e2a087b7344e8d961850c943cd8700c18e407f9ca8b43a5a343`;
root process proof SHA is `5e4af0e00f4f906bf0f8ae7895762a6ca5aab22b68fb545e602e76c26a212db1`.
Detailed measurements are in `execution/per_question.jsonl` and the linked NPZ
artifacts; termination is recorded in `execution/release.json` and
`root_terminal_process_proof.json`.

This is a feature-cache pruning diagnostic. It loaded no 8B model or FA2,
recomputed no visual/text features, read no GT or answers, generated no text
and calculated no accuracy. The unchanged final features apply to these 16
cases; they do not establish all 5349 outcomes, the paper recipe or a cause of
the remaining reproduction gap. No full-run variant or retry is scheduled.
