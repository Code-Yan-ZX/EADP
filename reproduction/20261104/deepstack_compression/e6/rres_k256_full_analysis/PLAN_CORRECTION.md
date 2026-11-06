# Correction to the frozen R-res scoring plan

The frozen `PLAN.md` incorrectly calls E1 P1-D1 an uncompressed reference. It is EADP greedy selection from 1024 visual tokens to K=256, followed by compact contiguous one-dimensional position IDs on all three axes and zero RoPE delta; DeepStack remains enabled. E1 P3-D1 is the corresponding K=256 EADP reference using retained original three-axis positions and DeepStack. The historical identifier `e1_full` in scoring files and proofs refers to P1-D1 and means neither keep-all nor an uncompressed model. `e1full001` names the complete 384-question E1 panel.

Evidence: the accepted runtime snapshot `e1/e1full001_sources/position_adapter.py` constructs every FourArm with keep=256 and changes P1 positions; `e1/e1full001_sources/adapter.py` verifies that the EADP greedy quota and actual selected output contain 256 tokens.

The selected reference rows, scores, per-question pairings, bootstrap intervals and formal or independent proofs are unchanged. This correction is recorded separately because `PLAN.md` and the scoring sources are frozen and bound by the accepted source-hash chain. No uncompressed reference is included or claimed.
