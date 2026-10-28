# V2 evidence snapshot

Only numbers reported in the EADP paper and traceable existing local experiment scores are accepted. Directly supplied numbers without experiments are excluded.

Regenerate with Python's standard library (CPU only):

```sh
python scripts/versioning/build_v2_evidence.py --source-root /media/disk2/YZX/research/EADP_amp --output docs/versions/v2
```

Accepted: 889 records ({'eadp_paper_reported': 783, 'local_experiment': 106}); quarantined: 58.

`evidence.json/csv` contains accepted values; `quarantine.json/csv` explains withheld cells. `provenance.json` records checked source hashes and the exact scope and limitations. Paper budgets are nominal totals; local budgets retain their recorded CLI units (NeXT CLI128 is nominal640). Paper VizWiz test/validation values are separate. Local recipe, English-instruction CN and native/legacy caveats remain explicit. No new generation or score maximization is performed. Derived local averages are emitted only when every required component is accepted.
