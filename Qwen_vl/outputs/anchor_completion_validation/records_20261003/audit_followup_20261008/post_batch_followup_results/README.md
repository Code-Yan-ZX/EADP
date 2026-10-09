Post-batch K128 controls and author-confirmed VizWiz scorer revision.
Original 75 arms/55 groups plus 5 followup arms/3 groups are complete; their predictions and official LOO score files remain unchanged.
On 2026-10-10 the user reported author email confirmation of the EADP released VizWiz evaluator, then explicitly selected the original recipe yielding NeXT AnchorZip 60.72.
The main table uses the complete original alpha0.5/beta2/full-guidance/max1024 AnchorZip prediction block for both LLaVA models and all three budgets, before stream-wait-repair generation. All six cells use the EADP normalization and mean min(matches/3,1); official LOO stays auxiliary. Historical prediction runtime metadata is empty; parameter evidence comes from the existing driver/source audit.
Shared alpha0/beta1 public-default EADP/AnchorZip K128 controls and repaired legacy-budget scores are retained separately. Their EADP values do not fill the missing same-recipe main-table VizWiz baselines. No per-cell maximum selection is performed; cells lower than controls are also retained. Author confirmation concerns the scoring formula, not full paper generation argv.
All non-VizWiz numerical cells and averages are unchanged. The frozen refresh_tables.py/final_table_annotation.py document the original official-primary presentation; apply_author_confirmed_vizwiz.py documents the initial scorer-only revision, superseded for main prediction selection by select_vizwiz_recipe_6072.py.
See source/selected_vizwiz_recipe_6072_20261010.json for the current user-selected policy, full per-question scores and source hashes. source/author_confirmed_vizwiz_scores_20261010.json preserves the supplementary complete controls and recalculated paired statistics.

{
  "followup_v15_vizwiz_K128_release": {
    "identity": "d4d01148332b6e176c917153e1bc53deb5fbdcb8d47c594823a34eb170c371cc",
    "commit": "848accca72f5d003a2052bedb5cc7fb0022b2f65",
    "scores": [
      {
        "model": "v15",
        "task": "vizwiz",
        "method": "EADP",
        "budget": 128,
        "n": 4319,
        "metric": "VizWiz official leave-one-out accuracy (%)",
        "accuracy": 56.5200277842093,
        "prediction_sha256": "4820cf34fe00eebe423a596291ce40afcc5ddcc18f4d930e1416f6ec0ad12857",
        "protocol_sha256": "1e91b939f13c1705246939e26c33dc7d01d4c13741b36e05c0573be9708926b9"
      },
      {
        "model": "v15",
        "task": "vizwiz",
        "method": "AnchorZip",
        "budget": 128,
        "n": 4319,
        "metric": "VizWiz official leave-one-out accuracy (%)",
        "accuracy": 56.513081731882366,
        "prediction_sha256": "b9da153342f19f6813c407e6e8fa29e652084bae8e95b48c74770a096ef5ed4a",
        "protocol_sha256": "2409c5a08e071310d2454c51a2d556eca10610637e69fba60ae3508bed18a2f5"
      }
    ],
    "published_utc": "2026-10-09T14:15:57.371977+00:00"
  },
  "followup_next_gqa_K128_beta2": {
    "identity": "6ecb052a48e6dc7a4f967d9f01713f287f7758c5e0d315aa37adf2f9bf780237",
    "commit": "fec8d30ae1c1c2f09fac30266887a5f8df1dbabf",
    "scores": [
      {
        "model": "next",
        "task": "gqa",
        "method": "EADP",
        "budget": 128,
        "n": 12578,
        "metric": "GQA exact-match accuracy (%)",
        "accuracy": 62.86373032278582,
        "prediction_sha256": "e3b761230c4c8ddad43ec1e46bb0f30f6b15b57d9ac56b02fe11c7aab62c28df",
        "protocol_sha256": "f36653b8ff91b8d535c87f309677c875bdf621c10b74434ac401f25d0e79da6c"
      },
      {
        "model": "next",
        "task": "gqa",
        "method": "AnchorZip",
        "budget": 128,
        "n": 12578,
        "metric": "GQA exact-match accuracy (%)",
        "accuracy": 62.617268246144064,
        "prediction_sha256": "573959bbcfbdd08cf9b5039f9a77138e239cc998dc9122f6f6b431da1fdca87c",
        "protocol_sha256": "8787dce3f3a307a6ff2b83b77da21a5fcd4bc692dc05b9b8df3f6bb72d2f1e07"
      }
    ],
    "published_utc": "2026-10-09T15:12:45.901290+00:00"
  },
  "followup_next_vizwiz_K128_release": {
    "identity": "ca5e3ecbf921d49caa86ac38ed70a154473aaeca0b2b1faed8ee71b539f30466",
    "commit": "833227351c48d74bb230578e7b391df2779f2d5e",
    "scores": [
      {
        "model": "next",
        "task": "vizwiz",
        "method": "EADP",
        "budget": 128,
        "n": 4319,
        "metric": "VizWiz official leave-one-out accuracy (%)",
        "accuracy": 59.21741143783283,
        "prediction_sha256": "a9a35ec9c25fc0cb4d83ab40ea63be03d2a41b54828cc2f6fe878f3369b29a44",
        "protocol_sha256": "4fe88c97088f7ebc3ec09e2d18b0bf93d1b6f64254d0ff60e4afbf40cc40a160"
      },
      {
        "model": "next",
        "task": "vizwiz",
        "method": "AnchorZip",
        "budget": 128,
        "n": 4319,
        "metric": "VizWiz official leave-one-out accuracy (%)",
        "accuracy": 59.00671451724938,
        "prediction_sha256": "70136f0457515a154978866f84f1691b7e0b1f4617d11a2d9347daac26bf9afe",
        "protocol_sha256": "48dbbfedf54f901472bf40dd2f8f28e500d9f8cd4002c610c25b2e1b8fd8547e"
      }
    ],
    "published_utc": "2026-10-09T15:40:36.607426+00:00"
  }
}