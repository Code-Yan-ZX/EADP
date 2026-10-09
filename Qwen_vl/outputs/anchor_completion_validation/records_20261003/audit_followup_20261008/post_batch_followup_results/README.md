Post-batch predeclared K128 controls. Original 75 arms/55 groups are unchanged.
NeXT GQA pairs a new EADP arm with the completed original AZ128 under the same beta2 recipe.
VizWiz uses two new full arms per model under predeclared public defaults alpha0/beta1; not verified paper argv. Official leave-one-out remains primary; no score-driven selection.
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
    ]
  }
}