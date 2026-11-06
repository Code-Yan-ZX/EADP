"""Post-run compare persisted actual identity against accepted E1 P3-D1."""
FIELDS=('processor','visual_sha256','deepstack_full_sha256','ordered_ids','spatial_ids','prefill_embed_sha256','attention_mask_sha256','prefill_position_ids','rope_delta','injected_mask','injected_deepstack')
def actual_identity(actual,reference):
 if not isinstance(actual,dict) or set(actual)!=set(FIELDS) or any(k not in reference or actual[k]!=reference[k] for k in FIELDS):raise ValueError('actual full E1 identity mismatch')
 return True
