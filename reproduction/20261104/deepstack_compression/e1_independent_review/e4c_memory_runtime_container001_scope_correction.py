"""Correct disclosure in historical no-GPU runtime preflight review; no journals read."""
import hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent.parent
names=['e1_independent_review/e4c_memory_runtime_container001_review.json','e4_c_memory_runtime/container_preflight001/worker_preflight.json','e4_c_memory_runtime/runtime_preflight_entry.py','e4_c_memory/worker_preflight.py','e4_c_latency/latency_worker.py','e4_c_latency/input_preflight.py','e4_c_latency/e1_reference_gate.py']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((B/names[0]).read_text())['predictions_opened'] is False
assert json.loads((B/names[1]).read_text())['predictions'] is False
assert 'prepare_memory' in (B/names[2]).read_text()
assert 'prepared=prepare(stage)' in (B/names[3]).read_text()
assert 'load_accepted_e1_references' in (B/names[4]).read_text()
assert 'read_jsonl(Path(proof[\'files\'][\'journal\'][\'path\']))' in (B/names[6]).read_text()
r={'status':'scope_correction','original_record_preserved':True,'predictions_opened':True,'prediction_source':'previously accepted E1 DEV mixed prediction journal','use':'identity fields only; no accuracy scoring','GT_opened':False,'scoring_performed':False,'gpu_authorized':False,'explanation':'The recorded false prediction-opened fields in both historical worker preflight and independent review are incorrect. The preparation path parses the mixed journal to reconstruct input identity. The present correction reads source and metadata only and does not read the journal.','source_sha256':{n:sha(B/n) for n in names},'correction_script_sha256':sha(Path(__file__))}
out=B/'e1_independent_review/e4c_memory_runtime_container001_scope_correction.json'
out.write_text(json.dumps(r,indent=2)+'\n')
print(out)
