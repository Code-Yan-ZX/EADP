"""Complete code/plan inventory for the PriorTR DEV32 execution chain."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];PREP=HERE.parent/'preparation';INSTALL=HERE.parent/'install001'
def sources():
 own=('sample_001.json','sample_001_independent_review.json','comparison_amendment_001.json','resource_plan.json','freeze_sample.py','review_sample.py','source_inventory.py','gate.py','panel_core.py','runtime.py','panel_worker.py','launch.py','preflight.py','panel_verify.py','cpu_checks.py','test_panel_core.py','test_runtime.py','test_worker.py','test_launch.py','test_verify.py','test_gate.py')
 paths=[HERE/x for x in own]+[PREP/x for x in ('preprocess.py','trace.py','env_contract.py','admission.py','smoke_verify.py','smoke_worker.py')]+[INSTALL/'env_fingerprint.json',INSTALL/'provenance.json',BASE/'initial_plan.json',BASE/'monitor/supervise.py']
 e2=BASE/'e2_e3_full';paths += [e2/x for x in ('preflight.py','full_imports.py','panel_gate.py','source_inventory.py')]
 provenance=json.loads((INSTALL/'provenance.json').read_text());paths += [INSTALL/'source'/x for x in provenance['original_source_sha256']]
 runplan=BASE.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';plan=json.loads(runplan.read_text());paths.append(runplan);paths.extend((BASE/'e4_panel_v2/panel_verify.py',BASE/'e1_independent_review/review_e4panel.py'))
 for key in ('source_sha256','implementation_sha256'):paths += [Path(x) for x in plan[key]]
 result=sorted({str(x.absolute()) for x in paths})
 if any(not Path(x).is_file() for x in result):raise ValueError('missing source inventory item')
 return result
