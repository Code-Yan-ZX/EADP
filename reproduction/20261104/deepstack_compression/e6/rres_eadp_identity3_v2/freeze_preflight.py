"""Freeze diagnostic CPU-only inputs, never granting GPU authorization."""
import json,os
from pathlib import Path
import gate
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];TIMING=ROOT/'e6/rres_eadp_timing_pilot_v4';REV=ROOT/'e1_independent_review'
def main():
 if not gate.ended(HERE/'monitor_cpu008/started.json',HERE/'monitor_cpu008/terminal.json') or not gate.ended(HERE/'monitor_import_order001/started.json',HERE/'monitor_import_order001/terminal.json'):raise ValueError('CPU/import-order not completed')
 prior=gate.prior();sources=gate.read(TIMING/'preflight_plan001.json')['source_sha256'].copy()
 names=['PLAN.md','identity_core.py','formal_contract.py','worker.py','timing_imports.py','check_import_order.py','gate.py','preflight_entry.py','preflight_launch.py','freeze_preflight.py','test_identity_core.py','test_formal_contract.py','monitor_cpu008/started.json','monitor_cpu008/terminal.json','monitor_import_order001/started.json','monitor_import_order001/terminal.json']
 extra=[HERE/x for x in names]+[TIMING/'preflight_plan001.json',REV/'rres_eadp_timing_v4_preflight_result_review.json',REV/'review_rres_eadp_timing_v4_preflight_result.py',REV/'monitor_rres_eadp_timing_v4_preflight_result/started.json',REV/'monitor_rres_eadp_timing_v4_preflight_result/terminal.json',ROOT/'e6/rres_k256_pilot_v2/capture.py',ROOT/'e4/decode_observer.py']
 for p in extra:sources[str(p.resolve())]=gate.sha(p)
 if any(gate.sha(path)!=digest for path,digest in sources.items()):raise ValueError('source drift')
 result={'status':'frozen','scope':'identity3_cpu_preflight_only','prior':prior,'source_sha256':dict(sorted(sources.items())),'anchors':[['DocVQA_VAL','64588'],['TextVQA_VAL','36511'],['AI2D_TEST','207406']],'arms':['eadp','rres'],'GPU_authorized':False,'GT_opened':False,'existing_predictions_opened':True,'limits':['Actual model and GPU diagnostic still locked','Hook capture is diagnostic only; high-dimensional arrays not retained','Prior v3 preflight covered imports/accepted references, not model execution']}
 with (HERE/'preflight_plan001.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
