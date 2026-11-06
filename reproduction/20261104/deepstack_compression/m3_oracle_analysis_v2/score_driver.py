"""Preserve frozen scoring implementation; substitute reviewed v2 gate only."""
import argparse,importlib.util,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(HERE))
import prelaunch_gate,score_gate_v2
def main():
 # Fail closed before importing the scoring implementation or opening inputs.
 audit=prelaunch_gate.validate();binding=json.loads((ROOT/'m3_oracle_verify_v2/acceptance_binding_002.json').read_text())
 if sys.version!=binding['python_version'] or prelaunch_gate.sha(sys.executable)!=binding['executable_sha256']:raise ValueError('scoring interpreter differs from oracle run')
 original=ROOT/'m3_oracle_analysis/score_oracle.py';spec=importlib.util.spec_from_file_location('_m3_score_v2_original',original);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 module.score_gate=score_gate_v2
 rc=module.main()
 if rc!=0:raise ValueError('original scorer returned failure')
 args=sys.argv;assert args.count('--out')==1;out=Path(args[args.index('--out')+1]).resolve()
 if any(prelaunch_gate.sha(path)!=digest for path,digest in audit['source_sha256'].items()):raise ValueError('v2 scoring source changed')
 proof={'status':'completed','v2_review_sha256':audit['review_sha256'],'v2_cpu_proof_sha256':audit['cpu_sha256'],'source_sha256':audit['source_sha256'],
        'scores_sha256':prelaunch_gate.sha(out/'scores.jsonl'),'analysis_sha256':prelaunch_gate.sha(out/'analysis.json'),'scope':'DEV answer-dependent oracle; no deployable or final-set claim'}
 with (out/'v2_driver_proof.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0
if __name__=='__main__':sys.exit(main())
