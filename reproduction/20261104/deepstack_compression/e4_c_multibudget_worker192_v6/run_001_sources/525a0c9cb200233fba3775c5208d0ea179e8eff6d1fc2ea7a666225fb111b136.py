"""Separate-process E1 acceptance and input identity gate; no GT or E4 output."""
import json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(BASE/'e4_panel_v2'),str(BASE/'e4'),str(BASE/'e1_analysis')]
from panel_v2_imports import load_local
from identity_worker import read_jsonl
from input_preflight import bind_e1_references

def main():
 if len(sys.argv)!=4:raise ValueError('expected E1 proof, verifier started and terminal paths')
 proof,inputs=load_local('panel_gate').validated_dev(*(Path(x) for x in sys.argv[1:]))
 if proof.get('GT_opened') is not False or proof.get('scoring_performed') is not False:raise ValueError('E1 proof scope invalid')
 rows=[dict(row,split='DEV') for row in inputs]
 references=bind_e1_references(rows,read_jsonl(Path(proof['files']['journal']['path'])))
 print(json.dumps({'rows':rows,'references':list(references.values())},ensure_ascii=False))
if __name__=='__main__':main()
