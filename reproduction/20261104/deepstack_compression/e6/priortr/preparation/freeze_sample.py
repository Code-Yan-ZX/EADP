"""Freeze eight deterministic DEV items and the unchanged PriorTR smoke plan."""
import json,os
from pathlib import Path
from admission import BASE,E1,DATA,admit,sha

def main():
 proof,rows=admit();out=Path(__file__).parent/'sample_manifest.json'
 plan={'status':'frozen','scope':'DEV only; unscored until identity and CPU scoring gates pass',
       'selection':'3 DocVQA, 3 TextVQA, 2 AI2D unique groups; excludes all old E0 anchors; ascending sha256(20261104|e6-priortr|task|group_id)',
       'e1_completion_sha256':sha(E1/'e1full001_completion.json'),'e1_review_sha256':sha(BASE/'e1_independent_review/e1full001_review.json'),
       'inputs_sha256':proof['files']['inputs']['sha256'],'panel_sha256':proof['files']['panel']['sha256'],
       'old_anchor_sha256':sha(DATA/'anchor_exclusion/anchor_groups.jsonl'),
       'method_configs':[{'name':'native_off','enabled':False},{'name':'author_default','enabled':True,'strategy':'priortr','prune_layer':3,'keep_ratio':0.1111},
                         {'name':'k256','enabled':True,'strategy':'priortr','prune_layer':3,'keep_tokens':256},
                         {'name':'priortr_keep_all','enabled':True,'strategy':'priortr','prune_layer':3,'keep_tokens':1024}],
       'rows':rows,'source_sha256':{'admission.py':sha(Path(__file__).parent/'admission.py'),'freeze_sample.py':sha(__file__)}}
 with out.open('x',encoding='utf8') as f:json.dump(plan,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'frozen','n':len(rows),'manifest_sha256':sha(out)}))
if __name__=='__main__':main()
