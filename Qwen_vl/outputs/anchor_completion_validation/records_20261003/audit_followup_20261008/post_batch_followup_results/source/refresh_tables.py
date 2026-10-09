"""Refresh the prescribed three-row table, never select old/new by score."""
from pathlib import Path
import hashlib,subprocess,sys
OUT=Path(__file__).resolve().parent
TABLE=OUT.parents[1]/'submission_tables_20261009'
base=TABLE/'build_tables.py'
if hashlib.sha256(base.read_bytes()).hexdigest()!='1b718ec81ce246af17174682f413843005c28740d0d8174de52c0b3009fd92a8':raise RuntimeError('Frozen table builder changed')
text=base.read_text()
insert='''# Preserve exact official per-question reduction for old v15 VizWiz cells.
for arm,(budget,method) in arm_info.items():
    p=rescore/f'vizwiz_v15_{arm}_official.json'
    if not p.exists():continue
    d=read(p)
    if d.get('summary',{}).get('complete'):
        put('v15',budget,method,'VizWiz',100*sum(r['acc'] for r in d['per_question'])/len(d['per_question']),'existing',p)
# Prespecified followup: replace only when the whole selected pair is complete.
fpath=BASE/'post_batch_followup_20261009/registered_followup/followup.state.json'
if fpath.exists():
    fs=read(fpath)
    for group in fs.get('groups',{}).values():
        if group['status']!='complete':continue
        for key in group['job_ids']:
            job=fs['jobs'][key];score=read(job['output_paths']['score'])
            assert score['success'] and score['n']==job['n'] and sha(job['output_paths']['prediction'])==score['prediction_sha256']
            status='release_default_control' if job['task']=='vizwiz' else 'complete_same_recipe_followup'
            put(job['model'],128,job['method'],{'gqa':'GQA','vizwiz':'VizWiz'}[job['task']],score['value'],status,job['output_paths']['score'])
'''
text=text.replace('# Qwen uses the already audited',insert+'\n# Qwen uses the already audited')
text=text.replace("'Strict exclusive-GPU vs concurrent-GPU comparison is not established by this table.'", "'Strict exclusive-GPU vs concurrent-GPU comparison is not established by this table.', 'Completed VizWiz K128 followup cells use predeclared public validation defaults alpha0/beta1, both EADP and ours; not verified paper argv. Other budget cells retain their explicitly recorded original recipes.'")
text=text.replace("Reported rows are references, not local reproductions.'", "Reported rows are references, not local reproductions. Completed VizWiz K128 followup: alpha0/beta1 public defaults, not proven paper argv.'")
exec(compile(text,str(base)+'[followup-table-extension]','exec'),{'__name__':'__main__','__file__':str(base)})
# Rendering is CPU-only, using the installed plotting environment, not changing the model environment.
subprocess.run(['/home/dell/miniconda3/bin/python',str(TABLE/'render_tables.py')],check=True)
