"""CPU-only submission snapshot; completed scores take precedence over old cells.

Other reported baseline rows are parsed from the verified arXiv v1 tables.
No training, generation, queue mutation, or selection by score is performed.
"""
from pathlib import Path
import csv
import datetime
import hashlib
import html
import json
from decimal import Decimal, ROUND_HALF_UP

ROOT = Path('/media/disk2/YZX/research/EADP_amp')
BASE = ROOT / 'Qwen_vl/outputs/audit_followup_20261008'
OUT = Path(__file__).resolve().parent
STAMP = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
SOURCES = {}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    p = Path(p)
    raw = p.read_bytes()
    SOURCES[str(p)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

def cell(v, status='complete', source=None):
    return dict(value=v, status=status, source=str(source) if source else None)

LL = ['GQA', 'VizWiz', 'SQA_IMG', 'TextVQA', 'POPE', 'MME', 'MMB_EN', 'MMB_CN', 'MMVet', 'Avg7']
QW = ['TextVQA','ChartQA','AI2D','OCRBench','HallusionBench','MME','MMB_EN','MMB_CN','DocVQA','InfoVQA','Avg10']
local = {}

def put(model, budget, method, task, value, status='complete', source=None):
    local.setdefault((model, budget, method), {})[task] = cell(value,status,source)

# Legacy rows retained only with an explicit stale marker where a rerun is pending.
rescore = ROOT/'Qwen_vl/outputs/stage1_roundtrip_pilot/legacy_full/audit_20261008/rescored'
arm_info = {'FULL':(0,'FULL'), 'EGATHER':(128,'EADP'), 'LRMAIN025':(128,'AnchorZip'), 'LRMAIN0125':(64,'AnchorZip'), 'LRMAIN00625':(32,'AnchorZip')}
for model in ['v15','next']:
    for arm,(budget,method) in arm_info.items():
        for task in ['gqa','vizwiz','pope']:
            p = rescore/f'{task}_{model}_{arm}_official.json'
            if not p.exists() or (task=='pope' and not((model=='v15' and arm=='FULL') or (model=='next' and arm in ['LRMAIN025','LRMAIN0125']))):
                continue
            d = read(p)
            if task=='pope' and 'average_f1' in d:
                assert sum(d[k]['n_pred'] for k in ['random','popular','adversarial'])==8910
                put(model,budget,method,'POPE',d['average_f1'],'stale_pending_rerun',p)
                continue
            if not d.get('summary',{}).get('complete'): continue
            if task=='gqa':
                value = sum(x['correct'] for x in d['per_question'])/len(d['per_question'])*100
            elif task=='pope':
                value=d['summary']['average_f1']
            else: value=d['summary']['accuracy']
            # v1.5 VizWiz is not in the affected wait-only queue.
            status = 'existing' if task=='vizwiz' and model=='v15' else 'stale_pending_rerun'
            put(model,budget,method,{'gqa':'GQA','vizwiz':'VizWiz','pope':'POPE'}[task],value,status,p)

summary_path=BASE/'rerun_batch/repaired_results_summary.json'
summary=read(summary_path)
assert not summary['errors']
for r in summary['rows']:
    if r['model']!='v15' or r['status']!='complete': continue
    assert r['generated_n']==r['expected_n']
    assert sha(r['output'])==r['prediction_sha256']
    put('v15',r['budget_parameter'],r['method'],{'sqa':'SQA_IMG','textvqa':'TextVQA'}[r['task']],r['accuracy'],'complete',summary_path)

# Exact macro F1 from integer confusion counts (the saved headline is rounded).
for method,arm in [('EADP','EADP32'),('AnchorZip','AZ32')]:
    p=BASE/f'rerun_fast/pope_v15_{arm}_streamwait.official.score.json'
    d=read(p)
    assert d['summary']['complete'] and d['summary']['n']==8910
    assert sha(d['inputs']['predictions']['path'])==d['inputs']['predictions']['sha256']
    cats=[d['categories'][k] for k in ['random','popular','adversarial']]
    value=sum(200*x['TP']/(2*x['TP']+x['FP']+x['FN']) for x in cats)/3
    put('v15',32,method,'POPE',value,'complete',p)

for method,arm in [('EADP','EADP'),('AnchorZip','AZ')]:
    stem=BASE/f'next_gap_diagnosis_20261008/stability/next_text_K32_{arm}_beta2_streamwait'
    p=Path(str(stem)+'.official.score.json');d=read(p)
    assert d['success'] and d['n']==5000
    assert sha(str(stem)+'.jsonl')==d['prediction_sha256']
    put('next',32,method,'TextVQA',d['accuracy_percent'],'complete',p)

queues={}
for name in ['streamwait_repair_continuation','legacy_streamwait_continuation']:
    p=BASE/f'streamwait_continuation_20261008/{name}.state.json'
    state=read(p)
    queues[name]={'status':state['status'],'jobs':{k:j['status'] for k,j in state['jobs'].items()}}
    for key,j in state['jobs'].items():
        if j['status']!='complete': continue
        paths=j['output_paths'];s=read(paths['score'])
        assert s['success'] and s['n']==j['n'],key
        assert sha(paths['prediction'])==s['prediction_sha256'],key
        assert sha(paths['runtime'])==s['runtime_sha256'],key
        st=read(paths['state']);fin=read(paths['finished'])
        assert st.get('exit_code')==0 or st.get('generation_exit_code')==0,(key,list(st))
        assert fin.get('success'),key
        task={'sqa':'SQA_IMG','textvqa':'TextVQA','mme':'MME','pope':'POPE','gqa':'GQA','vizwiz':'VizWiz','mmben':'MMB_EN','mmbcn':'MMB_CN'}[j['task']]
        value=s.get('value',s.get('accuracy_percent'))
        assert value is not None,key
        status='protocol_unresolved' if task=='MMB_CN' else 'complete'
        put(j['model'],j['budget'],j['method'],task,value,status,paths['score'])

# Qwen uses the already audited official-legacy path; no new Qwen GPU run.
p=ROOT/'Qwen_vl/outputs/stage1_roundtrip_pilot/legacy_full/p1_unified_20261008.json'
d=read(p)
for method,prefix in [('EADP','eg'),('AnchorZip','lr')]:
    for task,v in d['avg10'][prefix+'_columns'].items(): put('qwen',256,method,task,v,'existing',p)
    put('qwen',256,method,'Avg10',d['avg10'][prefix],'existing',p)
status_path=ROOT/'Qwen_vl/outputs/anchor_completion_validation/records_20261003/STATUS_20261003.md'
SOURCES[str(status_path)]=sha(status_path)
for budget,values in [(128,[67.38,42.64,73.09,555,38.44,2148.5,78.95,77.17,45.95,36.03,62.26]),(64,[59.15,33.04,70.98,495,34.68,2088.3,78.02,77.24,32.52,30.52,57.01])]:
    for task,value in zip(QW,values): put('qwen',budget,'AnchorZip',task,value,'existing_rounded_status',status_path)

paper_path=BASE/'paper_reference.json';paper=read(paper_path)
tables=paper['tables']
reported={}
for table in tables:
    model={1:'v15',2:'next',4:'qwen'}[table['table']]
    for r in table['rows']:
        budget=0 if r['method']=='FULL' else r['budget_per_crop_equivalent']
        vals={k:cell(v,'reported',paper_path) for k,v in r['values'].items() if k!='Avg'}
        if model=='qwen': vals['Avg10']=cell(r['values']['Avg'],'reported',paper_path)
        else:
            tasks=['GQA','SQA_IMG','TextVQA','POPE','MME','MMB_EN','MMB_CN']
            vals['Avg7']=cell(sum(r['values'][k]/(20 if k=='MME' else 1) for k in tasks)/7,'derived_from_reported',paper_path)
        reported[model,budget,r['method']]=vals

# Reported comparison rows are read from the verified arXiv v1 HTML tables.
import re
paper_other={}
for number,model in [(1,'v15'),(2,'next'),(4,'qwen')]:
    path=OUT/f'paper_table_{number}.json'
    original_rows=read(path)
    budget=None
    for row in original_rows[1:]:
        if len(row)==1:
            match=re.search(r'Retain ([0-9,]+) Tokens',row[0])
            budget=int(match.group(1).replace(',','')) if match else None
            if budget is not None and model=='next':budget//=5
            continue
        if budget is None or row[0].startswith('EADP'):continue
        name=row[0].split('(')[0].strip()
        cols=QW if model=='qwen' else LL[:-1]
        vals=row[1:] if model=='qwen' else row[2:-1]
        assert len(cols)==len(vals),(number,name)
        cs={k:cell(float(re.search(r'\d+(?:\.\d+)?',v).group()),'reported_paper',path) for k,v in zip(cols,vals)}
        if model!='qwen':
            cs['VizWiz']['validation_star']='*' in vals[1] or '∗' in vals[1]
            cs['Avg7']=cell(sum(cs[k]['value']/(20 if k=='MME' else 1) for k in ['GQA','SQA_IMG','TextVQA','POPE','MME','MMB_EN','MMB_CN'])/7,'derived_from_reported',path)
        paper_other.setdefault((model,budget),[]).append(dict(model=model,budget=budget,method=name,kind='reported',cells=cs))
SOURCES[str(OUT/'paper_tables_source.html')]=sha(OUT/'paper_tables_source.html')
attachments=[Path('/home/dell/.codex/attachments/2592c72d-5932-4eb7-b533-5bfa2c966a22/codex-clipboard-7701d166-7fca-4710-86c5-5349c98c310b.png'),Path('/home/dell/.codex/attachments/dfb4bd31-d32f-4aec-a218-2d3f20b87f0a/codex-clipboard-bd00b8c7-42d3-490c-989e-00e7fe6e9fff.png'),Path('/home/dell/.codex/attachments/27b99e9d-8955-47f1-9048-3b06403f91bb/codex-clipboard-1da31ab0-5166-47ea-b195-0b3caf5155ae.png')]
for p in attachments: SOURCES[str(p)]=sha(p)

def other_rows(model,budget):
    return paper_other.get((model,budget),[])

rows=[]
for model,budgets in [('v15',[0,128,64,32]),('next',[0,128,64,32]),('qwen',[0,512,256,128,64])]:
    for budget in budgets:
        if budget: rows.extend(other_rows(model,budget))
        method='FULL' if budget==0 else 'EADP'
        if (model,budget,method) in reported:
            rows.append(dict(model=model,budget=budget,method=method+' (reported)',kind='reported',cells=reported[model,budget,method]))
        for method in (['FULL'] if budget==0 else ['EADP','AnchorZip']):
            cs=local.get((model,budget,method))
            if not cs: continue
            # A CN value generated with English instructions is audit-only.
            for task in ['MMB_EN','MMB_CN','MMVet','Avg7'] if model!='qwen' else []:
                cs.setdefault(task,cell(None,'unavailable_or_protocol_pending'))
            rows.append(dict(model=model,budget=budget,method=method+' (ours)' if method=='AnchorZip' else method+' (reproduced)',kind='local',cells=cs))

def display(c,task,latex=False):
    if c is None or c['value'] is None or c['status']=='protocol_unresolved': return '--' if latex else '—'
    v=c['value'];v=str(Decimal(str(v)).quantize(Decimal('1') if task=='OCRBench' else Decimal('0.1'),rounding=ROUND_HALF_UP))
    if c['status']=='stale_pending_rerun': v+=r'$^{\dagger}$' if latex else '†'
    if task=='VizWiz' and (c.get('validation_star') or c['status'] not in ['reported_attachment','reported_paper']): v+=r'$^{*}$' if latex else '*'
    return v

names={'v15':'LLaVA-1.5-7B','next':'LLaVA-NeXT-7B','qwen':'Qwen3-VL-8B'}
def btext(model,budget): return 'FULL' if not budget else str(budget*5 if model=='next' else budget)

snapshot={'snapshot_beijing':STAMP,'selection_policy':'Use completed prescribed repaired run; never choose the higher result. Dagger explicitly retains old result pending rerun.','queues':queues,'sources_sha256':SOURCES,'rows':rows,'notes':[
    'SQA image subset: 2017 questions, shipped CQM-A input. Authors actual CQM-I file is unavailable.',
    'POPE: three-category mean F1; random2910/popular3000/adversarial3000.',
    'LLaVA MME is perception only, Qwen MME is perception+cognition.',
    'NeXT budgets640/320/160 are nominal equivalents of CLI128/64/32; realized counts vary.',
    'Missing or unresolved MMB cells keep Avg7 empty. No available-task average is substituted.',
    'NeXT MMB-CN wait-only reruns retain lang=en and do not resolve the language issue.',
    'Qwen128/64 use previously completed rounded status entries, not fresh generation; K64 TextVQA59.15 is conservatively displayed59.2 as in supplied table.',
    'Qwen EADP HallusionBench recorded/current bank SHA remains unresolved; score is existing audited scoring result, not complete generation provenance.',
    'Other reported methods are parsed from arXiv2607.02484v1 Tables1/2/4; no new reproduction is implied.',
    'Strict exclusive-GPU vs concurrent-GPU comparison is not established by this table.'
]}
(OUT/'latest_tables.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
with (OUT/'latest_tables.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['model','budget_nominal','method','task','value','display','status','source'])
    for row in rows:
        for task in (QW if row['model']=='qwen' else LL):
            c=row['cells'].get(task,cell(None,'unavailable'))
            w.writerow([names[row['model']],btext(row['model'],row['budget']),row['method'],task,c['value'],display(c,task),c['status'],c['source']])

notes='Snapshot: '+STAMP+'. Dagger: old complete result, repaired rerun pending; not a repaired final score. Dash: unavailable or protocol unresolved. Star: VizWiz validation. SQA uses the shipped CQM-A input (2017 image questions). POPE uses mean F1 over 2910/3000/3000 questions. LLaVA MME: perception; Qwen MME: perception + cognition. NeXT budgets are nominal. Reported rows are references, not local reproductions.'
tex=[r'\documentclass[10pt,a4paper]{article}',r'\usepackage[margin=12mm]{geometry}',r'\usepackage{booktabs,graphicx,array}',r'\usepackage[table]{xcolor}',r'\pagestyle{empty}',r'\setlength{\parindent}{0pt}',r'\begin{document}']
web=['<!doctype html><meta charset="utf-8"><title>Updated benchmark tables</title><style>body{font-family:Georgia,serif;margin:30px;color:#17202a}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}th,td{padding:5px 7px;text-align:right;border-bottom:1px solid #ddd}th:first-child,td:first-child{text-align:left}thead{border-top:2px solid;border-bottom:2px solid}.budget{background:#e7e9ec;text-align:center}.ours{font-weight:bold;background:#f0f6ff}p{font:14px/1.5 sans-serif;color:#52606d}h2{margin-top:40px}@media print{section{break-after:page}body{margin:0}table{font-size:10px}}</style>', '<p>'+html.escape(notes)+'</p>']
core=[]
for index,model in enumerate(['v15','next','qwen'],1):
    cols=QW if model=='qwen' else LL
    title=f'Table {index}. {names[model]} -- updated results'
    tex.extend([r'{\large\textbf{'+title+r'}}\par\medskip',r'\begingroup\setlength{\tabcolsep}{3.2pt}\renewcommand{\arraystretch}{1.03}',r'\resizebox{\textwidth}{!}{\begin{tabular}{l'+'r'*len(cols)+'}',r'\toprule','Method & '+' & '.join(x.replace('_',r'\_') for x in cols)+r' \\ \midrule'])
    web.append('<section><h2>'+title+'</h2><table><thead><tr><th>Method</th>'+''.join('<th>'+x+'</th>' for x in cols)+'</tr></thead><tbody>')
    core.append(names[model]+'\n\n| 预算 | 方法 | '+' | '.join(cols)+' |\n|---|---|'+'---:|'*len(cols))
    last=None
    for row in [r for r in rows if r['model']==model]:
        budget=row['budget']
        if budget!=last:
            bt='Full model' if not budget else 'Retain '+btext(model,budget)+' tokens'+(' (nominal)' if model=='next' else '')
            tex.append(r'\rowcolor{black!12}\multicolumn{'+str(len(cols)+1)+r'}{c}{\textit{'+bt+r'}} \\')
            web.append('<tr><td class="budget" colspan="'+str(len(cols)+1)+'">'+bt+'</td></tr>');last=budget
        values=[display(row['cells'].get(k),k,True) for k in cols]
        method=row['method']
        if 'AnchorZip' in method:
            values=[r'\textbf{'+v+'}' for v in values];method=r'\textbf{'+method+'}'
        tex.append(method+' & '+' & '.join(values)+r' \\')
        web.append('<tr'+(' class="ours"' if 'AnchorZip' in row['method'] else '')+'><td>'+row['method']+'</td>'+''.join('<td>'+display(row['cells'].get(k),k)+'</td>' for k in cols)+'</tr>')
        if row['kind']=='local' or row['method'] in ['EADP (reported)','FULL (reported)']:
            core.append('| '+btext(model,budget)+' | '+row['method']+' | '+' | '.join(display(row['cells'].get(k),k) for k in cols)+' |')
    tex.extend([r'\bottomrule\end{tabular}}\endgroup\par\medskip',r'{\footnotesize '+notes.replace('_',r'\_')+r'\par}',r'\smallskip{\footnotesize '+(r'$\mathrm{Avg}_{7}=(\mathrm{GQA}+\mathrm{SQA}_{IMG}+\mathrm{TextVQA}+\mathrm{POPE}+\mathrm{MME}/20+\mathrm{MMB}_{EN}+\mathrm{MMB}_{CN})/7$. Missing or unresolved inputs leave the average blank.' if model!='qwen' else r'$\mathrm{Avg}_{10}=(\text{eight percentage columns}+\mathrm{OCRBench}/10+\mathrm{MME}/20)/10$. Qwen results are prior completed official-legacy runs; EADP K128/K512 local baselines are unavailable. EADP K256 HallusionBench generation-bank provenance remains unresolved.')+r'\par}'])
    web.append('</tbody></table></section>')
    if model!='qwen':tex.append(r'\newpage')
tex.append(r'\end{document}')
# amsmath supplies text in the averaging formula.
tex.insert(3,r'\usepackage{amsmath}')
(OUT/'latest_tables.tex').write_text('\n'.join(tex)+'\n')
(OUT/'latest_tables.html').write_text('\n'.join(web))
(OUT/'core_tables.txt').write_text('Snapshot '+STAMP+'\n\n'+'\n'.join(core)+'\n\n'+notes+'\n')

# Old values below are the rounded cells in the user's screenshots. Differences
# describe changes of evaluation/input/execution, not method improvements.
old_ours={('v15',128):[66.0,56.7,87.5,1350.4],('v15',64):[66.1,55.2,86.5,1312.9],('v15',32):[66.7,52.6,80.0,1043.0],('next',128):[67.3,57.6,87.5,1358.2],('next',64):[66.8,55.9,85.8,1328.9],('next',32):[66.5,53.2,82.6,1269.2]}
changes=[]
for (model,budget),old in old_ours.items():
    for task,previous in zip(['SQA_IMG','TextVQA','POPE','MME'],old):
        c=local[model,budget,'AnchorZip'][task]
        done=c['status']=='complete'
        changes.append({'model':names[model],'budget':btext(model,budget),'task':task,'old_screenshot_rounded':previous,'new_exact':c['value'] if done else None,'new_display':display(c,task) if done else 'Pending','delta_vs_rounded_old':c['value']-previous if done else None,'status':c['status'],'source':c['source']})
with (OUT/'ours_old_vs_new.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(changes[0]));w.writeheader();w.writerows(changes)
changes_html='<!doctype html><meta charset="utf-8"><title>Ours: old vs repaired</title><style>body{font:15px/1.6 sans-serif;margin:30px}table{border-collapse:collapse}th,td{padding:6px 14px;border-bottom:1px solid #ccc;text-align:right}th:first-child,td:first-child{text-align:left}</style><h2>AnchorZip: old screenshot vs completed repaired run</h2><p>'+STAMP+'. Old screenshot cells are rounded; differences are approximate and are not method gains. All increases and decreases are retained.</p><table><tr><th>Model</th><th>Budget</th><th>Task</th><th>Old</th><th>New</th><th>Change</th></tr>'
for c in changes:
    delta='Pending' if c['delta_vs_rounded_old'] is None else f"{c['delta_vs_rounded_old']:+.2f}"
    changes_html+='<tr>'+''.join('<td>'+str(x)+'</td>' for x in [c['model'],c['budget'],c['task'],c['old_screenshot_rounded'],c['new_display'],delta])+'</tr>'
(OUT/'ours_old_vs_new.html').write_text(changes_html+'</table>')
print(json.dumps({'snapshot_beijing':STAMP,'rows':len(rows),'sources':len(SOURCES),'phase2_complete':sum(s=='complete' for s in queues['legacy_streamwait_continuation']['jobs'].values()),'files':[str(OUT/x) for x in ['latest_tables.tex','latest_tables.html','latest_tables.csv','latest_tables.json','core_tables.txt']]},ensure_ascii=False,indent=2))
