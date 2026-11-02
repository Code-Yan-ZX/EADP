"""Explicit post-exit local scoring; never invoked by generation controller.

Gate validates full generation and root-observed actual exit before first access
of the answer-bearing TSV. No online evaluation server or judge is used.
"""
import ast,copy as cp,hashlib,json,os,re,string,sys
from pathlib import Path
import common as c

GT=c.HERE.parent/'data/mmbench_dev_20230712.tsv'
GT_SHA='52d45f29876e56138d3d4feb169dc9a7323f6ed2ee3f439ae7e4af94192b3324'
UTILS=c.HERE.parents[3]/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit/vlmeval'
ARMS=('source','question_only')

def check_proof():
    plan=c.read(c.PLAN);c.check(plan);plan_sha=c.sha(c.PLAN)
    proof=c.read(c.HERE/'root_exit_proof.json');state=c.read(c.OUT/'state.json');release=c.read(c.OUT/'native_release.json')
    claim=c.read(c.HERE/'launch.claim.json');identity=c.read(c.OUT/'worker_identity.json')
    report=c.read(c.OUT/'generation_verification.json');complete=c.read(c.OUT/'native_complete.json')
    assert proof['root_pty_actual_exit_code']==0 and proof['root_pty_exit_observed_by']=='functions.write_stdin'
    assert type(proof['root_pty_session']) is int and proof['root_pty_session']>0 and proof['verifier_container_not_running']
    assert proof['state_sha256']==c.sha(c.OUT/'state.json') and proof['verifier_sha256']==c.sha(c.OUT/'generation_verification.json')
    allocation_path=c.HERE/'root_gpu_allocation.json';allocation=c.read(allocation_path)
    assert proof['allocation_sha256']==state['allocation_sha256']==release['allocation_sha256']==claim['allocation_sha256']==identity['allocation_sha256']==c.sha(allocation_path)
    assert allocation['authorization']=='explicit_root_gpu_allocation' and allocation['run_id']==plan['run_id'] and allocation['plan_sha256']==plan_sha
    assert state['status']=='complete' and state['native_exit_code']==state['verifier_exit_code']==0 and state['verifier_container_absent']
    assert release['status']=='native_exited_validation_pending' and release['native_exit_code']==0 and release['original_worker_exited'] and release['GPU_pid_absent']
    assert state['plan_sha256']==release['plan_sha256']==claim['plan_sha256']==identity['plan_sha256']==report['plan_sha256']==complete['plan_sha256']==plan_sha
    assert identity['identity']==release['worker']==state['worker'] and identity['parent']==release['controller']==state['controller']==claim['controller']
    processes={x['role']:x for x in proof['process_checks']}
    assert len(proof['process_checks'])==3 and set(processes)=={'controller','worker','verifier_client'}
    for role,actual in (('controller',state['controller']),('worker',state['worker']),('verifier_client',state['verifier_client'])):
        assert processes[role]['original']==actual and processes[role]['original_process_absent'] and c.identity(actual['pid'])!=actual
    assert report['success'] and report['n_rows']==4377 and report['n_generations']==8754 and report['n_anchor_rows']==60
    assert report['CPU_only'] and not report['model_loaded'] and not report['CUDA_initialized'] and not report['GT_read'] and not report['scoring_started']
    assert complete['status']=='complete' and complete['n_rows']==4377 and complete['n_generations']==8754 and not complete['scoring']
    assert report['native_complete_sha256']==c.sha(c.OUT/'native_complete.json') and report['native_release_sha256']==c.sha(c.OUT/'native_release.json')
    assert report['anchor_gate_sha256']==c.sha(c.OUT/'anchor_gate.json') and report['runtime_sha256']==c.sha(c.OUT/'runtime.json')
    assert complete['anchor_gate']['sha256']==report['anchor_gate_sha256'] and complete['runtime']['sha256']==report['runtime_sha256']
    rows=c.read(c.HERE/'inputs/manifest.json')['rows'];assert len(rows)==4377 and len({x['index'] for x in rows})==4377
    predictions={}
    for arm in ARMS:
        for key in ('predictions','observations'):
            entry=complete['ordered'][arm][key];path=c.OUT/(arm+'_'+key+'.jsonl')
            reported=report['prediction_sha256'][arm] if key=='predictions' else report['observation_sha256'][arm]
            assert Path(entry['path'])==path and c.sha(path)==entry['sha256']==reported
        path=c.OUT/(arm+'_predictions.jsonl');values=[json.loads(s) for s in path.read_text().splitlines()]
        assert len(values)==4377
        assert [(x['position'],x['index'],x['group_id']) for x in values]==[(i,r['index'],r['group_id']) for i,r in enumerate(rows)]
        predictions[arm]=values
    return plan,proof,report,rows,predictions

def original_helpers(path,names,namespace):
    tree=ast.parse(path.read_bytes(),filename=str(path));selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected}==set(names)
    module=ast.Module(body=selected,type_ignores=[]);exec(compile(module,str(path),'exec',dont_inherit=True,optimize=0),namespace)
    return dict(path=str(path),sha256=c.sha(path),functions=names,ast_sha256=hashlib.sha256(ast.dump(module).encode()).hexdigest())

def score():
    assert not (c.OUT/'score_terminal.json').exists()
    plan,proof,verification,rows,predictions=check_proof()
    # FIRST ground-truth-file access is here, after all generation/exit gates.
    assert plan['GT_file_expected_sha256']==GT_SHA
    assert c.sha(GT)==GT_SHA
    for path,expected in plan['score_only_files'].items():assert c.sha(path)==expected
    import pandas as pd
    data=pd.read_table(GT);assert len(data)==4377 and data['index'].is_unique
    assert data['index'].tolist()==[x['index'] for x in rows]
    assert set(data['index']%1000000)==set(data.loc[data['index']==data['index']%1000000,'index'])
    os.environ.pop('VERBOSE',None);namespace=dict(cp=cp,os=os,re=re,string=string,pd=pd)
    sources=[original_helpers(UTILS/'utils/matching_util.py',['can_infer_option','can_infer_text','can_infer'],namespace),
        original_helpers(UTILS/'dataset/utils/multiple_choice.py',['build_choices','prefetch_answer','prefetch_circular_group'],namespace)]
    scored={}
    for arm in ARMS:
        current=data.copy();current['prediction']=[x['text'] for x in predictions[arm]];current['GT']=current['answer'];current['g_index']=current['index']%1000000
        row_out=[];local_groups={}
        for _,item in current.iterrows():
            public=namespace['prefetch_answer'](item);m=re.search(r'\b([A-E])\b',item['prediction'].strip());local=m.group(1) if m else None
            local_hit=local==item['answer'];local_groups.setdefault(int(item['g_index']),[]).append(local_hit)
            row_out.append(dict(index=int(item['index']),group=int(item['g_index']),prediction=item['prediction'],answer=item['answer'],public=public,local=local,local_hit=local_hit,parser_disagrees=public!=local))
        groups=[]
        for group,items in current.groupby('g_index',sort=True):
            result=namespace['prefetch_circular_group'](items);hit=0 if result is None else result['hit'];assert hit in (0,1)
            groups.append(dict(group=int(group),public_hit=int(hit),public_unresolved=result is None,local_hit=int(all(local_groups[int(group)]))))
        assert len(groups)==1176 and len(row_out)==4377
        public_hits=sum(x['public_hit'] for x in groups);local_hits=sum(x['local_hit'] for x in groups)
        scored[arm]=dict(n_rows=4377,n_groups=1176,public_exact_matching_percent=100*public_hits/1176,local_exact_letter_percent=100*local_hits/1176,
            public_correct_groups=public_hits,local_correct_groups=local_hits,unresolved_public_groups=sum(x['public_unresolved'] for x in groups),parser_disagreements=sum(x['parser_disagrees'] for x in row_out),rows=row_out,groups=groups)
    c.save_new(c.OUT/'score_terminal.json',dict(success=True,plan_sha256=c.sha(c.PLAN),root_exit_proof_sha256=c.sha(c.HERE/'root_exit_proof.json'),
        generation_verification_sha256=c.sha(c.OUT/'generation_verification.json'),dataset_sha256=GT_SHA,helpers=sources,runs=scored,
        GT_read=True,API_used=False,model_loaded=False,CUDA_initialized=False,external_official_server_evaluated=False,
        limitations=['Public author-vendored VLMEvalKit exact-matching helpers and independent local exact-letter circular scores; external official server/judge was not run.',
         'Unresolved public circular groups count as zero under exact matching; local and public parsers may differ.']))
    print(json.dumps(dict(success=True,public={a:scored[a]['public_exact_matching_percent'] for a in ARMS},local={a:scored[a]['local_exact_letter_percent'] for a in ARMS})))

if __name__=='__main__':score()
