"""CPU paired FULL/K128 analysis; no prediction is read before terminal gates."""
import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import string
import subprocess
import sys

sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parent
FULL=BASE/'full_token_baseline_control'
OUTPUT=BASE/'full_vs_pruned_complete'
EXPECTED_FULL_PROTOCOL='70911f00b89d34999d46ef754782de967877f8d881b8d70439d7b3dd5329e635'
RUNS={'FULL':FULL,'K128_beta1':BASE/'streamwait_control','K128_beta2':BASE/'beta2_control'}
read=lambda p:json.loads(Path(p).read_text())


class Pending(Exception):pass


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def ticks(pid):
    try:return int(Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19])
    except FileNotFoundError:return None


def gates():
    # Control-plane files only until FULL generation, verification, release and
    # the original controller have all terminated. No runtime/raw-prefix reads.
    if (FULL/'failure.json').exists():raise AssertionError('Preserved FULL terminal failure; no paired scoring')
    required=['state.json','release.json','independent_verification.json']
    try:state=read(FULL/'state.json')
    except (FileNotFoundError,json.JSONDecodeError) as error:raise Pending('FULL control record absent/being written') from error
    if state.get('exit_code') not in (None,0):raise AssertionError('FULL native exit was nonzero')
    if state.get('status')!='complete' or any(not (FULL/n).exists() for n in required):
        raise Pending('Waiting for full4377 native/scoring/XLSX/independent/release proof')
    try:release=read(FULL/'release.json');verified=read(FULL/'independent_verification.json')
    except json.JSONDecodeError as error:raise Pending('FULL terminal record is being written') from error
    assert release['success'] and release['n_rows']==4377 and release['n_groups']==1176
    assert verified['success'] and verified['all_rows_checked'] and verified['n_rows']==4377 and verified['n_circular_groups']==1176
    assert verified['mask_observation_checks']==verified['generated_token_decoding_checks']==verified['complete_input_token_SHA_checks']==4377
    assert verified['actual_retained_tokens_per_row']==576 and verified['native_cli_visual_token_num']==0
    assert sha(FULL/'protocol.json')==EXPECTED_FULL_PROTOCOL==state['protocol_sha256']==release['protocol_sha256']
    assert state['FULL_independent_verification_sha256']==release['independent_verification_sha256']==sha(FULL/'independent_verification.json')
    assert state['release_sha256']==sha(FULL/'release.json')
    identity=state['runner_process'];current=ticks(identity['pid'])
    if current==identity['start_ticks']:raise Pending('FULL original controller is completing its terminal return')
    gpu_text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    gpu={int(x.strip()) for x in gpu_text.splitlines() if x.strip()}
    exits=[]
    for role,original in [('controller',identity),('native_worker',state['native_process'])]:
        current=ticks(original['pid']);assert current!=original['start_ticks'] and original['pid'] not in gpu
        exits.append(dict(role=role,original=original,original_exited=True,current_start_ticks_if_reused=current,GPU_pid_absent=True))
    prior=module('completed_MMB_pruned_control_gate',BASE/'compare_beta_and_v11.py')
    prior.complete_gate()  # All completed beta1/beta2 metadata, author XLSX and actual exits; no raw reads.
    metadata={}
    for name,directory in RUNS.items():
        plan=read(directory/'protocol.json');state=read(directory/'state.json');verification=read(directory/'independent_verification.json')
        assert state['status']=='complete' and state['exit_code']==0 and state['n_answers']==4377
        assert verification['success'] and verification['n_rows']==4377 and verification['n_circular_groups']==1176
        assert sha(directory/'protocol.json')==state['protocol_sha256']==verification['protocol_sha256']
        assert plan['dataset']==read(FULL/'protocol.json')['dataset']
        assert plan['parameters']['alpha']==.5 and plan['parameters']['beta']==(2 if name=='K128_beta2' else 1)
        assert plan['parameters']['K']==(0 if name=='FULL' else 128)
        for item in plan['source_files']:assert sha(item['path'])==item['sha256'],item['path']
        command=plan['command'];prediction=Path(command[command.index('--answers-file')+1])
        # Raw hashing happens only after all full terminal gates above.
        assert sha(prediction)==verification['prediction_sha256']==state['prediction_sha256']
        assert sha(directory/'local_circular_score.json')==state['score_sha256']
        conversion=read(directory/'conversion.json');assert conversion['exit_code']==0 and not conversion['submitted_to_external_server']
        assert sha(conversion['output'])==verification['submission_sha256']
        native=state['native_process'];current=ticks(native['pid']);assert current!=native['start_ticks'] and native['pid'] not in gpu
        metadata[name]=dict(plan=plan,state=state,verification=verification,prediction=str(prediction),directory=str(directory))
    reference=metadata['K128_beta1']['plan']
    for name,item in metadata.items():
        plan=item['plan'];expected=dict(reference['parameters'],K=(0 if name=='FULL' else 128),beta=(2 if name=='K128_beta2' else 1))
        assert plan['parameters']==expected
        assert plan['image_manifest_sha256']==reference['image_manifest_sha256']
        # Worker/output directories differ; every native generation option is
        # otherwise identical except the declared K and beta control factors.
        argv=list(plan['command'][3:]);baseline=list(reference['command'][3:])
        for flag in ('--answers-file','--visual_token_num','--beta'):
            for command in (argv,baseline):command[command.index(flag)+1]='<declared control/output>'
        assert argv==baseline
    return metadata,exits


def transitions(a,b):
    return dict(correct_to_correct=sum(x and y for x,y in zip(a,b)),
                wrong_to_correct=sum(not x and y for x,y in zip(a,b)),
                correct_to_wrong=sum(x and not y for x,y in zip(a,b)),
                wrong_to_wrong=sum(not x and not y for x,y in zip(a,b)))


def compare(metadata,exits):
    import pandas as pd
    public=module('released_public_MMB_pair_helpers',BASE/'public_mcq_metric_check.py')
    namespace=dict(cp=copy,os=type('OfflineOS',(),{'environ':{}})(),re=re,string=string,pd=pd)
    helper_sources=[public.original_helpers(public.UTILS/'utils/matching_util.py',['can_infer_option','can_infer_text','can_infer'],namespace),
                    public.original_helpers(public.UTILS/'dataset/utils/multiple_choice.py',['build_choices','prefetch_answer','prefetch_circular_group'],namespace)]
    spec=metadata['FULL']['plan']['dataset'];assert sha(spec['path'])==spec['sha256']
    data=pd.read_table(spec['path']);assert len(data)==4377 and data['index'].is_unique
    native=Path(metadata['FULL']['plan']['command'][3]);tree=ast.parse(native.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('is_none','get_options')]
    assert len(nodes)==2;native_helpers=dict(math=math)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(native)+'[unchanged option helpers]','exec'),native_helpers)
    all_rows={};all_groups={};scores={};raw_answers={};source_paths=[]
    for name,item in metadata.items():
        directory=Path(item['directory']);prediction=Path(item['prediction'])
        answers=[json.loads(line) for line in prediction.read_text().splitlines() if line.strip()]
        assert len(answers)==4377 and [int(a['question_id']) for a in answers]==data['index'].tolist()
        frame=data.copy();frame['prediction']=[a['text'] for a in answers];frame['GT']=frame['answer'];frame['g_index']=[int(i%1000000) for i in frame['index']]
        rows=[]
        for position,((_,truth),answer) in enumerate(zip(frame.iterrows(),answers)):
            options=native_helpers['get_options'](truth,list('ABCD'))
            prompt=truth['question']
            if not native_helpers['is_none'](truth['hint']):prompt=truth['hint']+'\n'+prompt
            for letter,option in zip('ABCD',options):prompt+='\n'+letter+'. '+option
            assert answer['prompt']==prompt and answer['options']==options and answer['option_char']==list('ABCD')[:len(options)] and answer['round_id']==0
            assert answer['model_id']=='llava-v1.5-7b'
            parsed=namespace['prefetch_answer'](truth)
            rows.append(dict(position=position,index=int(truth['index']),base_group=int(truth['g_index']),GT=truth['answer'],
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),options=options,raw_answer=answer['text'],public_option=parsed,public_correct=parsed==truth['answer']))
        groups={};unresolved=[]
        for idx,group in frame.groupby('g_index',sort=True):
            result=namespace['prefetch_circular_group'](group)
            if result is None:unresolved.append(int(idx));hit=0
            else:hit=result['hit']
            assert hit in (0,1);groups[int(idx)]=bool(hit)
        assert len(groups)==1176
        value=100*sum(groups.values())/1176
        if name=='FULL':assert abs(value-item['verification']['primary_value_percent'])<1e-12
        scores[name]=dict(released_public_circular_percent=value,public_correct_groups=sum(groups.values()),public_unresolved_groups=unresolved,
            preserved_local_exact_letter_percent=item['verification']['circular_accuracy_percent'],
            original_reported_public_percent=item['verification'].get('released_public_circular_accuracy_percent'))
        all_rows[name]=rows;all_groups[name]=groups;raw_answers[name]=answers
        source_paths.extend([directory/n for n in ['protocol.json','state.json','independent_verification.json','local_circular_score.json','conversion.json']])
        source_paths.append(prediction)
    pair_rows=[]
    for position in range(4377):
        records={name:rows[position] for name,rows in all_rows.items()}
        common=records['FULL']
        for row in records.values():
            for k in ('position','index','base_group','GT','prompt_sha256','options'):assert row[k]==common[k]
        pair_rows.append(dict(index=common['index'],base_group=common['base_group'],GT=common['GT'],prompt_sha256=common['prompt_sha256'],
                              runs={name:{k:v for k,v in row.items() if k in ('raw_answer','public_option','public_correct')} for name,row in records.items()}))
    group_ids=sorted(all_groups['FULL']);paired={}
    for name in ['K128_beta1','K128_beta2']:
        assert sorted(all_groups[name])==group_ids
        paired[name+'_to_FULL']=dict(direction='Pruned control -> FULL; wrong_to_correct means restored by FULL',
            row_correctness_transitions=transitions([r['public_correct'] for r in all_rows[name]],[r['public_correct'] for r in all_rows['FULL']]),
            circular_group_transitions=transitions([all_groups[name][g] for g in group_ids],[all_groups['FULL'][g] for g in group_ids]),
            raw_answer_changed_rows=sum(a['raw_answer']!=b['raw_answer'] for a,b in zip(all_rows[name],all_rows['FULL'])),
            FULL_minus_pruned_pp=scores['FULL']['released_public_circular_percent']-scores[name]['released_public_circular_percent'])
    OUTPUT.mkdir(exist_ok=False)
    rows_path=OUTPUT/'paired_rows.jsonl';rows_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pair_rows))
    groups_path=OUTPUT/'paired_groups.jsonl';groups_path.write_text(''.join(json.dumps(dict(base_group=g,runs={name:all_groups[name][g] for name in RUNS}))+'\n' for g in group_ids))
    source_paths.extend([FULL/'release.json',Path(spec['path']),Path(__file__)])
    report=dict(status='complete',success=True,CPU_only=True,model_loaded=False,GPU_or_API_used=False,n_rows=4377,n_groups=1176,
        scores=scores,paired=paired,actual_FULL_process_exits=exits,completed_utc=datetime.now(timezone.utc).isoformat(),
        sources=[dict(path=str(p),sha256=sha(p)) for p in source_paths],unchanged_public_helper_sources=helper_sources,
        paired_rows=dict(path=str(rows_path),sha256=sha(rows_path)),paired_groups=dict(path=str(groups_path),sha256=sha(groups_path)),
        limits=['All three full4377 controls share exact dataset bytes, ordered question IDs/native prompts/options/GT and the same three-wait source; no V11 subset is substituted.',
                'The FULL bypass intentionally omits pruning/text-guidance arithmetic; this is not K576 facility execution.',
                'Public exact-matching circular is fixed in advance; all original local metrics and raw answers are retained. No higher score or score-direction gate is used.',
                'No external matching server/API or unpublished paper argv identity is claimed. Historical/current runtime labels are not backfilled.'])
    (OUTPUT/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    assert 'torch' not in sys.modules
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    try:
        metadata,exits=gates()
        if not args.run:print(json.dumps(dict(ready=True,raw_answers_parsed=False,scoring_performed=False)))
        else:
            result=compare(metadata,exits);print(json.dumps({k:result[k] for k in ['status','n_rows','n_groups','scores','paired']}))
    except Pending as error:
        print(json.dumps(dict(ready=False,status='pending_full_terminal',reason=str(error),raw_answers_parsed=False,scoring_performed=False)))
