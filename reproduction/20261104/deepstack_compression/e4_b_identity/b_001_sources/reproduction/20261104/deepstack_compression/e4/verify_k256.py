"""Independent K256 E1 smoke comparison; never opens labels or scores."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from identity_worker import check_e1
from identity_worker import HERE, PLAN, sha, read_jsonl
from smoke_contract import select_smoke
from completion import alive
from decode_observer import validate_trace


def validate_smoke_terminal(run,started,terminal,launch,exited):
    if (terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or
        terminal.get('remaining_process_group')!=[] or terminal.get('worker_pid')!=started.get('worker_pid')):
        raise ValueError('smoke supervisor did not exit cleanly')
    controller=launch.get('controller');container=exited.get('container_identity')
    if (not isinstance(controller,dict) or not isinstance(container,dict) or
        controller!=exited.get('controller') or controller.get('pid')!=terminal.get('worker_pid') or
        not container.get('container_id') or not isinstance(container.get('host_init'),dict)):
        raise ValueError('smoke process identity mismatch')
    for ident in (controller,container['host_init']):
        if (not isinstance(ident.get('pid'),int) or ident['pid']<=0 or not isinstance(ident.get('start_ticks'),int)
                or ident['start_ticks']<=0 or not ident.get('boot_id') or alive(ident)):
            raise ValueError('smoke process is alive or identity incomplete')
    if (exited.get('returncode')!=0 or exited.get('container_exit_code')!=0 or exited.get('timed_out') is not False
        or exited.get('interrupted') is not None or exited.get('container_stopped_verified') is not True
        or exited.get('client_exited_while_container_running') is not False or exited.get('gpu_pids_after')!=[]):
        raise ValueError('smoke container did not exit cleanly')
    command=started.get('command',[]);cwd=Path(started.get('cwd','/'))
    scripts=[(cwd/x).resolve() for x in command if Path(x).name=='smoke_launch.py']
    if scripts!=[(HERE/'smoke_launch.py').resolve()] or command.count('--run-id')!=1:
        raise ValueError('supervisor did not invoke E4 smoke launcher')
    idx=command.index('--run-id')
    if idx+1>=len(command) or run.name!='smoke_'+command[idx+1]:raise ValueError('smoke directory does not match launcher')
    dock=launch.get('command',[])
    if dock.count(str(HERE/'smoke_worker.py'))!=1 or dock.count('--out')!=1 or Path(dock[dock.index('--out')+1]).resolve()!=run:
        raise ValueError('container did not invoke this smoke worker')
    if launch.get('worker_sha256')!=sha(HERE/'smoke_worker.py') or launch.get('launcher_sha256')!=sha(HERE/'smoke_launch.py'):
        raise ValueError('smoke launcher/worker source changed')
    from smoke_launch import check_approval
    if launch.get('approval_bundle')!=check_approval():raise ValueError('smoke source approval bundle mismatch')
    inspection=subprocess.run(['docker','inspect',container['container_id']],capture_output=True,text=True,timeout=10,check=True)
    state=json.loads(inspection.stdout)[0]
    if state['Id']!=container['container_id'] or state['State']['Running'] or state['State']['ExitCode']!=0:
        raise ValueError('independent Docker inspection failed')


FIELDS = {
    'task':'task', 'qid':'qid', 'group_id':'group_id', 'message_sha256':'message_sha256',
    'processor':'processor', 'main_visual':'visual_sha256', 'pruned_visual':'pruned_sha256',
    'full_deepstack':'deepstack_full_sha256', 'selected_deepstack':'selected_deepstack',
    'ordered_ids':'ordered_ids', 'spatial_ids':'spatial_ids', 'retained_ids':'retained_ids',
    'full_position_ids':'original_position_ids', 'full_delta':'full_rope_delta',
    'prefill_position_ids':'prefill_position_ids', 'adjusted_delta':'rope_delta',
    'prefill_embeds':'prefill_embed_sha256', 'prefill_mask':'attention_mask_sha256',
    'injected_mask':'injected_mask', 'injected_deepstack':'injected_deepstack',
    'generation_scalars':'generation_scalars', 'output_ids':'output_ids', 'answer':'answer',
    'eos_ids':'eos_ids', 'eos':'eos', 'truncated':'truncated'}


def compare_record(e4, e1, input_row):
    """Compare independently loaded artifacts; this alone is NOT acceptance."""
    if e4.get('budget') != 256 or e4.get('selection_mode') != 'author_greedy' or e4.get('greedy_calls') != 1:
        raise ValueError('K256 real greedy evidence required')
    if e4.get('actual_injection_verified') is not True or e1.get('forced') is not False:
        raise ValueError('natural verified injection required')
    if e1.get('arm') != ('P3-D1' if e4.get('ds_on') is True else 'P3-D0') or type(e4.get('ds_on')) is not bool:
        raise ValueError('wrong E1 arm')
    for left, right in FIELDS.items():
        if left not in e4 or right not in e1 or e4[left] != e1[right]:
            raise ValueError('record mismatch: '+left)
    for key in ('task','qid','group_id','message_sha256'):
        if key not in input_row or str(e4[key]) != str(input_row[key]):
            raise ValueError('frozen input mismatch: '+key)
    if not input_row.get('image_sha256') or e4.get('image_sha256') != input_row['image_sha256']:
        raise ValueError('frozen image mismatch')
    if e4.get('max_new_tokens') != 2048 or e4['generation_scalars'].get('do_sample') is not False or e4['generation_scalars'].get('max_new_tokens') != 2048:
        raise ValueError('unsupported generation protocol')
    return True


def validate_budget_header(r,reference,tokenizer):
    """Validate recorded selection, P3, injection metadata and token decoding."""
    import numpy as np
    k=r['budget'];on=r['ds_on'];start=r['image_start'];length=r['original_length']
    if k not in (128,256,512,1024) or type(on) is not bool or not isinstance(start,int) or not isinstance(length,int) or start<0 or start+1024>length:
        raise ValueError('invalid budget/visual segment')
    order=r['ordered_ids'];spatial=r['spatial_ids']
    if (len(order)!=k or any(type(i) is not int or i<0 or i>=1024 for i in order) or len(set(order))!=k or
        spatial!=sorted(order) or r['greedy_calls']!=(0 if k==1024 else 1) or
        r['selection_mode']!=('keep_all' if k==1024 else 'author_greedy')):
        raise ValueError('invalid budget selection')
    if k==1024 and order!=list(range(1024)):raise ValueError('invalid keep-all order')
    retained=list(range(start))+[start+i for i in spatial]+list(range(start+1024,length))
    full=r['full_position_ids']
    if r['retained_ids']!=retained or len(full)!=3 or any(len(axis)!=1 or len(axis[0])!=length for axis in full):
        raise ValueError('invalid retained or original position shape')
    expected=[[[axis[0][i] for i in retained]] for axis in full]
    if r['prefill_position_ids']!=expected or r['adjusted_delta']!=[[r['full_delta'][0][0]+1024-k]]:
        raise ValueError('invalid P3 gather or decode delta')
    if (full!=reference['original_position_ids'] or r['full_delta']!=reference['full_rope_delta'] or
        start!=reference['image_start'] or length!=reference['original_length']):
        raise ValueError('original position geometry differs from accepted E1')
    mask=[[False]*start+[True]*k+[False]*(length-start-1024)]
    if r['injected_mask']!=(mask if on else None):raise ValueError('invalid actual DS mask')
    if r['injected_deepstack']!=(r['selected_deepstack'] if on else None):raise ValueError('invalid actual DS injection')
    forced=r.get('forced');expected_scalars=dict(reference['generation_scalars'])
    if forced is True:expected_scalars['max_new_tokens']=33
    if type(forced) is not bool:raise ValueError('missing generation mode')
    if (r['processor']!=reference['processor'] or r['main_visual']!=reference['visual_sha256'] or
        r['full_deepstack']!=reference['deepstack_full_sha256'] or r['generation_scalars']!=expected_scalars or
        r['eos_ids']!=([] if forced else reference['eos_ids']) or r['max_new_tokens']!=(33 if forced else 2048)):
        raise ValueError('processor/features/generation mismatch E1')
    expected_mask=np.ones((1,len(retained)),dtype=np.int64)
    if r['prefill_mask']!={'dtype':'torch.int64','shape':[1,len(retained)],'sha256':hashlib.sha256(expected_mask.tobytes()).hexdigest()}:
        raise ValueError('invalid prefill attention mask')
    if (r['visual_calls']!=1 or r['main_merger_calls']!=1 or r['ds_calls']!=[1,1,1] or
        not r['block_calls'] or any(n!=1 for n in r['block_calls']) or r['actual_injection_verified'] is not True):
        raise ValueError('visual execution counts invalid')
    if tokenizer.decode(r['output_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)!=r['answer']:
        raise ValueError('independent output decoding mismatch')
    validate_trace(r['decode_trace'],r['output_ids'],len(retained),r['adjusted_delta'][0][0])
    if r['decode_trace'][0]['position_ids']!=r['prefill_position_ids']:raise ValueError('actual prefill position trace mismatch')
    if forced and (type(r.get('forced_token')) is not int or r['output_ids']!=[r['forced_token']]*33):
        raise ValueError('forced 32-step token prefix mismatch')
    return k,on,start,retained,spatial


def validate_budget_record(r,reference,arrays,tokenizer):
    """Reconstruct selected features and prefill from full raw smoke arrays."""
    import numpy as np
    k,on,start,retained,spatial=validate_budget_header(r,reference,tokenizer)
    expected_names={'visual','prefill','ds0','ds1','ds2'}|({'injected0','injected1','injected2'} if on else set())
    if r['forced']:expected_names |= {'processor_input_ids','processor_attention_mask','processor_pixel_values','processor_image_grid_thw'}
    if r.get('prefill_logits') is not None:expected_names.add('prefill_logits')
    if r['forced'] and r.get('prefill_logits') is None:raise ValueError('forced prefill logits missing')
    if set(arrays)!=expected_names:raise ValueError('invalid archived tensor inventory')
    metas=[r['main_visual']]+r['full_deepstack'];selected=[r['pruned_visual']]+r['selected_deepstack']
    for name,meta,sel in zip(('visual','ds0','ds1','ds2'),metas,selected):
        tensor=arrays[name]
        if (tensor.dtype!=np.uint8 or meta['dtype']!='torch.bfloat16' or len(meta['shape'])!=2 or meta['shape'][0]!=1024
            or list(tensor.shape)!=[1024,2*meta['shape'][1]] or hashlib.sha256(tensor.tobytes()).hexdigest()!=meta['sha256']):
            raise ValueError('archived full feature mismatch: '+name)
        if (sel['dtype']!=meta['dtype'] or sel['shape']!=[k,meta['shape'][1]] or
            hashlib.sha256(tensor[spatial].tobytes()).hexdigest()!=sel['sha256']):
            raise ValueError('independent selected feature mismatch: '+name)
    prefill=arrays['prefill'];width=2*metas[0]['shape'][1]
    if (prefill.dtype!=np.uint8 or list(prefill.shape)!=[1,len(retained),width] or
        r['prefill_embeds']!={'dtype':'torch.bfloat16','shape':[1,len(retained),metas[0]['shape'][1]],
                              'sha256':hashlib.sha256(prefill.tobytes()).hexdigest()} or
        not np.array_equal(prefill[0,start:start+k],arrays['visual'][spatial])):
        raise ValueError('actual prefill visual features mismatch')
    if on:
        for i in range(3):
            injected=arrays['injected'+str(i)]
            if injected.dtype!=np.uint8 or not np.array_equal(injected,arrays['ds'+str(i)][spatial]):
                raise ValueError('actual injected DS features mismatch')
    if r['forced'] or r.get('prefill_logits') is not None:
        items=({'processor_'+key:r['processor'][key] for key in ('input_ids','attention_mask','pixel_values','image_grid_thw')}
               if r['forced'] else {})
        if r.get('prefill_logits') is not None:items['prefill_logits']=r['prefill_logits']
        sizes={'torch.bfloat16':2,'torch.float16':2,'torch.float32':4,'torch.float64':8,'torch.int64':8,'torch.int32':4,'torch.uint8':1,'torch.bool':1}
        for name,meta in items.items():
            tensor=arrays[name];itemsize=sizes.get(meta.get('dtype'))
            count=1
            for n in meta.get('shape',[]):count*=n
            if (itemsize is None or tensor.dtype!=np.uint8 or tensor.size!=count*itemsize or
                hashlib.sha256(tensor.tobytes()).hexdigest()!=meta.get('sha256')):
                raise ValueError('diagnostic original tensor mismatch: '+name)
            if name=='prefill_logits':
                raw=tensor.tobytes();dtype=meta['dtype']
                if dtype=='torch.bfloat16':
                    values=(np.frombuffer(raw,dtype='<u2').astype(np.uint32)<<16).view(np.float32)
                elif dtype in ('torch.float16','torch.float32','torch.float64'):
                    values=np.frombuffer(raw,dtype={'torch.float16':'<f2','torch.float32':'<f4','torch.float64':'<f8'}[dtype])
                else:raise ValueError('prefill logits must be floating point')
                if values.size==0 or not np.isfinite(values).all():raise ValueError('nonfinite prefill logits')
    return hashlib.sha256(prefill[0,:start].tobytes()+prefill[0,start+k:].tobytes()).hexdigest()


def validate_budget_summary(r,reference,tokenizer):
    """Check recorded summaries; raw tensor selection cannot be reconstructed."""
    k,on,start,retained,_=validate_budget_header(r,reference,tokenizer)
    evidence=r['arrays']
    if set(evidence)!={'kind','tensors'} or evidence['kind']!='summary_only':raise ValueError('summary-only tensor evidence required')
    tensors=evidence['tensors'];names={'visual','prefill','ds0','ds1','ds2'}|({'injected0','injected1','injected2'} if on else set())
    if set(tensors)!=names or tensors['visual']!=r['main_visual'] or tensors['prefill']!=r['prefill_embeds']:
        raise ValueError('summary tensor inventory mismatch')
    for i in range(3):
        if tensors['ds'+str(i)]!=r['full_deepstack'][i]:raise ValueError('summary full DS mismatch')
        if on and tensors['injected'+str(i)]!=r['injected_deepstack'][i]:raise ValueError('summary actual DS mismatch')
    metas=[r['main_visual']]+r['full_deepstack'];selected=[r['pruned_visual']]+r['selected_deepstack']
    for meta,sel in zip(metas,selected):
        if (meta.get('dtype')!='torch.bfloat16' or len(meta.get('shape',[]))!=2 or meta['shape'][0]!=1024 or
            sel.get('dtype')!=meta['dtype'] or sel.get('shape')!=[k,meta['shape'][1]]):raise ValueError('summary feature shape mismatch')
        if k==1024 and sel!=meta:raise ValueError('keep-all summary mismatch')
    d=metas[0]['shape'][1];full=r['prefill_embeds'];before=r['prefill_text_before'];after=r['prefill_text_after']
    if (full.get('dtype')!='torch.bfloat16' or full.get('shape')!=[1,len(retained),d] or
        before.get('dtype')!=full['dtype'] or before.get('shape')!=[1,start,d] or
        after.get('dtype')!=full['dtype'] or after.get('shape')!=[1,len(retained)-start-k,d]):
        raise ValueError('summary prefill/text shape mismatch')
    for meta in (full,before,after,*metas,*selected):
        digest=meta.get('sha256')
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('invalid tensor hash')
    return before,after


def main():
    p=argparse.ArgumentParser();p.add_argument('--e1-acceptance',type=Path,required=True)
    p.add_argument('--e1-acceptance-started',type=Path,required=True);p.add_argument('--e1-acceptance-terminal',type=Path,required=True)
    p.add_argument('--smoke',type=Path,required=True);p.add_argument('--smoke-started',type=Path,required=True)
    p.add_argument('--smoke-terminal',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();proof=check_e1(a.e1_acceptance,a.e1_acceptance_started,a.e1_acceptance_terminal)
    from score import recheck_acceptance
    recheck_acceptance(a.e1_acceptance)
    run=a.smoke.resolve();started=json.loads(a.smoke_started.read_text());terminal=json.loads(a.smoke_terminal.read_text())
    launch=json.loads((run/'launch.json').read_text());exited=json.loads((run/'exit.json').read_text())
    validate_smoke_terminal(run,started,terminal,launch,exited)
    frozen=json.loads((run/'frozen.json').read_text());done=json.loads((run/'worker_complete.json').read_text())
    if (frozen.get('stage')!='smoke' or frozen.get('n_questions')!=32 or frozen.get('e1_acceptance_sha256')!=sha(a.e1_acceptance)
        or frozen.get('plan_sha256')!=sha(PLAN) or frozen.get('gt_read') is not False or frozen.get('scoring') is not False
        or frozen.get('budgets')!=[128,256,512,1024] or frozen.get('ds_on')!=[False,True]
        or frozen.get('forced_text')!=' the' or frozen.get('forced_decode_steps')!=32
        or done.get('n_questions')!=32 or done.get('n_natural_generations')!=256 or done.get('n_forced_generations')!=8):
        raise ValueError('smoke frozen protocol mismatch')
    for path,digest in frozen.get('code_sha256',{}).items():
        if sha(path)!=digest:raise ValueError('smoke source changed')
    for path in ('smoke_worker.py','smoke_contract.py','budget_runner.py','budget_adapter.py','decode_observer.py','off_adapter.py','skip_deepstack.py','identity_records.py','identity_worker.py'):
        if frozen.get('code_sha256',{}).get(str(HERE/path))!=sha(HERE/path):raise ValueError('missing smoke source binding')
    panel=select_smoke(proof);keys=('task','qid','group_id','message_sha256','image_sha256')
    if frozen.get('selected')!=[{k:r[k] for k in keys} for r in panel]:raise ValueError('smoke cohort mismatch')
    if frozen.get('forced_anchor')!={k:panel[0][k] for k in ('task','qid','group_id')}:raise ValueError('forced anchor mismatch')
    records=read_jsonl(run/'records.jsonl')
    forced_records=read_jsonl(run/'forced_records.jsonl')
    generated=read_jsonl(run/'generated.jsonl')
    expected=[(r['task'],str(r['qid']),k,on) for r in panel for k in (128,256,512,1024) for on in (False,True)]
    actual=[(r.get('task'),str(r.get('qid')),r.get('budget'),r.get('ds_on')) for r in records]
    if actual!=expected:raise ValueError('smoke arm inventory/order mismatch')
    forced_expected=[(panel[0]['task'],str(panel[0]['qid']),k,on) for k in (128,256,512,1024) for on in (False,True)]
    if [(r.get('task'),str(r.get('qid')),r.get('budget'),r.get('ds_on')) for r in forced_records]!=forced_expected:
        raise ValueError('forced arm inventory mismatch')
    if any(r.get('forced') is not False for r in records) or any(r.get('forced') is not True for r in forced_records):
        raise ValueError('natural/forced inventory mismatch')
    generated_fields=('task','qid','group_id','budget','ds_on','output_ids','eos_ids','generation_scalars','forced')
    all_records=records+forced_records
    if len(generated)!=len(all_records) or any(g!={k:r[k] for k in generated_fields} for g,r in zip(generated,all_records)):
        raise ValueError('generated and observation journals differ')
    e1=read_jsonl(proof['files']['journal']['path'])
    e1map={(r['task'],str(r['qid']),r['arm']):r for r in e1 if r['forced'] is False}
    if len(e1map)!=1536:raise ValueError('E1 record inventory mismatch')
    bykey={(r['task'],str(r['qid'])):r for r in panel};compared=[];text_hashes={}
    from transformers import AutoTokenizer
    import numpy as np
    plan=json.loads(PLAN.read_text())
    tokenizer=AutoTokenizer.from_pretrained(plan['recipe']['model_path'],local_files_only=True,trust_remote_code=False)
    forced_ids=tokenizer.encode(' the',add_special_tokens=False)
    if not forced_ids:raise ValueError('forced prefix tokenizer result empty')
    anchor_records={(r['budget'],r['ds_on']):r for r in records if r['task']==panel[0]['task'] and str(r['qid'])==str(panel[0]['qid'])}
    if len(anchor_records)!=8 or any(r.get('prefill_logits') is None for r in anchor_records.values()):
        raise ValueError('natural anchor logits incomplete')
    common=('processor','image_start','original_length','ordered_ids','spatial_ids','retained_ids','main_visual','pruned_visual',
            'full_deepstack','selected_deepstack','injected_mask','injected_deepstack','full_position_ids','full_delta',
            'prefill_position_ids','adjusted_delta','prefill_embeds','prefill_mask','prefill_text_before','prefill_text_after','prefill_logits')
    for r in forced_records:
        natural=anchor_records[(r['budget'],r['ds_on'])]
        if any(r.get(k)!=natural.get(k) for k in common):raise ValueError('forced anchor differs from natural prefill')
    for on in (False,True):
        natural=anchor_records[(256,on)];e1anchor=e1map[(natural['task'],str(natural['qid']),'P3-D1' if on else 'P3-D0')]
        if not e1anchor.get('logits') or natural['prefill_logits']!=e1anchor['logits'][0]:raise ValueError('K256 anchor logits differ from E1')
    for r in all_records:
        item=bykey[(r['task'],str(r['qid']))]
        if any(r.get(k)!=item[k] for k in keys):raise ValueError('smoke record input mismatch')
        ids=r.get('output_ids');eos=r.get('eos_ids')
        if r['forced']:
            if r.get('forced_token')!=forced_ids[0] or ids!=[forced_ids[0]]*33 or eos!=[] or r.get('eos') is not False or r.get('truncated') is not False:
                raise ValueError('invalid fixed-prefix generation')
        elif (not isinstance(ids,list) or not ids or len(ids)>2048 or not isinstance(eos,list)
            or r.get('eos')!=bool(ids[-1] in eos) or r.get('truncated')!=bool(len(ids)==2048 and ids[-1] not in eos)
            or not (r['eos'] or r['truncated'])):raise ValueError('invalid smoke generation endpoint')
        archive=Path(r['arrays']['path']).resolve()
        name=hashlib.sha256(json.dumps([r['task'],str(r['qid']),r['budget'],r['ds_on'],r['forced']],separators=(',',':')).encode()).hexdigest()+'.npz'
        if r['arrays'].get('kind')!='full_arrays' or archive.parent!=(run/'arrays').resolve() or archive.name!=name or sha(archive)!=r['arrays']['sha256']:
            raise ValueError('smoke array archive identity mismatch')
        with np.load(archive,allow_pickle=False) as arrays:
            text_hash=validate_budget_record(r,e1map[(r['task'],str(r['qid']),'P3-D0')],arrays,tokenizer)
        question=(r['task'],str(r['qid']))
        if question in text_hashes and text_hashes[question]!=text_hash:raise ValueError('prefill text differs across arms')
        text_hashes[question]=text_hash
        if r['budget']==256 and not r['forced']:
            arm='P3-D1' if r['ds_on'] else 'P3-D0'
            compare_record(r,e1map[(r['task'],str(r['qid']),arm)],item)
            compared.append({'task':r['task'],'qid':str(r['qid']),'group_id':r['group_id'],'arm':arm})
    if len(compared)!=64:raise ValueError('K256 comparison inventory mismatch')
    files=[a.e1_acceptance,a.e1_acceptance_started,a.e1_acceptance_terminal,a.smoke_started,a.smoke_terminal,
           run/'launch.json',run/'exit.json',run/'frozen.json',run/'worker_complete.json',run/'records.jsonl',run/'forced_records.jsonl',run/'generated.jsonl']
    result={'status':'passed','scope':'32 smoke questions only; no global 384-question reuse approval',
            'compared':compared,'all_arms_validated':256,'forced_arms_validated':8,'forced_decode_steps':32,
            'files_sha256':{str(p.resolve()):sha(p) for p in files},
            'array_sha256':{str(Path(r['arrays']['path']).resolve()):r['arrays']['sha256'] for r in all_records},'verifier_sha256':sha(__file__),
            'GT_opened':False,'scoring_performed':False}
    with a.out.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')

if __name__ == '__main__': main()
