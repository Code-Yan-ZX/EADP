"""Metadata, guards and pure observation contracts; no tensor/model imports."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
POPE = HERE.parent
PANEL = POPE / 'alpha1_beta2_native_panel'
PLAN = HERE / 'plan.json'
OUT = HERE / 'execution'
PYTHON = '/home/dell/miniconda3/envs/llava_pruner/bin/python'
RUN_ID = 'POPE_full8910_alpha1_beta2_K32_single_beta_control'
PARAMETERS = dict(visual_token_num=32, alpha=1., beta=2., entropy_keep_ratio=.2,
    temperature=0., top_p=None, num_beams=1, max_new_tokens=128,
    conv_mode='vicuna_v1', use_cache=True, source_caller_stream_wait=True)
PREDECESSOR = {
    'plan.json':'404ee2ad79241f91b8c392e69b3958fe096727e84fe9a1eaa6a2cd1a9d3c6605',
    'root_terminal_process_proof.json':'2c5c74592cdd58e44c754f994936033ce6315d56f10cb5c4c8f3766eb9d8a1f9',
    'peer_terminal_review.json':'0a7356219219b4d126cfc92f0e79cc4c1902a5b6f5b8658d4a643415253a487b',
    'execution/independent_verification.json':'2342e6b9cae6cb19f2056ace0f24fe618ea1254f93413a78f5f51c93c114069f',
    'execution/release.json':'849b8422c5ae842c036ecda58d2457f26de6a05fb2928de94dce668d8b38adc7'}
UTC = lambda: datetime.now(timezone.utc).isoformat()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
text_sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def save(path, value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def stat(path):
    path=Path(path);s=path.stat()
    return dict(resolved=str(path.resolve()),bytes=s.st_size,device=s.st_dev,inode=s.st_ino,
        mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def describe(path):
    return dict(path=str(path),sha256=sha(path),bytes=Path(path).stat().st_size)


def int64_sha(values):
    return hashlib.sha256(struct.pack('<'+'q'*len(values),*values)).hexdigest()


def guidance(prompt):
    return prompt.replace('\nAnswer the question using a single word or phrase.','')


def check_observation(row, question, position):
    assert row['position']==position and row['question_id']==question['question_id']
    assert row['image']==question['image'] and row['full_prompt_sha256']==text_sha(question['text'])
    assert row['actual_guidance_sha256']==text_sha(guidance(question['text']))
    assert row['actual_alpha_beta_K']==[1.,2.,32]
    assert row['calls']==dict(encode=1,greedy=1)
    assert row['post_generation_tensor_reads_only'] is True
    ids=row['input_token_ids'];assert all(type(i) is int for i in ids)
    assert ids.count(-200)==1 and row['input_ids_shape']==[1,len(ids)] and row['input_ids_dtype']=='torch.int64'
    assert row['input_ids_sha256']==int64_sha(ids)
    assert row['quotas']==[32] and row['mask_shape']==[1,576]
    prefix=row['effective_ordered_prefix'];mask=row['effective_mask_indices']
    assert len(prefix)==len(set(prefix))==len(mask)==32 and mask==sorted(prefix)
    assert all(type(i) is int and 0<=i<576 for i in prefix)
    assert row['returned_retained_tokens']==32 and row['projected_feature_shape']==[1,576,4096]
    assert row['projected_feature_dtype']=='torch.float16' and row['features_finite'] and row['importance_finite']
    assert row['importance_shape']==[1,576]
    out=row['output_token_ids'];assert out and out[0]==1 and all(type(i) is int and 0<=i<32000 for i in out)
    assert 1<len(out)<=129 and row['new_tokens']==len(out)-1
    assert row['EOS_present']==(2 in out) and row['generation_horizon']==128
    assert isinstance(row['text'],str)


def check_anchor(row, anchor):
    for key in ['question_id','image','full_prompt_sha256','actual_guidance_sha256','input_ids_sha256',
                'quotas','effective_ordered_prefix','effective_mask_indices','output_token_ids','text']:
        assert row[key]==anchor[key],('accepted_beta2_panel_anchor',row['position'],key)


def check_predecessor():
    for name,digest in PREDECESSOR.items(): assert sha(PANEL/name)==digest,name
    root=read(PANEL/'root_terminal_process_proof.json');peer=read(PANEL/'peer_terminal_review.json')
    report=read(PANEL/'execution/independent_verification.json');release=read(PANEL/'execution/release.json')
    assert root['root_owned_exec_actual_exit_code']==0 and root['all_original_processes_exited']
    assert root['root_owned_exec_handle']==98439 and root['plan_sha256']==PREDECESSOR['plan.json']
    assert root['release_sha256']==PREDECESSOR['execution/release.json']
    assert root['independent_verification_sha256']==PREDECESSOR['execution/independent_verification.json']
    assert peer['success'] and peer['n_generations']==32 and peer['source_anchors_verified']==16
    assert peer['changed_counts']['actual_visual_FP16']==16 and peer['changed_counts']['outputIDs']==1
    assert report['success'] and report['n_generations']==32 and report['source_anchors_verified']==report['variant_inputs_verified']==16
    assert release['success'] and release['actual_native_exited'] and release['native_exit_code']==0


def check(plan):
    assert Path(sys.executable).resolve()==Path(PYTHON).resolve()
    assert plan['run_id']==RUN_ID and plan['parameters']==PARAMETERS and plan['n_questions']==8910
    for group in ['source_sha256','implementation_sha256','reference_sha256']:
        for path,digest in plan[group].items(): assert sha(path)==digest,path
    for group in ['model_stat_guards','image_stat_guards']:
        for path,guard in plan[group].items():
            assert stat(path)=={k:v for k,v in guard.items() if k!='prior_verified_sha256'},path
    check_predecessor()
