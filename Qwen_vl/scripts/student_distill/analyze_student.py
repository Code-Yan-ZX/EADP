"""CPU-only frozen student test analysis; never selects checkpoints or arms.

Run --self-check before freeze. Full analysis refuses incomplete or unverified
artifacts, uses every frozen test parent, and reports only preregistered paired
bootstrap contrasts. Answer-conditioned gold references are explicitly offline.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import numpy as np

HERE=Path(__file__).resolve().parent
QWEN=HERE.parents[1]
OUT=QWEN/'outputs/student_distill'
MULTI=QWEN/'scripts/multistream_oracle/analysis.py'
spec=importlib.util.spec_from_file_location('native_multistream_cpu_analysis',MULTI)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
TEACHERS=('magnitude_v','magnitude_joint','signed_v','signed_joint')
SEEDS=(11,23,37)
BASELINES=('full','rres512','b2','random')
STUDENTS=tuple(f'{t}__seed_{s}' for t in TEACHERS for s in SEEDS)
ORACLES=tuple('oracle_'+t for t in TEACHERS)
ARMS=BASELINES+STUDENTS+ORACLES
N=1024;K=256;DRAWS=10000;SEED=20261109
MANIFEST=OUT/'data/test_manifest.json'; FREEZE=OUT/'eval_freeze.json'


def stable_top(scores,k):
    a=np.asarray(scores,dtype=np.float32)
    if a.shape!=(N,) or not np.isfinite(a).all(): raise ValueError('invalid score map')
    return np.lexsort((np.arange(N),-a))[:k]


def preregistered_pairs():
    pairs=[(base,student) for student in STUDENTS for base in BASELINES]
    pairs += [(f'{family}_v__seed_{s}',f'{family}_joint__seed_{s}')
              for family in ('magnitude','signed') for s in SEEDS]
    if len(pairs)!=54 or len(set(pairs))!=54: raise AssertionError('bad preregistered contrast list')
    return pairs


def verify_hash(path,digest):
    if m.sha256(path)!=digest: raise ValueError('frozen hash mismatch: '+str(path))


def verify_chain(config,manifest_path,freeze_path):
    verify_hash(manifest_path,config['manifest_sha256'])
    verify_hash(freeze_path,config['eval_freeze_sha256'])
    f=json.loads(Path(freeze_path).read_text())
    verify_hash(manifest_path,f['test_manifest_sha256'])
    verify_hash(OUT/'data/dev_manifest.json',f['dev_manifest_sha256'])
    verify_hash(OUT/'train_freeze.json',f['train_freeze_sha256'])
    verify_hash(OUT/'student_inference_gates.json',f['inference_gates_sha256'])
    verify_hash(f['training_summary']['path'],f['training_summary']['sha256'])
    if f['arms']!=list(ARMS) or config['planned_arms']!=list(ARMS): raise ValueError('arm mismatch')
    if set(f['checkpoints'])!=set(STUDENTS): raise ValueError('checkpoint inventory mismatch')
    for info in f['checkpoints'].values(): verify_hash(info['path'],info['sha256'])
    if str(Path(__file__).resolve()) not in f['source_sha256']: raise ValueError('analysis source missing from eval freeze')
    for path,digest in f['source_sha256'].items(): verify_hash(path,digest)
    train=json.loads((OUT/'train_freeze.json').read_text())
    if str(Path(__file__).resolve()) not in train['source_sha256']: raise ValueError('analysis source missing from training freeze')
    for path,digest in train['source_sha256'].items(): verify_hash(path,digest)
    for role,digest in train['manifest_sha256'].items(): verify_hash(OUT/f'data/{role}_manifest.json',digest)
    verify_hash(OUT/'data/dev_gate_manifest.json',train['gate_adapter_sha256'])
    verify_hash(OUT/'gates.json',train['gates_sha256'])
    snap=Path(train['model']['snapshot'])
    for name,digest in train['model']['metadata_sha256'].items(): verify_hash(snap/name,digest)
    summary=json.loads(Path(f['training_summary']['path']).read_text())
    if not summary['complete'] or summary['train_freeze_sha256']!=m.sha256(OUT/'train_freeze.json'):
        raise ValueError('invalid training summary provenance')
    if summary['teachers']!=list(TEACHERS) or summary['seeds']!=list(SEEDS): raise ValueError('training plan mismatch')
    for name,digest in summary['sources_sha256'].items(): verify_hash(HERE/name,digest)
    for arm,info in f['checkpoints'].items():
        if summary['checkpoints'].get(Path(info['path']).name)!=info['sha256']: raise ValueError('checkpoint not from training')
    return dict(eval_freeze_sha256=m.sha256(freeze_path),train_freeze_sha256=m.sha256(OUT/'train_freeze.json'),
                train_summary_sha256=m.sha256(f['training_summary']['path']),
                source_files_verified=len(f['source_sha256'])+len(train['source_sha256']),
                checkpoints_verified=len(f['checkpoints']))


def keep_check(keep, expected_count, n=N):
    if not isinstance(keep,list) or len(keep)!=expected_count or keep!=sorted(set(keep)):
        raise ValueError('bad or non-raster keep indices')
    if any(not isinstance(i,int) or i<0 or i>=n for i in keep): raise ValueError('keep index out of range')


def validate_row(row,item):
    for field in ('item_id','parent_id','question','answers','image_sha256'):
        if row.get(field)!=item.get(field): raise ValueError('data mismatch: '+field)
    if set(row['predictions'])!=set(ARMS): raise ValueError('incomplete arms remain pending')
    teacher=row['teacher']
    if (teacher.get('K')!=K or teacher.get('n_visual_tokens')!=N or not teacher.get('teacher_deepstack')
        or teacher.get('append_eos') or teacher.get('checkpointing')
        or not teacher.get('finite_gradients') or not teacher.get('parameter_gradients_none')
        or not teacher.get('parameter_versions_unchanged')):
        raise ValueError('teacher native provenance invalid')
    if not np.isfinite(teacher['mean_answer_nll']) or teacher['mean_answer_nll']<0: raise ValueError('invalid teacher loss')
    if not teacher.get('gold_token_ids') or teacher['target_token_count']!=len(teacher['gold_token_ids']):
        raise ValueError('invalid target token count')
    for family in ('magnitude','signed'):
        streams=np.asarray(teacher[family+'_per_stream'],dtype=np.float32)
        if streams.shape!=(4,N) or not np.isfinite(streams).all(): raise ValueError('bad teacher streams')
        if family=='magnitude' and (streams<0).any(): raise ValueError('negative magnitude')
        for kind,expected in [('v',streams[0]),('joint',streams.sum(axis=0,dtype=np.float32))]:
            name=family+'_'+kind
            scores=np.asarray(teacher['scores'][name],dtype=np.float32)
            if scores.shape!=(N,) or not np.isfinite(scores).all() or not np.allclose(scores,expected,rtol=2e-5,atol=1e-12):
                raise ValueError('teacher stream sum mismatch')
            keep=teacher['keep_indices'][name];keep_check(keep,K)
            if keep!=sorted(stable_top(scores,K).tolist()): raise ValueError('teacher topK mismatch')
    stats={}
    for arm in ARMS:
        p=row['predictions'][arm]
        if p.get('error') or not isinstance(p.get('text'),str) or not isinstance(p.get('token_ids'),list):
            raise ValueError('prediction missing/failed, must remain pending')
        if not isinstance(p.get('truncated'),bool) or p.get('eos')!= (not p['truncated']):
            raise ValueError('invalid EOS/truncation')
        expected=N if arm=='full' else K
        if p['actual_visual_count']!=expected or p['ds_lengths']!=[expected]*3:
            raise ValueError('physical visual/DeepStack count mismatch')
        keep_check(p['keep_idx'],expected,n=K if arm=='rres512' else N)
        if p['cache_length']!=p['prefill_length']+len(p['decode_positions']): raise ValueError('cache mismatch')
        if arm in STUDENTS:
            keep=p['keep_idx'];scores=p['student_scores']
            if keep!=sorted(stable_top(scores,K).tolist()): raise ValueError('student topK mismatch')
            teacher_name=arm.rsplit('__seed_',1)[0]
            ts=teacher['scores'][teacher_name]; teacher_keep=set(teacher['keep_indices'][teacher_name]); chosen=set(keep)
            stats[arm]=dict(head_recall8=len(chosen.intersection(stable_top(ts,8).tolist()))/8,
                head_recall32=len(chosen.intersection(stable_top(ts,32).tolist()))/32,
                keep_overlap_count=len(chosen.intersection(teacher_keep)),
                keep_overlap_fraction=len(chosen.intersection(teacher_keep))/K)
        if arm in ORACLES and p['keep_idx']!=teacher['keep_indices'][arm[len('oracle_'):]]:
            raise ValueError('offline oracle keep mismatch')
    return stats


def analyze(payload,manifest,verify=None):
    if not payload.get('complete') or not manifest.get('complete') or manifest.get('role')!='test':
        raise ValueError('incomplete predictions/invalid test manifest')
    config=payload['config']
    if config['planned_arms']!=list(ARMS) or config['teachers']!=list(TEACHERS) or config['seeds']!=list(SEEDS):
        raise ValueError('plan mismatch')
    items=manifest['items'];rows=payload['records']
    if len(items)!=32 or len(rows)!=32: raise ValueError('must retain all 32 test questions')
    byid={r['item_id']:r for r in rows}
    if len(byid)!=32 or len({x['parent_id'] for x in items})!=32 or set(byid)!={x['item_id'] for x in items}:
        raise ValueError('wrong, duplicate or missing test items/parents')
    rows=[byid[i['item_id']] for i in items]
    anls,scoring=m.scorers()
    exact={a:[] for a in ARMS};ans={a:[] for a in ARMS};literal={a:[] for a in ARMS}
    diagnostics={a:[] for a in STUDENTS};detail=[]
    for row,item in zip(rows,items):
        stats=validate_row(row,item)
        for arm in STUDENTS: diagnostics[arm].append(stats[arm])
        answers=row['answers'];entry={k:row[k] for k in ('item_id','parent_id','question','answers','image_sha256')}
        entry['teacher_nll']=row['teacher']['mean_answer_nll'];entry['predictions']={}
        for arm in ARMS:
            p=row['predictions'][arm];text=p['text']
            e=int(m.score_exact(text,answers));a=anls(text,answers);l=int(text in answers)
            exact[arm].append(e);ans[arm].append(a);literal[arm].append(l)
            entry['predictions'][arm]=dict(text=text,token_ids=p['token_ids'],truncated=p['truncated'],eos=p['eos'],
                                           numeric_exact=e,anls=a,literal_exact=l)
        detail.append(entry)
    # One question per independent source parent; same draws for all contrasts.
    draws=np.random.default_rng(SEED).integers(0,len(rows),size=(DRAWS,len(rows)))
    def summary(vectors,binary=False):
        out={}
        for arm in ARMS:
            values=np.asarray(vectors[arm],dtype=float)
            out[arm]=dict(mean=float(values.mean()),n=len(rows))
            if binary: out[arm]['correct']=int(values.sum())
        return out
    pairs={}
    for before,after in preregistered_pairs():
        diff=np.asarray(exact[after])-np.asarray(exact[before])
        boot=diff[draws].mean(axis=1)
        pairs[after+'_minus_'+before]=dict(before=before,after=after,delta_accuracy=float(diff.mean()),
            bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist(),rescue=int((diff==1).sum()),
            breakage=int((diff==-1).sum()),tie=int((diff==0).sum()),
            rescue_item_ids=[rows[i]['item_id'] for i in np.flatnonzero(diff==1)],
            break_item_ids=[rows[i]['item_id'] for i in np.flatnonzero(diff==-1)],
            note='A degenerate interval does not establish equivalence.' if not diff.any() else None)
    by_teacher={}
    for t in TEACHERS:
        per_seed={}; acc=[]
        for seed in SEEDS:
            arm=f'{t}__seed_{seed}';acc.append(float(np.mean(exact[arm])))
            per_seed[str(seed)]={key:float(np.mean([d[key] for d in diagnostics[arm]]))
                                for key in ('head_recall8','head_recall32','keep_overlap_count','keep_overlap_fraction')}
            per_seed[str(seed)]['numeric_accuracy']=acc[-1]
        by_teacher[t]=dict(per_seed=per_seed,accuracy_seed_mean=float(np.mean(acc)),
            accuracy_seed_range=[float(min(acc)),float(max(acc))],
            proxy_seed_mean={key:float(np.mean([per_seed[str(s)][key] for s in SEEDS]))
                             for key in ('head_recall8','head_recall32','keep_overlap_count','keep_overlap_fraction')})
    return dict(schema_version=1,complete=True,n=len(rows),arms=list(ARMS),
        primary_numeric_exact=summary(exact,True),secondary_official_anls=summary(ans),
        secondary_literal_exact=summary(literal,True),paired_numeric_exact=pairs,
        student_teacher_diagnostics=by_teacher,records=detail,scoring=scoring,
        bootstrap=dict(draws=DRAWS,seed=SEED,unit='independent source parent',
            shared_draws=True,method='paired percentile; 54 preregistered comparisons only'),
        verification=verify,limitations='32 TRAIN questions; comparisons are exploratory and multiple. Gold-gradient oracle arms are answer-aware offline references, not deployable methods or strict upper bounds. All selected questions and truncations are included.')


def self_check():
    import copy
    assert m.score_exact(' $ 1,200.00 ',['1200']) and not m.score_exact('12',['12%'])
    assert m.scorers()[0]('1997',['1996'])==.75
    ramp=np.arange(N,dtype=np.float32)+1
    mag=np.stack((ramp,ramp[::-1],np.ones(N),np.ones(N))).astype(np.float32)
    signed=np.stack((-ramp,ramp[::-1],np.zeros(N),np.ones(N))).astype(np.float32)
    scores={t:(mag if t.startswith('magnitude') else signed) for t in TEACHERS}
    maps={t:(a[0] if t.endswith('_v') else a.sum(axis=0,dtype=np.float32)).tolist() for t,a in scores.items()}
    keeps={t:sorted(stable_top(maps[t],K).tolist()) for t in TEACHERS}
    teacher=dict(K=K,n_visual_tokens=N,teacher_deepstack=True,append_eos=False,checkpointing=False,
        finite_gradients=True,parameter_gradients_none=True,parameter_versions_unchanged=True,
        mean_answer_nll=.5,gold_token_ids=[1],target_token_count=1,magnitude_per_stream=mag.tolist(),
        signed_per_stream=signed.tolist(),scores=maps,keep_indices=keeps)
    items=[];rows=[]
    for i in range(32):
        item=dict(item_id=f'synthetic_{i}',parent_id=f'parent_{i}',question='value?',answers=['12'],image_sha256='synthetic')
        predictions={}
        for arm in ARMS:
            keep=list(range(N if arm=='full' else K))
            if arm in ORACLES: keep=keeps[arm[len('oracle_'):]]
            scores_student=None
            if arm in STUDENTS:
                scores_student=ramp.tolist();keep=sorted(stable_top(scores_student,K).tolist())
            n=len(keep); text='12'
            if i==0 and arm=='b2': text='13'
            if i==1 and arm=='magnitude_joint__seed_11': text='13'
            pred=dict(text=text,token_ids=[1],truncated=False,eos=True,actual_visual_count=n,ds_lengths=[n]*3,
                      keep_idx=keep,prefill_length=5,cache_length=5,decode_positions=[])
            if scores_student is not None: pred['student_scores']=scores_student
            predictions[arm]=pred
        items.append(item);rows.append(dict(**item,teacher=teacher,predictions=predictions))
    manifest=dict(complete=True,role='test',items=items)
    payload=dict(complete=True,config=dict(planned_arms=list(ARMS),teachers=list(TEACHERS),seeds=list(SEEDS)),records=rows)
    result=analyze(payload,manifest)
    assert result['n']==32 and len(result['paired_numeric_exact'])==54
    pair=result['paired_numeric_exact']['magnitude_joint__seed_11_minus_magnitude_v__seed_11']
    assert pair['rescue']==0 and pair['breakage']==1 and pair['delta_accuracy']==-1/32
    pair=result['paired_numeric_exact']['magnitude_v__seed_11_minus_b2']
    assert pair['rescue']==1 and pair['breakage']==0 and pair['delta_accuracy']==1/32
    for label,fn in [('missing',lambda x:x['records'].pop()),
                     ('bad_keep',lambda x:x['records'][0]['predictions']['magnitude_v__seed_11']['keep_idx'].pop()),
                     ('bad_teacher',lambda x:x['records'][0]['teacher']['scores']['magnitude_joint'].__setitem__(0,-1)),
                     ('failed',lambda x:x['records'][0]['predictions']['b2'].__setitem__('error','failed'))]:
        bad=copy.deepcopy(payload);fn(bad)
        try: analyze(bad,manifest)
        except (ValueError,KeyError): pass
        else: raise AssertionError('failed to reject '+label)
    return dict(complete=True,synthetic_only=True,GPU_used=False,n=32,arms=len(ARMS),paired=len(preregistered_pairs()),
                checks=['official ANLS and numeric exact','stable topK','native and oracle counts',
                        'teacher stream sums','shared parent paired bootstrap and rescue/break',
                        'reject incomplete, bad keep, corrupt teacher, operational failure'],
                analysis_script_sha256=m.sha256(__file__))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-check',action='store_true')
    p.add_argument('--input',type=Path,default=OUT/'test_predictions.json')
    p.add_argument('--manifest',type=Path,default=MANIFEST)
    p.add_argument('--freeze',type=Path,default=FREEZE)
    p.add_argument('--out',type=Path,default=OUT/'test_analysis.json')
    a=p.parse_args()
    if a.self_check:
        print(json.dumps(self_check(),indent=2));return
    payload=json.loads(a.input.read_text()); manifest=json.loads(a.manifest.read_text())
    provenance=verify_chain(payload['config'],a.manifest,a.freeze)
    result=analyze(payload,manifest,provenance)
    result.update(test_predictions_sha256=m.sha256(a.input),test_manifest_sha256=m.sha256(a.manifest),
                  analysis_script_sha256=m.sha256(__file__))
    tmp=a.out.with_suffix(a.out.suffix+'.tmp');tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');tmp.replace(a.out)
    print(json.dumps(dict(complete=True,n=result['n'],paired=len(result['paired_numeric_exact']),output=str(a.out)),indent=2))

if __name__=='__main__': main()
