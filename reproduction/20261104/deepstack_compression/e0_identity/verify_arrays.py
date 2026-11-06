"""Independent read-only E0 gate: fail closed on incomplete or inconsistent evidence."""
import hashlib, json, os, subprocess
import numpy as np
from pathlib import Path
HERE=Path(__file__).resolve().parent
RUN=HERE/'array16'
PANEL=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261028/qwen/full1024_two_path_panel/plan.json')
SOURCE_PLAN=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json')
EXCLUSION=HERE.parent/'data_manifest/anchor_exclusion'
MODES=('native','on','k256_on','off','zero')
errors=[]
evidence={}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p): return json.loads(p.read_text())
def check(ok, why):
    if not ok: errors.append(why)
def one(i, item, condition):
    tag=f'{i:02d}_{condition}'
    paths={m:RUN/f'{i:02d}_{m}_{condition}.json' for m in MODES}
    comparison_path=RUN/f'{i:02d}_comparison_{condition}.json'
    for path in list(paths.values())+[comparison_path]: evidence[str(path)]=sha(path)
    d={m:read(path) for m,path in paths.items()}
    comp=read(comparison_path)
    for m,r in d.items():
        check(r['processor'] == item['processor_expected'], f'{tag} {m} processor')
        check(len(r['position_values'])==len(r['cache_values'])==len(r['cache_lengths_before'])==len(r['cache_lengths_after'])==len(r['llm'])==len(r['rope_delta_values']),f'{tag} {m} forward length')
        n=len(r['llm']); length=r['llm'][0]['inputs_embeds']['shape'][1]
        check(n>0,f'{tag} {m} missing forwards')
        if condition=='forced':
            check(n==33 and len(r['logits'])==33 and len(r['output_ids'][0])==33,f'{tag} {m} forced count')
            check(r['output_ids'][0]==[comp['forced_token']]*33,f'{tag} {m} forced prefix')
        else:
            check(len(r['logits'])==1,f'{tag} {m} first logits')
            check(isinstance(r['response'],str) and len(r['output_ids'][0]) > 0,f'{tag} {m} output missing')
            check(r['eos_present'] or r['at_max_tokens'],f'{tag} {m} output end unknown')
        check(len(r['input_ids_values'])==1 and len(r['input_ids_values'][0])==len(r['input_attention_values'][0]),f'{tag} {m} original text ids')
        for step,(before,after) in enumerate(zip(r['cache_lengths_before'],r['cache_lengths_after'])):
            check(len(before)==len(after)==36,f'{tag} {m} layer count {step}')
            expected_before=0 if step==0 else length+step-1
            check(all(x==expected_before for x in before) and all(x==length+step for x in after),f'{tag} {m} layer kv {step}')
        check(r['cache_values'][0]==list(range(length)),f'{tag} {m} prefill cache')
        for step in range(1,n):
            check(r['cache_values'][step]==[length+step-1],f'{tag} {m} cache pos {step}')
        if m != 'native':
            delta=r['adjusted_delta_values'][0][0]
            check(all(x==r['adjusted_delta_values'] for x in r['rope_delta_values']),f'{tag} {m} deltas drift')
            for step in range(1,n):
                p=r['position_values'][step]
                check(p==[[[length+step-1+delta]]]*3,f'{tag} {m} decode position {step}')
    verify_arrays(i, condition, d)
    a,b=d['native'],d['on']
    for field in ('inputs_embeds','attention_mask','position_ids','cache_position','visual_pos_masks','deepstack_visual_embeds'):
        check(a['llm'][0][field]==b['llm'][0][field],f'{tag} native/on prefill {field}')
    check(a['visual']==b['visual'],f'{tag} native/on visual')
    check(a['output_ids']==b['output_ids'],f'{tag} native/on output ids')
    for lhs,rhs in ((a,b),(d['off'],d['zero'])):
        prefix=0
        for x,y in zip(lhs['output_ids'][0],rhs['output_ids'][0]):
            if x!=y:break
            prefix+=1
        n=min(len(lhs['llm']),len(rhs['llm']),prefix+1)
        check(lhs['output_ids']==rhs['output_ids'],f'{tag} identity output')
        check(len(lhs['llm'])==len(rhs['llm']),f'{tag} identity calls mismatch')
        if condition=='forced': check(n==33,f'{tag} identity forced length')
        check(len(lhs['logits'])==len(rhs['logits']) and lhs['logits']==rhs['logits'],f'{tag} identity logits')
        for k in ('position_values','cache_values','rope_delta_values'):
            check(lhs[k][:n]==rhs[k][:n],f'{tag} identity {k} shared prefix')
    off,zero,k=d['off'],d['zero'],d['k256_on']
    for m in ('off','zero','k256_on'):
        r=d[m];start=r['visual_start'];ids=r['input_ids_values'][0]
        spatial=r['spatial_values']; ordered=r['ordered_values']; retained=r['retained_values']
        check(len(spatial)==len(ordered)==256 and sorted(ordered)==spatial and len(set(spatial))==256 and all(0<=x<1024 for x in spatial),f'{tag} {m} selection ids')
        expected=list(range(start))+[start+x for x in spatial]+list(range(start+1024,len(ids)))
        check(retained==expected,f'{tag} {m} retain original text')
        p3=a['position_values'][0]
        expected_pos=[[[axis[0][j] for j in retained]] for axis in p3]
        check(r['position_values'][0]==expected_pos,f'{tag} {m} gathered 3-axis position')
        check(r['adjusted_delta_values'][0][0]==r['full_delta_values'][0][0]+768,f'{tag} {m} delta')
    check(k['ordered_values']==off['ordered_values']==zero['ordered_values'],f'{tag} same actual K256 selection')
    check(k['llm'][0]['deepstack_visual_embeds']==k['selected'],f'{tag} K256 selected DS')
    check(k['llm'][0]['visual_pos_masks']['shape']==k['llm'][0]['attention_mask']['shape'],f'{tag} K256 mask geometry')
    check(comp['k256_visual_mask_correct'] and comp['k256_ds_selected_equal'],f'{tag} K256 mask or DS')
    check(off['llm'][0]['visual_pos_masks'] is None and off['llm'][0]['deepstack_visual_embeds'] is None and off['ds_slot_all_zero'] is None,f'{tag} off parameters')
    check(zero['ds_slot_all_zero']==[True]*3 and len(zero['llm'][0]['deepstack_visual_embeds'])==3,f'{tag} zero slots')
    check(k['llm'][0]['inputs_embeds']==off['llm'][0]['inputs_embeds']==zero['llm'][0]['inputs_embeds'],f'{tag} K256 main embeddings')
    check(all(all(v) for v in comp['decode_matches_adjusted_delta'].values()),f'{tag} per-step adjusted delta')


def verify_arrays(i, condition, d):
    tag=f'{i:02d}_{condition}'
    archives={}
    try:
        for mode,r in d.items():
            path=Path(r['array_path'])
            digest=sha(path); evidence[str(path)]=digest
            check(path.parent==RUN and path.name==f'{i:02d}_{mode}_{condition}.npz' and digest==r['array_sha256'],f'{tag} {mode} array archive')
            z=np.load(path,allow_pickle=False);archives[mode]=z
            check(sorted(z.files)==r['array_keys'],f'{tag} {mode} keys')
            def matches(key,meta):
                check(key in z.files and list(z[key].shape)==meta['shape'] and hashlib.sha256(z[key].tobytes()).hexdigest()==meta['sha256'],f'{tag} {mode} tensor {key}')
            matches('visual_0',r['visual'][0])
            for j in range(3): matches('visual_1_'+str(j),r['visual'][1][j])
            for k in ('inputs_embeds','attention_mask','position_ids','cache_position','visual_pos_masks'):
                meta=r['llm'][0][k]
                if meta is not None: matches('prefill_'+k,meta)
            for j,meta in enumerate(r['llm'][0]['deepstack_visual_embeds'] or []):matches('prefill_deepstack_visual_embeds_'+str(j),meta)
            for j,meta in enumerate(r.get('selected') or []):matches('selected_'+str(j),meta)
            for key,meta in r['processor'].items():matches('processor_'+key,meta)
            for j,meta in enumerate(r['logits']):matches('logits_'+str(j),meta)
        a,b=archives['native'],archives['on']
        for key in ('visual_0','visual_1_0','visual_1_1','visual_1_2','prefill_inputs_embeds','prefill_attention_mask','prefill_position_ids','prefill_visual_pos_masks','prefill_deepstack_visual_embeds_0','prefill_deepstack_visual_embeds_1','prefill_deepstack_visual_embeds_2'):
            check(np.array_equal(a[key],b[key]),f'{tag} independent native/on {key}')
        for mode in ('k256_on','off','zero'):
            r=d[mode];z=archives[mode];idx=np.array(r['spatial_values']);start=r['visual_start'];retained=np.array(r['retained_values'])
            check(np.array_equal(z['prefill_inputs_embeds'],a['prefill_inputs_embeds'][:,retained,:]),f'{tag} {mode} independent text/visual input')
            check(np.array_equal(z['prefill_inputs_embeds'][:,start:start+256,:][0],z['visual_0'][idx]),f'{tag} {mode} independent main selection')
            check(np.array_equal(z['prefill_position_ids'],a['prefill_position_ids'][:,:,retained]),f'{tag} {mode} independent P3 gather')
            for j in range(3):
                check(np.array_equal(z['selected_'+str(j)],z['visual_1_'+str(j)][idx]),f'{tag} {mode} independent DS gather {j}')
                if mode=='k256_on':check(np.array_equal(z['selected_'+str(j)],z['prefill_deepstack_visual_embeds_'+str(j)]),f'{tag} {mode} independent DS injection {j}')
                if mode=='zero':check(not np.any(z['prefill_deepstack_visual_embeds_'+str(j)]),f'{tag} independent zero slot {j}')
            if mode=='off':check(not any(x.startswith('prefill_deepstack_visual_embeds') for x in z.files) and 'prefill_visual_pos_masks' not in z.files,f'{tag} independent off')
            else:
                mask=z['prefill_visual_pos_masks'];expected=np.zeros_like(mask);expected[:,start:start+256]=True
                check(np.array_equal(mask,expected),f'{tag} {mode} independent visual mask')
        for left,right in (('native','on'),('off','zero')):
            for j in range(33 if condition=='forced' else 1):
                check(np.array_equal(archives[left]['logits_'+str(j)],archives[right]['logits_'+str(j)]),f'{tag} independent {left}/{right} logits {j}')
    finally:
        for z in archives.values(): z.close()

def main():
    verified=0
    try:
        launch=read(HERE/'array16_launch.json'); ex=read(HERE/'array16_exit.json')
        monitor=read(HERE/'monitor_array16/terminal.json'); worker=read(RUN/'runtime.json'); done=read(RUN/'worker_done.json')
        panel=read(PANEL)['panel'][:16]
        split=read(EXCLUSION/'summary.json'); split_check=read(EXCLUSION/'verification.json')
        check(split_check['status']=='pass' and split_check['summary_sha256']==sha(EXCLUSION/'summary.json') and split['anchor_distinct_groups']==16,'anchor exclusion verification')
        check(split['anchor_question_counts']=={'DocVQA_VAL:CONFIRM':2,'DocVQA_VAL:DEV':1,'DocVQA_VAL:FINAL':13},'anchor split overlap counts')
        prior=read(SOURCE_PLAN); weight=read(HERE/'array16_weight_midrun.json'); post_weight=read(HERE/'array16_weights_postrun.json')
        check(post_weight['all_match'] and post_weight['start_unix'] >= ex['finished_unix'] and len(post_weight['records'])==4,'postrun weight hashes')
        for rec in post_weight['records']:
            check(rec['match'] and rec['sha256']==prior['model_stat_guards'][rec['path']]['prior_verified_sha256'] and rec['stat_before']==rec['stat_after']==prior['model_stat_guards'][rec['path']]['current_full_stat'],'postrun weight hash '+rec['path'])
        check(weight['all_match'] and launch['started_unix'] <= weight['unix'] <= ex['finished_unix'],'midrun weight stat evidence')
        for rec in weight['records']:
            check(rec['matches'] and rec['actual']==prior['model_stat_guards'][rec['path']]['current_full_stat'],'midrun weight stat '+rec['path'])
            p=Path(rec['path']); st=p.stat(); current=dict(resolved=str(p.resolve()),bytes=st.st_size,device=st.st_dev,inode=st.st_ino,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)
            check(current==rec['actual'],'postrun weight stat '+rec['path'])
        for group in ('source_sha256','implementation_sha256'):
            for path,digest in prior[group].items():check(sha(Path(path))==digest,'postrun original source identity '+path)
        check(len(panel)==16 and done['n']==16 and done['complete'] and done['forced_decode_steps']==32,'worker complete/count')
        check(ex['returncode']==0 and not ex['timed_out'] and ex['interrupted'] is None and ex['gpu_pids_after']==[], 'container client exit')
        check(ex['container_identity'] is not None and ex['container_identity']['host_init'] is not None,'missing container identity')
        worker_identity=read(HERE/'array16_worker_midrun.json')
        check(worker_identity['host_init']==ex['container_identity']['host_init'] and worker_identity['container_id']==ex['container_identity']['container_id'] and worker_identity['start_ticks_matches'],'worker identity mismatch')
        check(worker_identity['gpu_compute_pids']==[worker_identity['host_init']['pid']],'unexpected GPU worker')
        check(launch['started_unix'] <= worker_identity['observed_unix'] <= ex['finished_unix'],'worker identity timestamp')
        check(worker_identity['host_cmdline'][1:]==[str(HERE/'array_worker.py'),'--out',str(RUN),'--limit','16'],'worker argv mismatch')
        original = ex['container_identity']['host_init']
        stat_path=Path('/proc')/str(original['pid'])/'stat'
        if stat_path.exists() and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==original['boot_id']:
            live_ticks=int(stat_path.read_text().rsplit(')',1)[1].split()[19])
            check(live_ticks!=original['start_ticks'],'original container init still alive')
        inspect=subprocess.run(['docker','inspect',ex['container_identity']['container_id']],capture_output=True,text=True,timeout=10)
        if inspect.returncode==0:
            check(not json.loads(inspect.stdout)[0]['State']['Running'],'container still running')
        else:
            check('No such object' in inspect.stderr or 'No such container' in inspect.stderr,'docker inspect failed for reason other than removed container')
        check(monitor['reason']=='completed' and monitor['worker_returncode']==0 and not monitor['remaining_process_group'],'supervisor terminal')
        for path,digest in launch['source_sha256'].items(): check(sha(Path(path))==digest,'source changed '+path)
        check(worker['worker_sha256']==launch['source_sha256'][str(HERE/'array_worker.py')] and worker['adapter_sha256']==launch['source_sha256'][str(HERE/'adapter.py')], 'executed source binding')
        check(worker['panel_sha256']==sha(PANEL),'panel identity')
        for i,item in enumerate(panel):
            before=len(errors)
            check(sha(Path(item['image_path']))==item['image_sha256'],f'{i:02d} image hash')
            for condition in ('natural','forced'): one(i,item,condition)
            if len(errors)==before: verified+=1
    except Exception as exc:
        errors.append('verification exception: '+repr(exc))
    manifest=HERE/'anchor16_evidence_manifest.json'
    with manifest.open('x') as f:
        json.dump(evidence,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    report=dict(scope='E0 technical identity only; independent confirm/final require the separately verified anchor-excluded split',
        split_disclosure={'original_DEV_anchor_groups':1,'original_CONFIRM_anchor_groups':2,'original_FINAL_anchor_groups':13,
          'must_exclude_anchor_from_independent_inference':True,'revision':'anchor exclusion after generation; original split preserved'},
        evidence_manifest_sha256=sha(manifest), status='passed' if not errors and verified==16 else 'failed',n_expected=16,n_verified=verified,
        run_id='anchor16',worker_sha256=sha(HERE/'array_worker.py'),adapter_sha256=sha(HERE/'adapter.py'),
        run_manifest_sha256=sha(HERE/'array16_launch.json') if (HERE/'array16_launch.json').exists() else None,
        verifier_sha256=sha(Path(__file__)), supervisor_terminal_sha256=sha(HERE/'monitor_array16/terminal.json') if (HERE/'monitor_array16/terminal.json').exists() else None,
        container_exit_sha256=sha(HERE/'array16_exit.json') if (HERE/'array16_exit.json').exists() else None,
        split_exclusion_summary_sha256=sha(EXCLUSION/'summary.json') if (EXCLUSION/'summary.json').exists() else None,
        split_exclusion_verification_sha256=sha(EXCLUSION/'verification.json') if (EXCLUSION/'verification.json').exists() else None,
        panel_sha256=sha(PANEL), worker_midrun_sha256=sha(HERE/'array16_worker_midrun.json') if (HERE/'array16_worker_midrun.json').exists() else None,
        weight_midrun_sha256=sha(HERE/'array16_weight_midrun.json') if (HERE/'array16_weight_midrun.json').exists() else None,
        weight_postrun_sha256=sha(HERE/'array16_weights_postrun.json') if (HERE/'array16_weights_postrun.json').exists() else None,
        weight_identity_scope='mid/post stat matched prior identity; post-run full payload SHA checked, not pre-launch SHA', errors=errors)
    target=HERE/'anchor16_verification.json'
    with target.open('x') as f: json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report['status']=='passed'

if __name__=='__main__': raise SystemExit(0 if main() else 1)
