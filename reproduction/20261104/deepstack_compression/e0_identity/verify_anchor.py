"""Independent read-only E0 gate: fail closed on incomplete or inconsistent evidence."""
import hashlib, json, os
from pathlib import Path
HERE=Path(__file__).resolve().parent
RUN=HERE/'anchor16'
PANEL=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261028/qwen/full1024_two_path_panel/plan.json')
MODES=('native','on','k256_on','off','zero')
errors=[]

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
    d={m:read(RUN/f'{i:02d}_{m}_{condition}.json') for m in MODES}
    comp=read(RUN/f'{i:02d}_comparison_{condition}.json')
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


def main():
    verified=0
    try:
        launch=read(HERE/'anchor16_launch.json'); ex=read(HERE/'anchor16_exit.json')
        monitor=read(HERE/'monitor_anchor16/terminal.json'); worker=read(RUN/'runtime.json'); done=read(RUN/'worker_done.json')
        panel=read(PANEL)['panel'][:16]
        check(len(panel)==16 and done['n']==16 and done['complete'] and done['forced_decode_steps']==32,'worker complete/count')
        check(ex['returncode']==0 and not ex['timed_out'] and ex['interrupted'] is None and ex['gpu_pids_after']==[], 'container client exit')
        check(ex['container_identity'] is not None and ex['container_identity']['host_init'] is not None,'missing container identity')
        check(monitor['reason']=='completed' and monitor['worker_returncode']==0 and not monitor['remaining_process_group'],'supervisor terminal')
        for path,digest in launch['source_sha256'].items(): check(sha(Path(path))==digest,'source changed '+path)
        check(worker['worker_sha256']==launch['source_sha256'][str(HERE/'worker.py')] and worker['adapter_sha256']==launch['source_sha256'][str(HERE/'adapter.py')], 'executed source binding')
        check(worker['panel_sha256']==sha(PANEL),'panel identity')
        for i,item in enumerate(panel):
            before=len(errors)
            check(sha(Path(item['image_path']))==item['image_sha256'],f'{i:02d} image hash')
            for condition in ('natural','forced'): one(i,item,condition)
            if len(errors)==before: verified+=1
    except Exception as exc:
        errors.append('verification exception: '+repr(exc))
    report=dict(status='passed' if not errors and verified==16 else 'failed',n_expected=16,n_verified=verified,
        run_id='anchor16',worker_sha256=sha(HERE/'worker.py'),adapter_sha256=sha(HERE/'adapter.py'),
        run_manifest_sha256=sha(HERE/'anchor16_launch.json') if (HERE/'anchor16_launch.json').exists() else None,
        verifier_sha256=sha(Path(__file__)), errors=errors)
    target=HERE/'anchor16_verification.json'
    with target.open('x') as f: json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report['status']=='passed'

if __name__=='__main__': raise SystemExit(0 if main() else 1)
