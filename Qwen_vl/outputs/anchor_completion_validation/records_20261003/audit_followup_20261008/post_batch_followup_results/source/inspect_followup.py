"""CPU read-only followup snapshot. Writes evidence only to --output."""
import argparse,datetime,hashlib,importlib.util,json,re,subprocess,time
from pathlib import Path
OUT=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('followup_inspection_helpers',OUT/'followup_controller.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)

def snapshot():
 state=c.read(c.STATE);plan=c.read(c.PLAN);reg=c.read(OUT/'registration.json');errors=[];jobs={};processes={}
 try:c.verify(plan)
 except Exception as exc:errors.append(str(exc))
 def proc(pid):
  p=Path('/proc')/str(pid)
  if not p.exists():return dict(pid=pid,alive=False)
  st=(p/'stat').read_text().rsplit(')',1)[1].split();return dict(pid=pid,alive=st[0]not in ['Z','X'],state=st[0],ppid=int(st[1]),start_ticks=int(st[19]),argv=[v.decode()for v in (p/'cmdline').read_bytes().split(b'\0')if v])
 for role in ['supervisor','publisher']:
  entry=proc(reg[role+'_pid']);processes[role]=entry
  if entry['alive'] and entry['argv']!=reg[role+'_command']:errors.append(role+' argv mismatch')
  if role=='supervisor' and state['status']!='complete'and not entry['alive']:errors.append('Supervisor absent')
  if role=='supervisor'and entry['alive']and entry['start_ticks']!=state['supervisor_identity']['start_ticks']:errors.append('Supervisor startup identity differs')
 pub=c.read(OUT/'group_publish/publish_state.json')
 if pub['status']!='complete'and not processes['publisher']['alive']:errors.append('Publisher absent')
 if pub['status']=='failed_retryable'or pub.get('error'):errors.append('Publisher failure: '+str(pub.get('error')))
 if state['status']=='failed':errors.append('Controller failure: '+str(state.get('error')))
 active=set()
 for k,j in state['jobs'].items():
  data=dict(status=j['status'],pid=j.get('pid'),n=j['n']);jobs[k]=data
  if j['status']=='pending':continue
  identity=proc(j['pid']);data['process']=identity
  if j['status']in ['generating','loading_reserved','scoring']:
   active.add(j['pid'])
   if not identity['alive']or identity.get('argv')!=j['generate_command']or identity.get('ppid')!=state['supervisor_pid']or identity.get('start_ticks')!=j['process_identity']['start_ticks']:errors.append('Worker identity failed '+k)
  for name in ['prediction','runtime','log']:
   p=Path(j['output_paths'][name]);info=dict(path=str(p),exists=p.exists());data[name]=info
   if not p.exists():continue
   raw=p.read_bytes();st=p.stat();info.update(bytes=len(raw),mtime_ns=st.st_mtime_ns,age_seconds=time.time()-st.st_mtime)
   if name!='log':
    rows=[json.loads(v)for v in raw.split(b'\n')[:-1]if v.strip()];info.update(complete_lines=len(rows),partial_bytes=len(raw.rsplit(b'\n',1)[-1]))
    if name=='prediction':
     manifest=c.read(c.read(j['control_protocol'])['identity_manifest'])['rows'];want=[(str(q['question_id']),q['prompt'])for q in manifest[:len(rows)]]
     if [(str(r['question_id']),r['prompt'])for r in rows]!=want or any(not isinstance(r.get('text'),str)or r['text'].strip().upper().startswith('FAILED')for r in rows):errors.append('Prediction prefix invalid '+k)
    if name=='runtime':
     data['actual_loaded_object']=rows[-1].get('actual_model_object')if rows else None
     if any(r['question_position']!=i or r['visual_token_budget_parameter']!=128 or r['loaded_method_parameters']['alpha']!=j['parameters']['alpha']or r['loaded_method_parameters']['beta']!=j['parameters']['beta']for i,r in enumerate(rows)):errors.append('Runtime prefix invalid '+k)
   else:
    text=raw.decode(errors='replace');info['tail']=text.replace('\r','\n').splitlines()[-3:];info['error_signatures']=re.findall(r'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|^FAILED',text,re.M)
    if info['error_signatures']:errors.append('Worker log failure '+k)
  if j['status']=='complete':
   try:data['strict_score']=c.validate_job(plan['jobs'][k],plan)['score']['value']
   except Exception as exc:errors.append('Strict completed validation '+k+': '+str(exc))
   if identity['alive']:errors.append('Complete child still alive '+k)
  data['launch_gate']=j.get('generation_launch_gate')
  if j['status']in ['generating','loading_reserved']:
   elapsed=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(j['started_utc'])).total_seconds()
   if not data['prediction'].get('complete_lines',0)and elapsed>600:errors.append('Loading over10min '+k)
   if data['prediction'].get('exists')and data['log'].get('exists')and min(data['prediction']['age_seconds'],data['log']['age_seconds'])>600:errors.append('Output and log inactive over10min '+k)
 gpu=c.OLD.FROZEN.UTIL.gpu_snapshot();contexts=[r for r in gpu['entries']if not(r['memory_mib']==0 and r.get('alive')and r['command'].startswith('/usr/local/bin/ollama runner '))]
 if len(active)>2 or len(contexts)>2 or any(r['pid']not in active for r in contexts):errors.append('Unknown GPU context or excess reserved slots')
 return dict(checked_utc=c.now(),status='needs_attention'if errors else'healthy',errors=errors,scope=dict(new_jobs=5,new_groups=3,original_arms=75,original_groups=55),controller_status=state['status'],processes=processes,jobs=jobs,gpu=gpu,publisher=pub,source_gate_passed=not any('drift'in e for e in errors),all_experiments_complete=False)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path);args=a.parse_args();r=snapshot()
 if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(r,indent=2))
 print(json.dumps(r,indent=2))
