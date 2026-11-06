"""Fail closed before decoding any prediction or DEV truth.

Only accepted metadata/hashes are inspected before both full-panel reviews.
After those pass, smoke records are read solely to recheck their archived array
identities; the complete panel is still not scored until this gate returns.
"""
import hashlib,importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;E6=HERE.parent;BASE=E6.parent;FULL=E6/'selected_full';SMOKE=E6/'selected_smoke';REVIEW=BASE/'e1_independent_review'
RUN=FULL/'panel_001';PROOF=FULL/'panel_001_verification.json';FS=FULL/'monitor_verify001/started.json';FT=FULL/'monitor_verify001/terminal.json'
RS=FULL/'monitor_panel001/started.json';RT=FULL/'monitor_panel001/terminal.json'
RPROOF=REVIEW/'selected_full_panel001_review.json';RSTART=REVIEW/'monitor_selected_full_panel001/started.json';REND=REVIEW/'monitor_selected_full_panel001/terminal.json'
E1=BASE/'e1/e1full001_completion.json';SF=SMOKE/'smoke_001_verification.json';SS=SMOKE/'monitor_verify001/started.json';ST=SMOKE/'monitor_verify001/terminal.json'
SR=REVIEW/'selected_smoke001_review.json';SRS=REVIEW/'monitor_selected_smoke001/started.json';SRT=REVIEW/'monitor_selected_smoke001/terminal.json'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def supervisor(started,terminal,script,out):
 s=read(started);t=read(terminal);cmd=s.get('command',[]);cwd=Path(s.get('cwd','/'))
 if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or t.get('worker_pid')!=s['worker_pid'] or
     [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or cmd.count('--out')!=1 or cmd.index('--out')+1>=len(cmd) or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()):raise ValueError('review supervisor identity or exit mismatch')
 return s

def validate_review_fields(formal,review):
 if (formal.get('status')!='passed' or formal.get('stage')!='full_dev' or formal.get('questions')!=384 or formal.get('generations')!=1536 or formal.get('pairs')!=768 or
     formal.get('GT_opened') is not False or formal.get('scoring') is not False or formal.get('verifier_sha256')!=sha(FULL/'verify.py') or Path(formal.get('run','/')).resolve()!=RUN.resolve()):raise ValueError('full formal proof mismatch')
 if (review.get('status')!='passed' or review.get('independent_review') is not True or review.get('questions')!=384 or review.get('generations')!=1536 or review.get('GT_opened') is not False or review.get('scoring_performed') is not False or review.get('errors')!=[] or
     review.get('formal_proof_sha256')!=sha(PROOF) or review.get('formal_started_sha256')!=sha(FS) or review.get('formal_terminal_sha256')!=sha(FT) or review.get('run_started_sha256')!=sha(RS) or review.get('run_terminal_sha256')!=sha(RT) or review.get('source_manifest_sha256')!=formal.get('source_manifest_sha256') or review.get('reviewer_sha256')!=sha(REVIEW/'review_selected_full_panel.py')):raise ValueError('full independent review mismatch')

def manifest(path,expected,count):
 if sha(path)!=expected:raise ValueError('source manifest changed')
 sources=read(path).get('source_sha256');snapshot=path.parent/'files'
 if not isinstance(sources,dict) or len(sources)!=count:raise ValueError('source manifest inventory mismatch')
 for original,digest in sources.items():
  p=Path(original)
  if not p.is_absolute() or sha(snapshot/p.relative_to('/'))!=digest:raise ValueError('source snapshot changed')
 return sources

def gate():
 # Missing proof is rejected before hashing, reading predictions, or opening GT.
 for path in (PROOF,FS,FT,RS,RT,RPROOF,RSTART,REND):
  if not path.is_file():raise ValueError('full panel and both acceptance chains required')
 supervisor(FS,FT,FULL/'verify.py',PROOF);supervisor(RSTART,REND,REVIEW/'review_selected_full_panel.py',RPROOF)
 formal=read(PROOF);review=read(RPROOF);validate_review_fields(formal,review)
 files=formal.get('files_sha256');required=[RS,RT]+[RUN/x for x in ('frozen.json','worker_complete.json','launch.json','exit.json','records.jsonl','generated.jsonl','events.jsonl','pairs.jsonl')]
 if not isinstance(files,dict) or any(files.get(str(p.resolve()))!=sha(p) for p in required):raise ValueError('full accepted evidence missing or changed')
 if any(sha(path)!=digest for path,digest in files.items()):raise ValueError('full evidence changed')
 full_manifest=FULL/'panel_001_sources/manifest.json';full_sources=manifest(full_manifest,formal['source_manifest_sha256'],126)
 # Re-run runtime's read-only container identity/terminal checks using its own frozen source.
 sys.path.insert(0,str(FULL));from full_imports import load
 verify=load('eadp_selected_scoring_verify',FULL/'verify.py');_launch,_exit,inspection=verify.terminal(RUN,RS,RT)
 if formal.get('inspection')!=inspection:raise ValueError('actual container inspection differs')
 frozen=read(RUN/'frozen.json');complete=read(RUN/'worker_complete.json');full_approval=read(FULL/'approval.json');full_cpu=Path(full_approval['cpu_proof']);full_cpu_proof=read(full_cpu)
 if (sha(FULL/'approval.json')!=frozen.get('approval',{}).get('approval_sha256') or sha(full_cpu)!=frozen.get('approval',{}).get('cpu_sha256') or full_cpu_proof.get('source_sha256_before')!=full_sources or full_cpu_proof.get('source_sha256_after')!=full_sources or frozen.get('approval',{}).get('source_count_verified')!=len(full_sources)):
  raise ValueError('full source manifest differs from accepted CPU source set')
 if (frozen.get('stage')!='full_dev' or frozen.get('GT_opened') is not False or frozen.get('scoring') is not False or frozen.get('e1_completion_sha256')!=sha(E1) or
     complete.get('status')!='completed' or complete.get('questions')!=384 or complete.get('generations')!=1536 or complete.get('pairs')!=768):raise ValueError('full completion metadata differs')
 # Preserve the accepted diagnostics: archived arrays are referenced by smoke
 # records and are checked only after the two full-panel gates have passed.
 supervisor(SS,ST,SMOKE/'verify.py',SF);supervisor(SRS,SRT,REVIEW/'review_selected_smoke.py',SR)
 smoke=read(SF);smoke_review=read(SR)
 if (smoke.get('status')!='passed' or smoke.get('questions')!=3 or smoke.get('generations')!=16 or smoke.get('GT_opened') is not False or smoke.get('scoring') is not False or smoke.get('verifier_sha256')!=sha(SMOKE/'verify.py') or
     smoke_review.get('status')!='passed' or smoke_review.get('independent_review') is not True or smoke_review.get('questions')!=3 or smoke_review.get('generations')!=16 or smoke_review.get('GT_opened') is not False or smoke_review.get('scoring_performed') is not False or
     smoke_review.get('formal_proof_sha256')!=sha(SF) or smoke_review.get('formal_started_sha256')!=sha(SS) or smoke_review.get('formal_terminal_sha256')!=sha(ST) or
     smoke_review.get('run_started_sha256')!=sha(SMOKE/'monitor_smoke001/started.json') or smoke_review.get('run_terminal_sha256')!=sha(SMOKE/'monitor_smoke001/terminal.json') or
     smoke_review.get('source_manifest_sha256')!=smoke.get('source_manifest_sha256') or smoke_review.get('reviewer_sha256')!=sha(REVIEW/'review_selected_smoke.py')):raise ValueError('smoke proof invalid')
 smoke_files=smoke.get('files_sha256',{});smoke_required=[SMOKE/'monitor_smoke001/started.json',SMOKE/'monitor_smoke001/terminal.json']+[SMOKE/'smoke_001'/x for x in ('frozen.json','worker_complete.json','launch.json','exit.json','records.jsonl','generated.jsonl','events.jsonl','comparisons.jsonl')]
 if not isinstance(smoke_files,dict) or any(smoke_files.get(str(path.resolve()))!=sha(path) for path in smoke_required):raise ValueError('smoke evidence inventory mismatch')
 for path,digest in smoke_files.items():
  if sha(path)!=digest:raise ValueError('smoke evidence changed')
 smoke_sources=manifest(SMOKE/'smoke_001_sources/manifest.json',smoke['source_manifest_sha256'],102)
 smoke_frozen=read(SMOKE/'smoke_001/frozen.json');smoke_approval=read(SMOKE/'approval.json');smoke_cpu=Path(smoke_approval['cpu_proof']);smoke_cpu_proof=read(smoke_cpu)
 if (sha(SMOKE/'approval.json')!=smoke_frozen.get('approval',{}).get('approval_sha256') or sha(smoke_cpu)!=smoke_frozen.get('approval',{}).get('cpu_sha256') or smoke_cpu_proof.get('source_sha256_before')!=smoke_sources or smoke_cpu_proof.get('source_sha256_after')!=smoke_sources or smoke_frozen.get('approval',{}).get('source_count_verified')!=len(smoke_sources)):
  raise ValueError('smoke source manifest differs from accepted CPU source set')
 arrays={};smoke_records=SMOKE/'smoke_001/records.jsonl'
 with smoke_records.open() as stream:
  for line in stream:
   if not line.strip():continue
   record=json.loads(line);desc=record['arrays'];path=Path(desc['path']).resolve()
   if desc.get('kind')!='full_arrays' or path.parent!=(SMOKE/'smoke_001/arrays').resolve() or str(path) in arrays or sha(path)!=desc['sha256']:raise ValueError('smoke anchor array changed')
   arrays[str(path)]=desc['sha256']
 if len(arrays)!=16:raise ValueError('smoke anchor array inventory mismatch')
 return {'formal':formal,'review':review,'full_manifest':full_manifest,'smoke':smoke,'smoke_arrays':arrays,'inspection':inspection}
