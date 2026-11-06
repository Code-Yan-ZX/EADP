"""Bounded GPU launcher; requires reviewed bundle, preflight, and root scheduling."""
import argparse,json,os,shutil,signal,subprocess,sys,time
from full_imports import HERE,SMOKE,load
ROOT=HERE.parents[1];E4=ROOT/'e4'
sha=load('eadp_selected_full_plan',HERE/'freeze_plan.py').sha
supervise=load('eadp_selected_full_docker_supervisor',SMOKE/'docker_supervisor.py').supervise
add_audit_mount=load('eadp_selected_full_container_mount',SMOKE/'container_mount.py').add
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
 if not a.run_id.isalnum():raise ValueError('invalid run id')
 gate=load('eadp_selected_full_gate',HERE/'gate.py');approval=gate.check(require_future=True,host=True);plan=json.loads((HERE/'plan.json').read_text())
 monitor=json.loads((HERE/('monitor_panel'+a.run_id)/'started.json').read_text())
 if (monitor.get('worker_pid')!=os.getpid() or monitor.get('timeout_seconds')!=plan['outer_timeout_seconds'] or
     monitor.get('stale_seconds')!=plan['progress_stale_seconds'] or
     monitor.get('progress_paths')!=[str((HERE/('panel_'+a.run_id)/'events.jsonl').resolve())] or
     monitor.get('snapshot_seconds')!=plan['snapshot_seconds'] or monitor.get('poll_seconds')!=plan['poll_seconds'] or
     monitor.get('grace_seconds')!=plan['grace_seconds']):raise ValueError('supervisor bounds mismatch')
 if shutil.disk_usage(HERE).free<plan['minimum_free_bytes']:raise ValueError('insufficient disk')
 app=json.loads(gate.APPROVAL.read_text());pre=json.loads(open(app['preflight_proof']).read());identity=pre.get('container_id')
 check=subprocess.run(['docker','inspect',identity],capture_output=True,text=True,timeout=20,check=True);obj=json.loads(check.stdout)[0]
 if obj.get('Id')!=identity or obj['State'].get('Running') is not False or obj['State'].get('Pid')!=0 or obj['State'].get('ExitCode')!=0:
  raise ValueError('preflight real container state cannot be confirmed')
 iw=load('identity_worker',E4/'identity_worker.py');author=json.loads(iw.PLAN.read_text())
 for path,digest in author['implementation_sha256'].items():
  if sha(path)!=digest:raise ValueError('implementation changed: '+path)
 sys.path.insert(0,str(iw.PLAN.parent));load('diagnose',iw.PLAN.parent/'diagnose.py');run_panel=load('eadp_selected_full_run_panel',iw.PLAN.parent/'run_panel.py')
 if run_panel.gpu_pids():raise ValueError('GPU occupied')
 out=HERE/('panel_'+a.run_id);out.mkdir(exist_ok=False);receipt=out/'host_receipt.json'
 with receipt.open('x') as f:json.dump(approval,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 name='eadp-selected-full-'+a.run_id+'-'+str(os.getpid());cmd=add_audit_mount(run_panel.container_command(author,[str(HERE/'worker.py'),'--execute','--out',str(out)],True,name))
 if cmd.count('--rm')!=1:raise ValueError('unexpected container cleanup policy')
 cmd.remove('--rm');gate.check(require_future=True,host=True);controller=run_panel.identity(os.getpid())
 launch={'command':cmd,'controller':controller,'timeout_seconds':plan['inner_timeout_seconds'],'approval':approval,
         'host_receipt':{'path':str(receipt.resolve()),'sha256':sha(receipt)},'worker_sha256':sha(HERE/'worker.py'),
         'launcher_sha256':sha(__file__),'started_unix':time.time()}
 with (out/'launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 interrupted=[None]
 def stop(signum,_):interrupted[0]=signum
 for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
 r=supervise(cmd,name,out,controller,plan['inner_timeout_seconds'],run_panel.gpu_pids,run_panel.identity,interrupted)
 return 0 if (r['returncode']==0 and r['container_exit_code']==0 and r['timed_out'] is False and r['interrupted'] is None and
              r['container_stopped_verified'] is True and r['client_exited_while_container_running'] is False and
              r['errors']==[] and r['gpu_pids_after']==[]) else 1
if __name__=='__main__':sys.exit(main())
