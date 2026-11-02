"""Host coordinator for one no-device CPU preflight; no GPU launch or queue."""
import json,os,subprocess
import common as c
import cpu_container

def main():
    plan=c.read(c.PLAN);c.check(plan)
    assert not c.OUT.exists() and not (c.HERE/'cpu_preflight.json').exists()
    name='mmb-panel-cpu-preflight-'+str(os.getpid())
    cid=c.HERE/'cpu_preflight.cid'
    assert not cid.exists()
    command=cpu_container.command([str(c.HERE/'cpu_preflight_container.py')],name,cidfile=cid)
    try:
        p=subprocess.run(command,capture_output=True,text=True,timeout=120)
        gone=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
        absent=gone.returncode!=0 and 'No such' in gone.stderr
        details=json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode==0 else None
        result=dict(success=p.returncode==0 and absent and details is not None and details['success'],plan_sha256=c.sha(c.PLAN),
            docker_exit_code=p.returncode,container_name=name,container_id=cid.read_text().strip() if cid.exists() else None,
            container_absent=absent,stdout=p.stdout,stderr=p.stderr,details=details,GPU_started=False,model_loaded=False,scoring=False)
        c.save_new(c.HERE/'cpu_preflight.json',result)
        print(json.dumps({k:result[k] for k in ('success','docker_exit_code','container_id','container_absent','plan_sha256')}))
        if not result['success']:raise SystemExit(1)
    except BaseException:
        subprocess.run(['docker','stop','--time','5',name],capture_output=True,text=True)
        raise
if __name__=='__main__':main()
