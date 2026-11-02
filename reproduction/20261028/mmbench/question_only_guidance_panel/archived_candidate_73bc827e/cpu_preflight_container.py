"""Run frozen tests inside an author-runtime container with no device nodes."""
import glob, json, os, subprocess, sys
import common as c

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not glob.glob('/dev/nvidia*')
    p=c.read(c.PLAN);c.check(p)
    sys.path.insert(0,str(c.HERE.parent/'question_only_guidance_panel_design'))
    import source_boundary
    assert source_boundary.proof()['success']
    suites=[['-m','unittest','-v','test_sanitized_inputs'],['-m','unittest','-v','test_controller','test_observer','test_verifier'],['-m','unittest','-v','test_source_identity']]
    results=[]
    for args in suites:
        result=subprocess.run([sys.executable,'-B',*args],cwd=str(c.HERE),capture_output=True,text=True)
        results.append(dict(args=args,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    import torch
    assert torch.__version__=='2.1.2+cu118' and not torch.cuda.is_initialized()
    print(json.dumps(dict(success=all(r['returncode']==0 for r in results),plan_sha256=c.sha(c.PLAN),author_torch=torch.__version__,
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],device_nodes=glob.glob('/dev/nvidia*'),model_loaded=False,GPU_started=False,GT_read=False,scoring=False,results=results),ensure_ascii=False))
    assert all(r['returncode']==0 for r in results)
if __name__=='__main__':main()
