"""Execute bounded CPU design tests, preserve outcome, never touch GPU."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import source_boundary,panel_evidence

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 assert 'torch' not in sys.modules
 files=['README.md','source_boundary.py','test_source_boundary.py','panel_evidence.py','test_panel_evidence.py','run_cpu_design_tests.py']
 before={n:sha(HERE/n) for n in files}
 command=[sys.executable,'-B','-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v']
 test=subprocess.run(command,capture_output=True,text=True,env=os.environ.copy())
 result=dict(status='started',test_actual_exit=test.returncode,test_stdout=test.stdout,test_stderr=test.stderr,files_sha256=before,model_loaded=False,GPU_requested=False,GT_columns_loaded=False,scoring_started=False)
 target=HERE/'cpu_design_report.json';target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 assert test.returncode==0
 result['source_boundary']=source_boundary.proof();result['panel_evidence']=panel_evidence.check()
 assert 'torch' not in sys.modules and {n:sha(HERE/n) for n in files}==before
 result.update(status='passed',torch_imported=False,execution_plan_created=False,queue_created=False)
 target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(status=result['status'],report_sha256=sha(target),GPU_requested=False,tests_exit=test.returncode)))
if __name__=='__main__':main()
