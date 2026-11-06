import hashlib,json,shutil,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import score_oracle as s
class PipelineTests(unittest.TestCase):
    def test_complete_synthetic_panel_preserves_every_arm_and_denominator(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'m3_oracle').mkdir();(root/'m3_oracle_exec').mkdir();(root/'e1').mkdir()
            shutil.copyfile(s.ROOT/'m3_oracle/protocol_001.json',root/'m3_oracle/protocol_001.json')
            shutil.copyfile(s.EXEC/'protocol_amendment_001.json',root/'m3_oracle_exec/protocol_amendment_001.json')
            tasks=['DocVQA_VAL']*64+['TextVQA_VAL']*32+['AI2D_TEST']*32
            samples=[{'task':t,'qid':str(i),'group_id':str(i)} for i,t in enumerate(tasks)]
            sample_path=root/'m3_oracle/sample_001.json';sample_path.write_text(json.dumps({'examples':samples}))
            run=root/'run';(run/'panel').mkdir(parents=True);starts=[];files={}
            sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for i,row in enumerate(samples):
                name=f'{i}.jsonl';starts.append({'event':'question_started',**row,'journal':name})
                events=[]
                for r in (8,16):
                    for source in ('MAIN','MULTI','RANDOM'):
                        events.append({'event':'generation_completed','source':source,'r':r,'output':{'answer':'','eos':True,'truncated':False}})
                q=run/'panel'/name;q.write_text(''.join(json.dumps(x)+'\n' for x in events));files[str(q)]=sha(q)
            (run/'panel/panel_events.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in starts))
            data=root/'summary.json';data.write_text(json.dumps({'sources':{}}));journal=root/'journal';journal.write_bytes(b'fake');inputs=root/'inputs';inputs.write_bytes(b'fake')
            proof={'files':{'journal':{'path':str(journal)},'inputs':{'path':str(inputs)},'dataset_summary':{'path':str(data),'sha256':sha(data)}}}
            (root/'e1/e1full001_completion.json').write_text(json.dumps(proof))
            metrics_files=[]
            for name in ('vqa','misc','match'):
                x=root/name;x.write_text(name);metrics_files.append(x)
            metric=SimpleNamespace(VQA=metrics_files[0],MISC=metrics_files[1],MATCH=metrics_files[2],author_functions=lambda:None,_author_constants=lambda:None,
                                   score=lambda *args,**kwargs:{'official':0.5,'independent':0.5})
            score_module=SimpleNamespace(read_truths=lambda sources,needed:{k:{} for k in needed})
            base={'answer':'','eos':True,'truncated':False};raw=json.dumps(base).encode()
            fake_plan=json.loads((s.HERE/'plan.json').read_text());real_sha=s.score_gate.runtime_gate.sha
            def digest(path):return fake_plan['sample_sha256'] if Path(path)==sample_path else real_sha(path)
            argv=['score_oracle.py']
            for name in ('run','proof','verify-started','verify-terminal','run-started','run-terminal','review','review-started','review-terminal','out'):
                argv += ['--'+name,str(run if name=='run' else (root/'out' if name=='out' else root/name))]
            for name in ('proof','review'):(root/name).write_text(name)
            with patch.object(s.analysis_gate,'validate',return_value={'synthetic':'ok'}),patch.object(s,'ROOT',root),patch.object(s,'EXEC',root/'m3_oracle_exec'),patch.object(s.sys,'argv',argv), \
                 patch.object(s.score_gate,'gate',return_value={'result_files_sha256':files}),patch.object(s.e1_admission,'validate',return_value=object()), \
                 patch.object(s.e1_baseline,'select',return_value=(raw,{},{})),patch.object(s,'author_modules',return_value=(score_module,metric)), \
                 patch.object(s.score_gate.runtime_gate,'sha',side_effect=digest):
                self.assertEqual(s.main(),0)
            report=json.loads((root/'out/analysis.json').read_text());self.assertEqual(report['n_questions'],128)
            self.assertEqual(report['reports']['ALL']['terminal']['EADP']['empty'],128)
            self.assertEqual(len((root/'out/scores.jsonl').read_text().splitlines()),128)
if __name__=='__main__':unittest.main()
