import hashlib,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import e1_admission as a
class AdmissionTests(unittest.TestCase):
    def test_formal_reviewer_exit_and_weight_payload_are_bound(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);e1=root/'e1';review=root/'e1_independent_review';analysis=root/'e1_analysis'
            for p in (e1,review,analysis):p.mkdir()
            completion=analysis/'completion.py';completion.write_text('formal')
            review_source=review/'review_full.py';review_source.write_text('independent')
            def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
            weights=[]
            for i in range(1,5):
                f=root/f'model-{i:05d}-of-00004.safetensors';f.write_bytes(bytes([i]))
                weights.append({'path':str(f),'bytes':1,'sha256':h(f),'expected_lfs_sha256':h(f)})
            manifest=root/'weights.json';save(manifest,{'files':weights})
            files={}
            for name in ('journal','inputs','plan','launch','terminal','container_exit','runner','position_adapter'):
                p=root/(name+'.json');p.write_text(name);files[name]={'path':str(p),'sha256':h(p)}
            frozen=root/'frozen.json';save(frozen,{'weights_identity_sha256':h(manifest),'plan_sha256':files['plan']['sha256'],'inputs_sha256':files['inputs']['sha256'],'gt_read':False})
            files['frozen']={'path':str(frozen),'sha256':h(frozen)}
            proof=e1/'e1full001_completion.json';save(proof,{'status':'passed','stage':'full','natural_questions':384,'natural_records':1536,'tokenizer_decode_checked':1536,'GT_opened':False,'scoring_performed':False,'verifier_sha256':h(completion),'files':files})
            def supervisor(monitor,script,out,pid):
                save(monitor/'started.json',{'worker_pid':pid,'cwd':str(root),'command':['python',str(script),'--out',str(out)]})
                save(monitor/'terminal.json',{'worker_pid':pid,'worker_returncode':0,'reason':'completed','remaining_process_group':[]})
            fm=e1/'monitorverifye1full001';supervisor(fm,completion,proof,10)
            independent=review/'e1full001_review.json'
            save(independent,{'status':'passed','n_verified':384,'errors':[],'completion_proof_sha256':h(proof),'completion_started_sha256':h(fm/'started.json'),'completion_terminal_sha256':h(fm/'terminal.json'),'reviewer_sha256':h(review_source)})
            rm=review/'monitor_e1full001';supervisor(rm,review_source,independent,11)
            with patch.object(a,'ROOT',root),patch.object(a,'E1',e1),patch.object(a,'REVIEW',review),patch.object(a,'WEIGHTS',manifest):
                receipt=a.validate();self.assertEqual(receipt.proof_sha,h(proof))
                journal=Path(files['journal']['path']);journal.unlink()
                self.assertEqual(a.validate(metadata_only=True).proof_sha,h(proof))
                with self.assertRaises(FileNotFoundError):a.validate()
                journal.write_text('journal')
                end=json.loads((rm/'terminal.json').read_text());end['worker_pid']=12;save(rm/'terminal.json',end)
                with self.assertRaisesRegex(ValueError,'supervisor'):a.validate()
                end['worker_pid']=11;save(rm/'terminal.json',end);Path(weights[0]['path']).write_bytes(b'x')
                with self.assertRaisesRegex(ValueError,'payload mismatch'):a.validate()
if __name__=='__main__':unittest.main()
