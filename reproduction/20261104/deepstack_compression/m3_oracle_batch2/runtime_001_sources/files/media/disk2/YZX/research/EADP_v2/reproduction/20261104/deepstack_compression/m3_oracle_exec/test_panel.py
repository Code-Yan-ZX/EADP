import hashlib,importlib.util,json,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import panel,e1_admission
class PanelTests(unittest.TestCase):
    def fixture(self,root):
        image=root/'image';image.write_bytes(b'synthetic');ih=hashlib.sha256(image.read_bytes()).hexdigest();inputs=[];records=[]
        tasks=['DocVQA_VAL']*128+['TextVQA_VAL']*128+['AI2D_TEST']*128
        for i,task in enumerate(tasks):
            row={'task':task,'qid':str(i),'group_id':str(i),'image_path':str(image),'image_sha256':ih,'message_sha256':'m'};inputs.append(row)
            for arm in ('P1-D0','P1-D1','P3-D0','P3-D1'):records.append({'task':task,'qid':str(i),'group_id':str(i),'arm':arm,'forced':False})
        chosen=inputs[:64]+inputs[128:160]+inputs[256:288];s=json.dumps({'examples':chosen,'GT_opened':False,'predictions_opened':False}).encode()
        j=('\n'.join(json.dumps(x) for x in records)+'\n').encode();b=('\n'.join(json.dumps(x) for x in inputs)+'\n').encode();h=lambda x:hashlib.sha256(x).hexdigest()
        p=json.dumps({'status':'passed','stage':'full','natural_questions':384,'natural_records':1536,'tokenizer_decode_checked':1536,'GT_opened':False,'scoring_performed':False,'files':{'journal':{'sha256':h(j)},'inputs':{'sha256':h(b)}}}).encode()
        receipt=e1_admission.Receipt(h(p),h(j),h(b),'d'*64,'e'*64,'f'*64)
        return s,p,j,b,receipt
    def test_full_denominator_and_fail_stop_no_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);s,p,j,b,receipt=self.fixture(root);sample_sha=hashlib.sha256(s).hexdigest()
            wrapper=SimpleNamespace(processor=SimpleNamespace(tokenizer=None));n={'calls':0}
            def question(*args):
                self.assertEqual(args[0]['split'],'DEV')
                n['calls']+=1
                if n['calls']==3:raise ValueError('synthetic failure')
                return {'outputs':[{}]*6}
            with patch.object(panel.extract,'extract',return_value={}),patch.object(panel.one_question,'run',side_effect=question):
                with self.assertRaisesRegex(ValueError,'synthetic failure'):
                    panel.run(s,sample_sha,p,j,b,receipt,root/'run',time.time()+1000,60,wrapper,None,None,None,None,None,None,
                              lambda _:self.fail('truth accessed'),None,{},[],lambda *_:None)
                self.assertFalse((root/'run/worker_complete.json').exists());events=[json.loads(x) for x in (root/'run/panel_events.jsonl').read_text().splitlines()]
                self.assertEqual(events[-1]['event'],'panel_failed');self.assertEqual(sum(x['event']=='question_completed' for x in events),2)
                with self.assertRaises(FileExistsError):
                    panel.run(s,sample_sha,p,j,b,receipt,root/'run',time.time()+1000,60,wrapper,None,None,None,None,None,None,None,None,{},[],None)
if __name__=='__main__':unittest.main()
