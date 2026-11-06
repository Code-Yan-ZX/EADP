import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import reference_candidate as r
HERE=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def anchors(self):return [{'task':str(i),'qid':str(i),'group_id':str(i),'split':'DEV'} for i in range(3)]
 def fake_refs(self,anchors):
  return {(arm,a['task'],a['qid']):{'processor':{'id':1},'eos_ids':[9],'selected_count':256,'tokens':[9]} for a in anchors for arm in ('pixel_t0','pixel_t045')}
 def fake_eadp(self,anchors):return {('eadp_k256',a['task'],a['qid']):{'processor':{'id':1},'eos_ids':[9],'selected_count':256,'tokens':[9]} for a in anchors}
 def test_cross_arm_processor_and_eos_mismatch(self):
  anchors=self.anchors();pixels=self.fake_refs(anchors);eadp=self.fake_eadp(anchors)
  with patch.object(r,'pixel_refs',return_value=pixels),patch.object(r,'eadp_refs',return_value=eadp):
   self.assertEqual(len(r.load_refs(anchors)),9)
   pixels[('pixel_t0','0','0')]['processor']={'id':2}
   with self.assertRaisesRegex(ValueError,'processor/EOS'):r.load_refs(anchors)
   pixels[('pixel_t0','0','0')]['processor']={'id':1};pixels[('pixel_t0','0','0')]['eos_ids']=[8]
   with self.assertRaisesRegex(ValueError,'processor/EOS'):r.load_refs(anchors)
 def test_nondev_forbidden_before_sources(self):
  anchors=self.anchors();anchors[0]['split']='FINAL'
  with patch.object(r,'pixel_refs') as pixel, self.assertRaises(ValueError):r.load_refs(anchors)
  pixel.assert_not_called()
 def test_supervisor_must_bind_command_and_output(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);script=p/'review.py';out=p/'proof.json';script.write_text('')
   (p/'started.json').write_text(json.dumps({'worker_pid':10,'cwd':d,'command':[sys.executable,str(script),'--out',str(out)]}))
   (p/'terminal.json').write_text(json.dumps({'reason':'completed','worker_returncode':0,'remaining_process_group':[],'worker_pid':10}))
   r.supervisor(p,script,out)
   with self.assertRaises(ValueError):r.supervisor(p,script,p/'wrong.json')
 def test_container_release_identity_and_timeout(self):
  import gpu_worker_candidate as w
  cid='a'*64;job={'order':0};launch={'controller':{'pid':5}};host={'pid':10,'start_ticks':20,'boot_id':'x'}
  release={'container_id':cid,'job':job,'controller':launch['controller'],'host_init':host,'plan_sha256':'p','launch_sha256':'l'}
  self.assertTrue(w.validate_release(job,launch,release,cid[:12],'l','p'))
  for field,value in [('container_id','b'*64),('launch_sha256','wrong'),('controller',{'pid':6})]:
   old=release[field];release[field]=value
   with self.assertRaises(ValueError):w.validate_release(job,launch,release,cid[:12],'l','p')
   release[field]=old
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(ValueError,'timed out'):w.wait_release(job,Path(d),launch,'p',timeout=0)
 def test_real_gpu_candidate_entry_remains_locked(self):
  proc=subprocess.run([sys.executable,str(HERE/'gpu_worker_candidate.py'),'--job-index','0'],capture_output=True,text=True,timeout=30)
  self.assertNotEqual(proc.returncode,0);self.assertIn('GPU locked',proc.stderr)
if __name__=='__main__':unittest.main(verbosity=2)
