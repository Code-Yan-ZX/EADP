import json
import tempfile
import unittest
from pathlib import Path
import upstream_identity as u

class BundleTests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory(); self.addCleanup(self.t.cleanup); self.d=Path(self.t.name)
  self.r=self.d/'proof.json'; self.script=self.d/'reviewer.py'; self.script.write_text('source')
  self.s=self.d/'started.json'; self.e=self.d/'terminal.json'; self.payload=self.d/'preflight.json'; self.payload.write_text('verified')
  self.proof={'status':'passed','independent_review':True,'scope':'three-anchor t020 real no-GPU preflight result only','gpu_authorized':False,'GT_opened':False,'errors':[],'images':3,'weights_verified':4,'metadata_aliases_verified':7,'source_sha256':{str(self.d/f'source{i}'): 'x' for i in range(226)},'payload_sha256':u.digest(self.payload),'reviewer_sha256':u.digest(self.script)}
  self.start={'cwd':str(self.d),'command':['python',str(self.script),'--out',str(self.r)],'worker_pid':42,'process_group':42}; self.end={'reason':'completed','worker_pid':42,'worker_returncode':0,'remaining_process_group':[]}
  self.write()
 def write(self):
  self.r.write_text(json.dumps(self.proof)); self.s.write_text(json.dumps(self.start)); self.e.write_text(json.dumps(self.end))
 def run_bundle(self):
  return u.validate_bundle({}, {'payload_sha256':self.payload,'reviewer_sha256':self.script},self.r,self.s,self.e,self.script,self.r,rehash_sources=False)
 def test_valid_nogpu_only(self):
  r=self.run_bundle(); self.assertTrue(r['no_gpu_preflight_only']); self.assertFalse(r['candidate_implementation_independently_reviewed'])
 def test_proof_and_scope_rejected(self):
  self.proof['gpu_authorized']=True; self.write()
  with self.assertRaises(ValueError): self.run_bundle()
 def test_wrong_or_unfinished_supervision_rejected(self):
  self.end['worker_returncode']=1; self.write()
  with self.assertRaises(ValueError): self.run_bundle()
  self.end['worker_returncode']=0; self.start['command'][-1]='wrong'; self.write()
  with self.assertRaises(ValueError): self.run_bundle()
 def test_changed_payload_rejected(self):
  self.payload.write_text('different')
  with self.assertRaises(ValueError): self.run_bundle()
 def test_changed_source_inventory_rejected(self):
  self.proof['source_sha256'].pop(next(iter(self.proof['source_sha256']))); self.write()
  with self.assertRaises(ValueError): self.run_bundle()
 def test_expected_hash_tamper_rejected(self):
  with self.assertRaises(ValueError):
   u.validate_bundle({self.script:'0'*64},{},self.r,self.s,self.e,self.script,self.r,rehash_sources=False)

if __name__=='__main__': unittest.main()
