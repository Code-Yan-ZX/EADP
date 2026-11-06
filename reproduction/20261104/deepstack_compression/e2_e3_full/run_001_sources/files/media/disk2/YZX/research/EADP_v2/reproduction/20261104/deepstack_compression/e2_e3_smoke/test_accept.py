import json,tempfile,unittest
from pathlib import Path
from local_imports import load_local
accept=load_local('accept')
class WeightTests(unittest.TestCase):
 def test_manifest_four_shards_and_stats_fail_closed(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);items=[];stats={}
   for i in range(1,5):
    p=root/('model-%05d-of-00004.safetensors'%i);p.write_bytes(bytes([i])*7);s=p.stat();h='a'*64
    items.append({'path':str(p),'bytes':7,'sha256':h,'expected_lfs_sha256':h,'matches':True})
    stats[str(p)]=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
   manifest=root/'weights.json';manifest.write_text(json.dumps({'files':items,'local_snapshot':str(root),'all_weight_payloads_match':True}))
   frozen={'weights_identity_sha256':accept.sha(manifest),'weight_stats_before':stats}
   accept.validate_weights(frozen,manifest)
   wrong=list(items);wrong[0]=dict(wrong[0],path=str(root/'unrelated.safetensors'))
   manifest.write_text(json.dumps({'files':wrong,'local_snapshot':str(root),'all_weight_payloads_match':True}))
   swapped=dict(frozen,weights_identity_sha256=accept.sha(manifest))
   with self.assertRaisesRegex(ValueError,'inventory'):accept.validate_weights(swapped,manifest)
   manifest.write_text(json.dumps({'files':items,'local_snapshot':str(root),'all_weight_payloads_match':True}))
   first=root/'model-00001-of-00004.safetensors'
   missing=dict(frozen,weight_stats_before=dict(stats));missing['weight_stats_before'].pop(str(first))
   with self.assertRaises(ValueError):accept.validate_weights(missing,manifest)
   first.write_bytes(b'changed-length')
   with self.assertRaises(ValueError):accept.validate_weights(frozen,manifest)
   manifest.write_text('{}')
   with self.assertRaises(ValueError):accept.validate_weights(frozen,manifest)
if __name__=='__main__':unittest.main()
