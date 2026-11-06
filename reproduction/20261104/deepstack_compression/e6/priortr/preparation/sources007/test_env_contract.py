import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import env_contract
class EnvTests(unittest.TestCase):
 def test_inventory_detects_change_or_injection(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);env=root/'env';env.mkdir();item=env/'module.py';item.write_bytes(b'original');(env/'alias').symlink_to('module.py')
   digest=hashlib.sha256(item.read_bytes()).hexdigest();fingerprint={'status':'passed','root':str(env),'files':{'module.py':{'bytes':8,'sha256':digest}},'symlinks':{'alias':'module.py'}}
   (root/'env_fingerprint.json').write_text(json.dumps(fingerprint))
   with patch.object(env_contract,'INSTALL',root):
    env_contract.verify_env();item.write_bytes(b'different')
    with self.assertRaises(ValueError):env_contract.verify_env()
    item.write_bytes(b'original');(env/'injected.py').write_text('x')
    with self.assertRaises(ValueError):env_contract.verify_env()
    (env/'injected.py').unlink();(env/'alias').unlink();(env/'alias').symlink_to('wrong')
    with self.assertRaises(ValueError):env_contract.verify_env()
if __name__=='__main__':unittest.main()
