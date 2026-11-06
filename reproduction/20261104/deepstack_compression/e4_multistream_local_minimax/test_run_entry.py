import tempfile,unittest,sys
from pathlib import Path
from unittest.mock import patch
import run
class TestRun(unittest.TestCase):
 def test_unadmitted_before_torch_or_archive(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);out=p/'result001.json'
   with patch.object(run,'D',p),patch.object(run,'admission',side_effect=ValueError('no prelaunch proof')),patch.object(sys,'argv',['run.py','--out',str(out)]):
    with self.assertRaisesRegex(ValueError,'no prelaunch proof'):run.main()
   self.assertFalse(out.exists());self.assertFalse((p/'rows001.jsonl').exists())
if __name__=='__main__':unittest.main()
