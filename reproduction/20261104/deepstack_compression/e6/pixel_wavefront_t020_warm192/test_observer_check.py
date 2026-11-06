import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import observer_check
class ObserverTest(unittest.TestCase):
 def test_absent_observer_rejects(self):
  with tempfile.TemporaryDirectory() as d,patch.object(observer_check,'D',Path(d)):
   with self.assertRaises(FileNotFoundError):observer_check.live()
if __name__=='__main__':unittest.main()
