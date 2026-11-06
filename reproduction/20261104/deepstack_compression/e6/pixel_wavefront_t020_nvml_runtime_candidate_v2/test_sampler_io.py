import hashlib,itertools,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import test_contract as fixtures
from sampler_io import sample_only

class WiringTests(unittest.TestCase):
 def test_fake_process_and_stopped_inspect_end_to_end(self):
  fixture=fixtures.MemoryHandshakeTests();fixture.setUp();e=fixture.e;t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);d=Path(t.name)
  inspect=d/'inspect.json';inspect.write_text(json.dumps([{'Id':e['container_id'],'State':{'Running':False,'Pid':0}}]));event={k:e[k] for k in ('nonce','container_id','host_pid','start_ticks','boot_id')};event.update(status='stopped',monotonic_ns=18,inspect_sha256=hashlib.sha256(inspect.read_bytes()).hexdigest());(d/'stopped.json').write_text(json.dumps(event))
  live=iter((True,True,True,False));queries=iter(('', 'GPU-1234-abcd, 123, 10', ''));ticks=itertools.count(10)
  def fake_run(*args,**kwargs):return SimpleNamespace(stdout=next(queries))
  result=sample_only(e,d/'journal.jsonl',d/'ready.json',d/'stopped.json',inspect,1,100,alive=lambda _:next(live),clock=lambda:next(ticks),sleep=lambda _:None,run=fake_run)
  data=[json.loads(x) for x in (d/'journal.jsonl').read_text().splitlines()];self.assertEqual(result['samples'],3);self.assertEqual(data[-1]['type'],'summary')
  ready=json.loads((d/'ready.json').read_text());self.assertEqual(ready['first_sample']['used_bytes'],None);self.assertEqual(ready['started']['host_pid'],123)
if __name__=='__main__':unittest.main()
