import copy,unittest
import contract,controller
import test_contract as fixtures
class PixelModes(unittest.TestCase):
 def test_three_real_modes_bind_release_and_sample(self):
  f=fixtures.MemoryHandshakeTests();f.setUp()
  for mode in ('t0_author','t020_author','t020_wavefront'):
   e=dict(f.e,mode=mode);r=dict(f.r,mode=mode);s=dict(f.s,mode=mode)
   release=contract.admit_release(e,r,s,f.first)
   self.assertEqual(contract.audit(e,release,f.samples,f.terminal,29)['mode'],mode)
   bad=dict(r,mode='C-ON')
   with self.assertRaises(ValueError):contract.admit_release(e,bad,s,f.first)
 def test_legacy_mode_cannot_be_mislabeled(self):
  f=fixtures.MemoryHandshakeTests();f.setUp()
  with self.assertRaises(ValueError):contract.identity(dict(f.e,mode='C-ON'))
  with self.assertRaises(ValueError):controller.construct(directory='.',mode='C-ON',nonce='a'*32,gpu_uuid='GPU-abcd',create_argv=[],name='x',image='x',worker_argv=[],sampler_command=[],sampler_worker_command=[],sampler_monitor='x')
