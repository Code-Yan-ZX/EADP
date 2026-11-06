import unittest
from unittest.mock import patch
import observer_guard_v3 as guard

class GuardTest(unittest.TestCase):
 def test_sufficient_and_exact_conservative_bound(self):
  v=guard.lifetime(100000,100,1010_000_000_000,1010_000_000_000,23000,21900)
  self.assertEqual(v['observer_start_lower_monotonic_ns'],999_990_000_000)
  self.assertGreater(v['deadline_lower_monotonic_ns'],v['checked_monotonic_ns']+21900_000_000_000)
 def test_insufficient_and_expired(self):
  for now in (2200,26000):
   with self.subTest(now=now),self.assertRaises(ValueError):guard.lifetime(100000,100,now*10**9,now*10**9,23000,21900)
 def test_suspend_offset_is_conservative(self):
  v=guard.lifetime(100000,100,1010*10**9,1020*10**9,23000,21900)
  self.assertEqual(v['observer_start_lower_monotonic_ns'],989_990_000_000)
 def test_wrong_parent_and_zombie_rejected_before_query(self):
  for obs,parent in [({'state':'S','parent_pid':5,'start_ticks':100},{'state':'S'}),({'state':'Z','parent_pid':7,'start_ticks':100},{'state':'S'}),({'state':'S','parent_pid':7,'start_ticks':100},{'state':'Z'})]:
   with self.subTest(obs=obs,parent=parent),patch.object(guard,'proc_stat',side_effect=[obs,parent]),self.assertRaises(ValueError):guard.verify(8,7,'/unused','/unused',21600,300)
 def test_bad_clocks_and_future_start_rejected(self):
  for x in [(0,100,1010*10**9,1010*10**9,23000,21900),(100000,100,1010*10**9,1009*10**9,23000,21900),(200000,100,1010*10**9,1010*10**9,23000,21900)]:
   with self.subTest(x=x),self.assertRaises(ValueError):guard.lifetime(*x)
if __name__=='__main__':unittest.main()
