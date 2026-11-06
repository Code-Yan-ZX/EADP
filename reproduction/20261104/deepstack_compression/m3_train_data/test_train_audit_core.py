import random,unittest
from train_audit_core import Graph,bucket_keys,fit_val,group_id

class GraphTests(unittest.TestCase):
 def test_partition_completeness(self):
  rng=random.Random(321)
  for _ in range(3000):
   x=rng.getrandbits(64);bits=rng.sample(range(64),rng.randrange(5));y=x
   for b in bits:y^=1<<b
   self.assertTrue(set(bucket_keys(x)) & set(bucket_keys(y)))
 def test_transitive_eval_exclusion(self):
  g=Graph();a=g.add('a','a',0,is_eval=True,eval_group='eval');b=g.add('b','b',15);c=g.add('c','c',255);d=g.add('d','d',(1<<64)-1)
  self.assertEqual(g.uf.find(a),g.uf.find(c));self.assertNotEqual(g.uf.find(a),g.uf.find(d));self.assertEqual(g.uf.find(a),g.uf.find(b))
 def test_eval_group_and_semantic_edges(self):
  g=Graph();a=g.add('a','a',0,is_eval=True,eval_group='same');b=g.add('b','b',(1<<64)-1,is_eval=True,eval_group='same');c=g.add('c','c',0xaaaaaaaaaaaaaaaa,semantic=('doc','3'));d=g.add('d','d',0x5555555555555555,semantic=('doc','3'));self.assertEqual(g.uf.find(a),g.uf.find(b));self.assertEqual(g.uf.find(c),g.uf.find(d))
 def test_raw_rgb_and_collision_guard(self):
  g=Graph();a=g.add('a','pix',0);b=g.add('b','pix',(1<<64)-1);self.assertEqual(g.uf.find(a),g.uf.find(b));self.assertEqual(g.add('a','pix',0,is_eval=True),a);self.assertTrue(g.items[a]['is_eval'])
  with self.assertRaises(ValueError):g.add('a','wrong',0)
 def test_split_group_consistency_and_order(self):
  names=[group_id([{'raw':str(i)}]) for i in range(11)];a=fit_val(names);self.assertEqual(a,fit_val(list(reversed(names))));self.assertEqual(list(a.values()).count('fit'),8);self.assertEqual(list(a.values()).count('validation'),3)
 def test_bucket_implementation_matches_brute_force(self):
  rng=random.Random(82);vs=[rng.getrandbits(64) for _ in range(300)];vs += [vs[0]^7,vs[0]^0x7f]
  g=Graph()
  for i,v in enumerate(vs):g.add(str(i),str(i),v)
  for i,x in enumerate(vs):
   for j,y in enumerate(vs[:i]):
    if (x^y).bit_count()<=4:self.assertEqual(g.uf.find(i),g.uf.find(j))
 def test_existing_eval_group_bridges_train_duplicates(self):
  g=Graph();a=g.add('eval_a','ra',0,eval_group='old_group',is_eval=True);b=g.add('eval_b','rb',(1<<64)-1,eval_group='old_group',is_eval=True);c=g.add('train','rc',(1<<64)-2)
  self.assertEqual(g.uf.find(a),g.uf.find(b));self.assertEqual(g.uf.find(a),g.uf.find(c));self.assertTrue(any(g.items[i]['is_eval'] for i in g.components()[g.uf.find(c)]))
 def test_page_index_and_reject_malformed(self):
  from train_audit import semantic_from_eval
  self.assertEqual(semantic_from_eval('a_b_p0.jpg'),('a_b','1'));self.assertEqual(semantic_from_eval('a_p80.png'),('a','81'))
  for bad in ('a.jpg','a_p-1.jpg',''):
   with self.assertRaises(ValueError):semantic_from_eval(bad)
if __name__=='__main__':unittest.main()
