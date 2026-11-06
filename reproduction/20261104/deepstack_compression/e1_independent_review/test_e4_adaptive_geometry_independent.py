"""Synthetic independent rectangle/occupancy/entropy recomputation."""
import collections,importlib.util,math,random,unittest
from pathlib import Path
P=Path(__file__).resolve().parent.parent/'e4_adaptive_geometry/features.py';sp=importlib.util.spec_from_file_location('untrusted_geometry_test_target',P);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
def grid(h,w):
 xy=[(3+i//w,7+i%w) for i in range(h*w)];return [[[0]*1024],[[x for x,y in xy]],[[y for x,y in xy]]]
def oracle(pos,ids):
 coords=list(zip(pos[1][0],pos[2][0]));a=min(x for x,y in coords);b=max(x for x,y in coords);c=min(y for x,y in coords);d=max(y for x,y in coords);chosen=[coords[i] for i in ids]
 near=sum(min(math.hypot((x-u)/(b-a),(y-v)/(d-c)) for j,(u,v) in enumerate(chosen) if i!=j) for i,(x,y) in enumerate(chosen))/len(chosen)
 def cells(n):return [((x-a)*n//(b-a+1),(y-c)*n//(d-c+1)) for x,y in chosen]
 occ=len(set(cells(8)));counts=collections.Counter(cells(4));entropy=-sum((n/len(ids))*math.log2(n/len(ids)) for n in counts.values());return near,occ,entropy
class Tests(unittest.TestCase):
 def test_unequal_rectangles_and_arbitrary_selection(self):
  for h,w in ((16,64),(8,128),(32,32)):
   pos=grid(h,w);ids=sorted(random.Random(20261105).sample(range(1024),128));out=m.descriptors(pos,0,ids);a,b,c=oracle(pos,ids)
   self.assertAlmostEqual(out['nearest_mean'],a,places=14);self.assertEqual(out['occupied8'],b);self.assertAlmostEqual(out['entropy4'],c,places=14)
 def test_nongrid_and_bad_indices_rejected(self):
  pos=grid(16,64);pos[1][0][0]=1000
  with self.assertRaises(ValueError):m.descriptors(pos,0,list(range(128)))
  for ids in (list(range(127))+[1024],list(range(127))+[-1],list(range(127))+[True]):
   with self.assertRaises(ValueError):m.descriptors(grid(16,64),0,ids)
if __name__=='__main__':unittest.main()
