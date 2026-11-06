"""CPU fakes test adapter wiring; they do not emulate actual CUDA graphs."""
import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import selector


class Tensor:
    def __init__(self, shape, value, device='cuda:0'):
        self.shape=shape;self.ndim=len(shape);self.value=value
        self.device=SimpleNamespace(type='cuda',name=device)
        self.dtype='float32';self.copies=[]
    def is_contiguous(self):return True
    def copy_(self, other):self.value=other.value;self.copies.append(other.value);return self


class Graph:
    def __init__(self):self.work=None;self.replays=0
    def replay(self):self.replays+=1;self.work()


class FakeTorch:
    float32='float32'
    def __init__(self):
        self.current=None;self.cuda=SimpleNamespace(CUDAGraph=Graph,graph=self.capture,synchronize=lambda _:None)
    @staticmethod
    def is_tensor(x):return isinstance(x,Tensor)
    @staticmethod
    def empty_like(x):
        t=Tensor(x.shape,None);t.device=x.device;return t
    @contextlib.contextmanager
    def capture(self, graph):
        self.current=graph
        try:yield
        finally:self.current=None


class TestAdapter(unittest.TestCase):
    def setUp(self):
        self.fake=FakeTorch();self.patch=patch.object(selector,'torch',self.fake);self.patch.start()
        self.imp=Tensor((1,4),'A');self.sim=Tensor((1,4,4),'sA');self.sim.device=self.imp.device
    def tearDown(self):self.patch.stop()
    def test_author_invoked_at_capture_and_input_copied_before_replays(self):
        calls=[];order=[];quotas=[]
        def author(imp,sim,k):
            calls.append((imp,sim,k))
            def replay():order[:]=[imp.value,sim.value];quotas[:]=[k]
            self.fake.current.work=replay
            return order,quotas
        candidate=selector.AuthorGraph(self.imp,self.sim,2,author)
        self.assertEqual(len(calls),1);self.assertIs(calls[0][0],candidate.importance)
        self.assertIs(calls[0][1],candidate.similarity)
        changed=Tensor((1,4),'B');changed.device=self.imp.device
        changed_sim=Tensor((1,4,4),'sB');changed_sim.device=self.imp.device
        observed=[]
        for i,s in [(self.imp,self.sim),(changed,changed_sim),(self.imp,self.sim)]:
            a,b=candidate(i,s);observed.append((list(a),list(b)))
        self.assertEqual(observed,[(['A','sA'],[2]),(['B','sB'],[2]),(['A','sA'],[2])])
        self.assertEqual(candidate.graph.replays,3);self.assertEqual(len(calls),1)
    def test_bad_shape_dtype_device_and_k_fail(self):
        for imp,sim,k in [(Tensor((1,4),'a'),Tensor((1,3,3),'b'),2),
                          (self.imp,self.sim,0),(self.imp,self.sim,5)]:
            with self.assertRaises(ValueError):selector.validate(imp,sim,k)
        self.imp.dtype='float16'
        with self.assertRaises(ValueError):selector.validate(self.imp,self.sim,1)
        self.imp.dtype='float32';self.imp.device=SimpleNamespace(type='cpu')
        with self.assertRaises(ValueError):selector.validate(self.imp,self.sim,1)
    def test_reject_missing_or_shape_changed(self):
        with self.assertRaises(ValueError):selector.AuthorGraph(self.imp,self.sim,1,None)
        def author(_a,_b,_k):self.fake.current.work=lambda:None;return [],[]
        candidate=selector.AuthorGraph(self.imp,self.sim,1,author)
        wrong=Tensor((1,5),'x');wrong.device=self.imp.device
        wrongsim=Tensor((1,5,5),'y');wrongsim.device=self.imp.device
        with self.assertRaises(ValueError):candidate(wrong,wrongsim)


if __name__=='__main__':unittest.main(verbosity=2)
