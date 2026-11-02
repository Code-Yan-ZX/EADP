import unittest
from observer import Observer

class Handle:
    def __init__(self): self.removed = False
    def remove(self): self.removed = True

class Tower:
    def __init__(self, fail=False): self.fail=fail; self.handle=Handle(); self.hook=None; self.text_tower=self
    def register_forward_pre_hook(self, fn, with_kwargs):
        if self.fail: raise RuntimeError('installation failure')
        self.hook=fn; return self.handle

class Model:
    def __init__(self, fail=False): self.tower=Tower(fail); self.result=(object(),object());self.events=[]
    def get_vision_tower(self): return self.tower
    def encode_images(self,*args,**kwargs): self.events.append('encode');return self.result
    def sim_cross(self,*args,**kwargs): self.events.append('cross');return self.result
    def greed_select(self,*args,**kwargs): self.events.append('greedy');return self.result
    def prepare_inputs_labels_for_multimodal(self,*args,**kwargs): self.events.append('prepare');return (1,2,3,4,5,6,7)

class Tests(unittest.TestCase):
    def test_passthrough(self):
        m=Model();orig=m.encode_images
        with Observer(m) as o:
            self.assertIs(m.encode_images(texts='full'),m.result)
            self.assertIs(m.sim_cross(1,2,3),m.result)
            self.assertIs(m.greed_select(1,2),m.result)
            self.assertEqual(m.prepare_inputs_labels_for_multimodal(9),(1,2,3,4,5,6,7))
            self.assertIsNone(m.tower.hook(None,(),{'input_ids':11,'attention_mask':12}))
            self.assertEqual(o.calls,dict(encode=1,cross=1,greedy=1,prepare=1,clip=1))
            self.assertEqual(o.arrays['clip_input_ids'],11)
        self.assertEqual(m.encode_images,orig);self.assertTrue(m.tower.handle.removed)

    def test_observation_failure_does_not_replace_result(self):
        m=Model()
        with Observer(m) as o:
            self.assertIs(m.sim_cross(),m.result)
            self.assertIsNone(m.tower.hook(None,(),{}))
            self.assertEqual(len(o.errors),2)
            self.assertEqual(m.events,['cross'])

    def test_installation_failure_restores_methods(self):
        m=Model(True);orig={k:getattr(m,k) for k in Observer.names}
        with self.assertRaises(RuntimeError): Observer(m).install()
        self.assertEqual({k:getattr(m,k) for k in Observer.names},orig)

    def test_real_torch_hook_cpu(self):
        import torch
        layer=torch.nn.Identity();seen=[]
        handle=layer.register_forward_pre_hook(lambda module,args,kwargs: seen.append((args,kwargs)),with_kwargs=True)
        try:
            x=torch.tensor([1,2]); result=layer(x)
            self.assertIs(result,x);self.assertEqual(len(seen),1)
        finally: handle.remove()

    def test_other_global_hook_is_rejected_and_restored(self):
        import torch
        m=Model();orig={k:getattr(m,k) for k in Observer.names}
        handle=torch.nn.modules.module.register_module_forward_pre_hook(lambda module,args: None)
        try:
            with self.assertRaises(RuntimeError): Observer(m).install()
            self.assertEqual({k:getattr(m,k) for k in Observer.names},orig)
        finally:handle.remove()

    def test_existing_local_hook_is_rejected(self):
        m=Model();m.tower._forward_pre_hooks={4:object()}
        with self.assertRaises(RuntimeError):Observer(m).install()

if __name__=='__main__':unittest.main()
