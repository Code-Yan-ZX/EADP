"""CPU-only execution identity checks for the two native entry code objects."""
import ast,inspect,sys,unittest
import common as c

class Tests(unittest.TestCase):
    def test_actual_source_module_and_wait_decorator(self):
        import torch
        assert not torch.cuda.is_initialized()
        sys.path.insert(0,str(c.SOURCE.parents[2]))
        sys.path.insert(0,str(c.HERE.parent/'question_only_guidance_panel_design'))
        sys.path.insert(0,str(c.HERE.parent/'streamwait_control'))
        import source_boundary as b
        from patch_only import build_patch
        original=ast.parse(c.SOURCE.read_bytes(),filename=str(c.SOURCE))
        for arm,tree in zip(('source','question_only'),(original,b.transform(original))):
            code=compile(tree,str(c.SOURCE),'exec',dont_inherit=True,optimize=0)
            ns={'__name__':'cpu_test_'+arm,'__file__':str(c.SOURCE)}
            exec(code,ns)
            self.assertEqual(ns['eval_model'].__code__,b.codefn(code))
            self.assertEqual(ns['eval_model'].__code__.co_filename,str(c.SOURCE))
        module,proof=build_patch()
        name=proof['source_file']+':in_memory_mmb_streamwait'
        code=compile(module,name,'exec',dont_inherit=True,optimize=0);ns={'torch':torch};exec(code,ns)
        actual=inspect.unwrap(ns['forward']).__code__
        expected=[v for v in code.co_consts if getattr(v,'co_name',None)=='forward']
        self.assertEqual(len(expected),1);self.assertEqual(actual,expected[0]);self.assertEqual(actual.co_filename,name)
        self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':unittest.main()
