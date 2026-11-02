"""Retain each native gain tensor reference; no forward copy or content read."""
import ast
import copy
import hashlib
import marshal
from pathlib import Path
import types

ARCH=Path('/media/disk2/YZX/research/EADP_v2/_reproduction/author_eadp/LLaVA/llava/model/llava_arch.py')
ARCH_SHA='b5bed1a80a72d5600b6229fcf072ab3321114eb1fa3e8b0a759892a8a064b50a'


def transformed():
    assert hashlib.sha256(ARCH.read_bytes()).hexdigest()==ARCH_SHA
    cls=next(n for n in ast.parse(ARCH.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='LlavaMetaForCausalLM')
    original=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='greed_select')
    changed=copy.deepcopy(original);loop=next(n for n in changed.body if isinstance(n,ast.For))
    at=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='best_idx')
    observer=ast.parse('gain_refs.append(total_gain)').body[0];loop.body.insert(at,observer)
    inverse=copy.deepcopy(changed);next(n for n in inverse.body if isinstance(n,ast.For)).body.pop(at)
    digest=lambda n:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
    assert digest(inverse)==digest(original)
    module=ast.fix_missing_locations(ast.Module(body=[changed],type_ignores=[]))
    code=compile(module,str(ARCH)+'[gain-reference observer]','exec')
    fncode=next(n for n in code.co_consts if isinstance(n,types.CodeType) and n.co_name=='greed_select')
    proof=dict(source_file=str(ARCH),source_sha256=ARCH_SHA,original_AST_sha256=digest(original),
        observed_AST_sha256=digest(changed),inverse_restores_original=True,
        insertion='gain_refs.append(total_gain) after selected-mask assignment, before native best_idx; total_gain is not mutated later in the iteration',
        forward_tensor_content_reads=False,forward_tensor_copies=False,
        expected_code_sha256=hashlib.sha256(marshal.dumps(fncode)).hexdigest())
    return code,proof


def install(native,gain_refs):
    namespace=dict(native.__func__.__globals__,gain_refs=gain_refs)
    code,proof=transformed();exec(code,namespace)
    function=namespace['greed_select']
    assert hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()==proof['expected_code_sha256']
    return types.MethodType(function,native.__self__),dict(proof,actual_code_sha256=proof['expected_code_sha256'])
