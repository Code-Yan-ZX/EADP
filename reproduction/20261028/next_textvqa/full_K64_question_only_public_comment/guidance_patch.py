"""One source-comment AST insertion; no model import or source-file edit."""
import ast
import copy
import hashlib
import marshal
from pathlib import Path
import types

ENTRY=Path('/media/disk2/YZX/research/EADP_v2/_reproduction/author_eadp/LLaVA/llava/eval/model_vqa_loader.py')
ENTRY_SHA='6c9580ed3e62231c0f30708c5f1ec2c47cf44c266c0c228ef22c1b3b0e79164a'
SEPARATOR='\nReference OCR token'
SUFFIX='\nAnswer the question using a single word or phrase.'
EXECUTED_PROOF=None
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def transformed():
    assert sha(ENTRY)==ENTRY_SHA
    text=ENTRY.read_text()
    assert '# question = question.split("\\nReference OCR token")[0]' in text
    original=ast.parse(text);tree=copy.deepcopy(original)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='eval_model')
    loops=[n for n in fn.body if isinstance(n,ast.For)]
    assert len(loops)==1
    loop=loops[0]
    at=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and ast.unparse(n)=='question = cur_prompt')
    assert ast.unparse(loop.body[at+1])=="question = question.replace('\\nAnswer the question using a single word or phrase.', '')"
    insertion=ast.parse("question = question.split('\\nReference OCR token')[0]").body[0]
    loop.body.insert(at+1,insertion)
    inverse=copy.deepcopy(tree)
    inverse_fn=next(n for n in inverse.body if isinstance(n,ast.FunctionDef) and n.name=='eval_model')
    inverse_loop=next(n for n in inverse_fn.body if isinstance(n,ast.For))
    assert ast.dump(inverse_loop.body.pop(at+1))==ast.dump(insertion)
    assert ast.dump(inverse)==ast.dump(original)
    digest=lambda t:hashlib.sha256(ast.dump(t,include_attributes=False).encode()).hexdigest()
    tree=ast.fix_missing_locations(tree)
    code=compile(tree,str(ENTRY)+'[question-only-guidance AST]','exec')
    eval_code=next(v for v in code.co_consts if isinstance(v,types.CodeType) and v.co_name=='eval_model')
    return tree,dict(source_file=str(ENTRY),source_file_sha256=ENTRY_SHA,
        source_AST_sha256=digest(original),patched_AST_sha256=digest(tree),inverse_restores_original=True,
        sole_change="Insert question=question.split('\\nReference OCR token')[0] after question=cur_prompt",
        CustomDataset_and_full_LLM_prompt_unchanged=True,generate_kwargs_and_pruning_math_unchanged=True,
        expected_eval_model_code_sha256=hashlib.sha256(marshal.dumps(eval_code)).hexdigest())


def proof():return transformed()[1]


def run_path(path,run_name):
    global EXECUTED_PROOF
    assert Path(path).resolve()==ENTRY and run_name=='eadp_full_default' and EXECUTED_PROOF is None
    tree,patch=transformed()
    namespace=dict(__name__=run_name,__file__=str(ENTRY),__package__='',__cached__=None,__spec__=None)
    exec(compile(tree,str(ENTRY)+'[question-only-guidance AST]','exec'),namespace)
    function=namespace['eval_model']
    assert hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()==patch['expected_eval_model_code_sha256']
    EXECUTED_PROOF=dict(patch,executed_eval_model_code_sha256=hashlib.sha256(marshal.dumps(function.__code__)).hexdigest(),
        executed_code_filename=function.__code__.co_filename)
    return namespace
