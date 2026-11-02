"""CPU-only proof of the proposed late CLIP-text AST boundary.

This module never runs or imports the author model, creates a plan, or
allocates a device. It returns ASTs only; no author source is written.
"""
import ast,copy,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
MMB=HERE.parent
ROOT=MMB.parents[2]
SOURCE=ROOT/'_reproduction/author_eadp/LLaVA/llava/eval/model_vqa_mmbench.py'
STREAM=MMB/'streamwait_control'
EXPECTED_SOURCE_SHA='957de2b23d0eae76afabb8e3e9d87ae62898185c046e8dd0c1bd398ccab7ca3e'

def target(fn):
 found=[n for n in ast.walk(fn) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
  and n.targets[0].id=='question' and isinstance(n.value,ast.Name) and n.value.id=='cur_prompt']
 if len(found)!=1:raise ValueError('expected one native question=cur_prompt assignment')
 return found[0]

def transform(original):
 changed=copy.deepcopy(original)
 f=next(n for n in changed.body if isinstance(n,ast.FunctionDef) and n.name=='eval_model')
 node=target(f);node.value=ast.Subscript(value=ast.Name(id='row',ctx=ast.Load()),slice=ast.Constant(value='question'),ctx=ast.Load())
 changed=ast.fix_missing_locations(changed)
 return changed

def bodyfn(tree):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='eval_model')
def ast_sha(tree):return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
def codefn(code):
 values=[v for v in code.co_consts if hasattr(v,'co_name') and v.co_name=='eval_model']
 if len(values)!=1:raise ValueError('compiled eval_model missing or duplicated')
 return values[0]

def proof():
 data=SOURCE.read_bytes();assert hashlib.sha256(data).hexdigest()==EXPECTED_SOURCE_SHA
 original=ast.parse(data,filename=str(SOURCE));f=bodyfn(original);node=target(f)
 assert node.lineno==118
 # All these occur in the same loop after the LLM prompt has been assembled:
 prompt=[n for n in ast.walk(f) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='prompt' for x in n.targets)]
 calls=[n for n in ast.walk(f) if isinstance(n,ast.Call) and any(k.arg=='texts' and isinstance(k.value,ast.Name) and k.value.id=='question' for k in n.keywords)]
 assert len(prompt)==len(calls)==1 and prompt[0].lineno<node.lineno<calls[0].lineno
 changed=transform(original);cf=bodyfn(changed);selected=[n for n in ast.walk(cf) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='question' and isinstance(n.value,ast.Subscript) and isinstance(n.value.value,ast.Name) and n.value.value.id=='row' and isinstance(n.value.slice,ast.Constant) and n.value.slice.value=='question' and n.lineno==node.lineno]
 assert len(selected)==1
 restored=copy.deepcopy(changed);rf=bodyfn(restored);r=[n for n in ast.walk(rf) if isinstance(n,ast.Assign) and n.lineno==node.lineno];assert len(r)==1
 r[0].value=ast.Name(id='cur_prompt',ctx=ast.Load());restored=ast.fix_missing_locations(restored)
 assert ast.dump(restored,include_attributes=False)==ast.dump(original,include_attributes=False)
 orig_code=codefn(compile(original,str(SOURCE),'exec',dont_inherit=True,optimize=0));patched_code=codefn(compile(changed,str(SOURCE),'exec',dont_inherit=True,optimize=0));restored_code=codefn(compile(restored,str(SOURCE),'exec',dont_inherit=True,optimize=0))
 assert orig_code==restored_code and orig_code!=patched_code and all(x.co_filename==str(SOURCE) for x in (orig_code,patched_code,restored_code))
 sys.path.insert(0,str(STREAM));from patch_only import build_patch
 _,wait=build_patch();prior=json.loads((STREAM/'runtime_patch_validation.json').read_text())
 for key in ('source_sha256','original_forward_ast_sha256','patched_forward_ast_sha256','exact_added_statement_count','added_statements','restored_ast_identical'):
  assert wait[key]==prior[key],key
 assert wait['exact_added_statement_count']==3 and wait['restored_ast_identical']
 return dict(success=True,author_source_sha256=EXPECTED_SOURCE_SHA,original_ast_sha256=ast_sha(original),variant_ast_sha256=ast_sha(changed),restored_ast_sha256=ast_sha(restored),
  original_assignment_line=node.lineno,LLM_prompt_line=prompt[0].lineno,texts_call_line=calls[0].lineno,one_assignment_changed=True,full_restored_AST_equal=True,
  same_process_compiled_original_equals_restored=True,same_process_compiled_original_differs_variant=True,filename_checked=True,
  same_prior_three_wait_AST=True,wait_forward_ast_sha256=wait['patched_forward_ast_sha256'],model_loaded=False,GPU_started=False,GT_read=False,scoring_started=False)
if __name__=='__main__':print(json.dumps(proof(),ensure_ascii=False))
