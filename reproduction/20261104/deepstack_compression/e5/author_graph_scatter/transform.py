"""Construct a one-statement local variant; never modify the author file."""
import ast,hashlib,importlib.util,inspect,textwrap
from pathlib import Path

AUTHOR=Path('/media/disk2/YZX/research/EADP_v2/_reproduction/author_eadp/Qwen_vl/model/pruner.py')
AUTHOR_SHA='bf809614c207142b8a4fc945f29a8828688c88a2896ebd349d55c4a0b1a917e6'
ORIGINAL='selected_mask[batch_indices, best_idx] = True'
REPLACEMENT='selected_mask.scatter_(1, best_idx.unsqueeze(1), True)'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_author():
    if digest(AUTHOR)!=AUTHOR_SHA:raise ValueError('author source changed')
    spec=importlib.util.spec_from_file_location('eadp_scatter_author',AUTHOR)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if Path(inspect.getfile(module._greed_select_impl)).resolve()!=AUTHOR.resolve():raise ValueError('wrong actual author import')
    return module

def build(module):
    source=textwrap.dedent(inspect.getsource(module._greed_select_impl))
    tree=ast.parse(source);old=ast.dump(ast.parse(ORIGINAL).body[0],include_attributes=False)
    new=ast.parse(REPLACEMENT).body[0]
    class Replace(ast.NodeTransformer):
        count=0
        def visit_Assign(self,node):
            if ast.dump(node,include_attributes=False)==old:
                self.count+=1;return ast.copy_location(new,node)
            return self.generic_visit(node)
    replacement=Replace();tree=replacement.visit(tree)
    if replacement.count!=1:raise ValueError('expected exactly one author mask assignment')
    ast.fix_missing_locations(tree)
    env=dict(vars(module));exec(compile(tree,'<author-one-statement-scatter>','exec'),env)
    return env['_greed_select_impl'],{'author_sha256':AUTHOR_SHA,'original_statement':ORIGINAL,
           'replacement_statement':REPLACEMENT,'replacement_count':replacement.count,
           'other_statements':'author AST unchanged','source_written':False}
