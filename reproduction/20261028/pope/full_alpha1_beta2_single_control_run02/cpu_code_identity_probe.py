"""Actual runpy identity preflight; never call eval_model or load model/labels."""
import os
from pathlib import Path
import runpy
import sys
import common as c
import code_identity as ident


def main():
    assert len(sys.argv) == 2 and sys.argv[1] in ('01', '02')
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    destination = c.HERE / ('cpu_code_identity_probe_' + sys.argv[1] + '.json')
    assert not destination.exists()
    panel = c.read(c.PANEL/'plan.json')
    entry = Path(panel['llava_root'])/'llava/eval/model_vqa_loader.py'
    expected_sha = panel['source_sha256'][str(entry)]
    assert c.sha(entry) == expected_sha
    sys.path.insert(0, panel['llava_root']); sys.path.insert(0, str(c.POPE))
    from wait_only_patch import install
    patch = install()
    assert patch['runtime_patch_installed']
    import torch
    assert not torch.cuda.is_initialized()
    actual = runpy.run_path(str(entry), run_name='fresh_author_pope_beta2')['eval_model'].__code__
    compiled = ident.source_code(entry)
    assert actual == compiled and actual.co_filename == str(entry)
    actual_fingerprint = ident.fingerprint(actual)
    assert actual_fingerprint == ident.fingerprint(compiled)
    assert c.sha(entry) == expected_sha
    # Deliberate non-executed modifications must be refused by identity gates.
    changed_bytecode = actual.replace(co_code=bytes([actual.co_code[0] ^ 1])+actual.co_code[1:])
    changed_constant = actual.replace(co_consts=actual.co_consts+('tampered',))
    changed_filename = actual.replace(co_filename='elsewhere')
    assert actual != changed_bytecode and actual_fingerprint != ident.fingerprint(changed_bytecode)
    assert actual != changed_constant and actual_fingerprint != ident.fingerprint(changed_constant)
    assert changed_filename.co_filename != str(entry) and actual_fingerprint != ident.fingerprint(changed_filename)
    assert not torch.cuda.is_initialized()
    c.save(destination, dict(success=True,source_sha256=expected_sha,entry=str(entry),
        eval_model_structural_fingerprint=actual_fingerprint,actual_matches_same_process_compiled_code=True,
        exact_filename=True,mutated_bytecode_rejected=True,mutated_constant_rejected=True,mutated_filename_rejected=True,
        helper_sha256=c.sha(c.HERE/'code_identity.py'),CPU_only=True,CUDA_initialized=False,model_loaded=False,
        eval_model_called=False,GT_read=False,scoring_started=False,torch_version=torch.__version__))


if __name__ == '__main__': main()
