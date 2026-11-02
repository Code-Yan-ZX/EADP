"""Targeted CPU source/provenance audit; no scoring, model or GPU imports."""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
POPE = OUT.parent
AUTHOR = POPE.parents[2] / '_reproduction/author_eadp/LLaVA'
OLD = Path('/media/disk2/YZX/research/EADP_amp/Qwen_vl/outputs/audit_followup_20261008/rerun_fast')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
ast_sha = lambda n: hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()


def main():
    wrapper = OLD / 'pope_loader_runtime.py'
    worker = OLD / 'run_pope_streamwait_control.py'
    recovery = OLD / 'resume_pope_streamwait_stage2.py'
    protocol_path = OLD / 'pope_v15_EADP32_streamwait.protocol.json'
    protocol = read(protocol_path)
    baseline = read(OLD / 'pope_v15_AZ32.protocol.json')
    recovery_path = OLD / 'pope_streamwait_stage2_recovery.json'
    recovered = read(recovery_path)
    native = AUTHOR / 'llava/eval/model_vqa_loader.py'
    assert protocol['generation_exit_code'] == 0 and protocol['question_count'] == 8910
    parameters = protocol['parameters']
    assert (parameters['visual_token_num'], parameters['alpha'], parameters['beta']) == (32,.5,2.)
    assert parameters['anchorzip'] is False and parameters['max_new_tokens'] == 128
    assert '--anchorzip' not in protocol['generate_command']
    assert sha(wrapper) == protocol['source_and_input_sha256'][str(wrapper)] == baseline['source_and_input_sha256'][str(wrapper)]
    assert sha(worker) == protocol['launcher_source_sha256'] == recovered['worker_sha256']
    assert sha(recovery) == protocol['recovery_source_sha256'] == recovered['recovery_launcher_sha256']
    assert sha(OLD / 'pope_v15_AZ32.protocol.json') == protocol['baseline_protocol_sha256']
    old_tree, new_tree = ast.parse(wrapper.read_text()), ast.parse(native.read_text())
    get = lambda tree, name: next(node for node in tree.body if getattr(node, 'name', None) == name)
    function_checks = []
    for name in ('split_list','get_chunk','CustomDataset','collate_fn','create_data_loader'):
        a,b = get(old_tree,name), get(new_tree,name)
        assert ast.dump(a) == ast.dump(b)
        function_checks.append(dict(name=name, AST_identical=True, AST_sha256=ast_sha(a),
                                    old_line=a.lineno, fresh_line=b.lineno))
    normalized = copy.deepcopy(get(old_tree,'eval_model'))
    removed = []
    def clean(block):
        output = []
        for node in block:
            if isinstance(node,ast.If) and 'getattr(args' in ast.unparse(node.test) and 'anchorzip' in ast.unparse(node.test):
                removed.append(dict(kind='disabled_anchorzip_install_branch',line=node.lineno,reason='Recorded argv has no --anchorzip and parameters.anchorzip=false.'))
                continue
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='actual_visual_token_num' for t in node.targets):
                assert ast.unparse(node.value) == 'int(visual_token_num)'
                removed.append(dict(kind='post_generate_count_logging',line=node.lineno,reason='Only observes the already returned count after native generation.'))
                continue
            for attr in ('body','orelse','finalbody'):
                if isinstance(getattr(node,attr,None),list):setattr(node,attr,clean(getattr(node,attr)))
            output.append(node)
        return output
    normalized.body = clean(normalized.body)
    for node in ast.walk(normalized):
        if isinstance(node,ast.Dict):
            for i,key in enumerate(node.keys):
                if isinstance(key,ast.Constant) and key.value=='metadata' and isinstance(node.values[i],ast.Dict) and node.values[i].keys:
                    removed.append(dict(kind='prediction_observer_metadata',line=node.values[i].lineno,reason='Remove extra count/budget metadata; raw text and all generation operations retained.'))
                    node.values[i]=ast.Dict(keys=[],values=[])
    assert len(removed)==3 and ast.dump(normalized)==ast.dump(get(new_tree,'eval_model'))
    old_patch_path = OLD / 'pope_v15_EADP32_streamwait.ast_validation.json'
    current_patch_path = POPE / 'run03_single_alpha05_beta1_wait_k32/runtime_patch_proof.json'
    old_patch,new_patch = read(old_patch_path),read(current_patch_path)
    assert old_patch['patched_forward_ast_sha256'] == new_patch['patched_forward_ast_sha256']
    assert old_patch['original_forward_ast_sha256'] == new_patch['original_forward_ast_sha256']
    log_path=OLD/'pope_v15_EADP32_streamwait.log'
    with log_path.open() as stream:launch_line=stream.readline()
    installed = json.loads(launch_line)['runtime_patch']
    assert installed['patched_forward_ast_sha256']==old_patch['patched_forward_ast_sha256']
    assert installed['production_source_sha256']==old_patch['production_source_sha256']
    assert sha(Path(old_patch['production_file']))==old_patch['production_source_sha256']
    current_runtime_path=POPE/'run03_single_alpha05_beta1_wait_k32/actual_loaded_runtime.json'
    current_runtime=read(current_runtime_path)
    consistency_path=POPE/'historical_alpha05_beta2_repeat_consistency.json'
    consistency=read(consistency_path)
    raw=Path(consistency['predictions'])
    assert sha(raw)==consistency['sha256'] and consistency['n']==8910
    assert consistency['repeated_groups']==2334 and consistency['yes_no_conflict_groups']==0
    old_binding = [
        dict(path=str(wrapper),sha256=sha(wrapper),generation_time_binding='Recorded in completed protocol and BASE_PROTOCOL.source_and_input_sha256; run_pope_streamwait_control.py:183–186 checks these hashes before runpy loader execution at194.',bound=True),
        dict(path=str(worker),sha256=sha(worker),generation_time_binding='Completed protocol.launcher_source_sha256 and recovery.worker_sha256; recovery lines189–201 checks it immediately before Popen.',bound=True),
        dict(path=str(recovery),sha256=sha(recovery),generation_time_binding='Recovery evidence.recovery_launcher_sha256 copied into completed protocol.recovery_source_sha256.',bound=True),
        dict(path=str(Path(old_patch['production_file'])),sha256=old_patch['production_source_sha256'],generation_time_binding='Frozen input-source hash plus actual worker first stdout runtime_patch record; exact patched AST installed at worker188–194.',bound=True),
    ]
    unbound = []
    for relative in ['llava/model/llava_arch.py','llava/model/builder.py','llava/model/language_model/llava_llama.py','llava/conversation.py','llava/mm_utils.py']:
        old_core=OLD.parents[3]/'LLaVA'/relative
        fresh_core=AUTHOR/relative
        assert str(old_core) not in protocol['source_and_input_sha256']
        assert sha(old_core)==sha(fresh_core)
        unbound.append(dict(historical_inspectable_path=str(old_core),fresh_path=str(fresh_core),
            current_byte_identical=True,current_SHA256=sha(old_core),generation_time_binding_in_this_protocol=False))
    sources=[Path(__file__),wrapper,worker,recovery,protocol_path,OLD/'pope_v15_AZ32.protocol.json',recovery_path,native,
             old_patch_path,current_patch_path,current_runtime_path,consistency_path,POPE/'recipe_and_source_audit.json',
             POPE/'preprocess_identity_comparison.json',POPE/'input_and_score_audit.json',AUTHOR/'llava/model/language_model/llava_llama.py']
    sources += [Path(r[key]) for r in unbound for key in ('historical_inspectable_path','fresh_path')]
    sources=list(dict.fromkeys(sources))
    report=dict(success=True,checked_utc=datetime.now(timezone.utc).isoformat(),CPU_only=True,torch_imported=False,
        model_loaded=False,GPU_started=False,benchmark_scores_computed=False,predictions_parsed=False,
        scope='Targeted inspectable old-wrapper/native source and existing provenance audit; no whole-filesystem search or new generation.',
        helper_generation_time_binding=old_binding,
        source_AST=dict(data_prompt_functions=function_checks,eval_model_matches_after_minimal_changes=True,
            minimal_changes=removed,normalized_old_eval_AST_sha256=ast_sha(normalized),fresh_eval_AST_sha256=ast_sha(get(new_tree,'eval_model')),
            interpretation='Source-level operational equality for the recorded no-AnchorZip path; not proof of historical loaded architecture or complete runtime identity.'),
        wait_patch=dict(original_forward_AST_sha256=old_patch['original_forward_ast_sha256'],
            patched_forward_AST_sha256=old_patch['patched_forward_ast_sha256'],same_as_current_three_waits=True,
            launch_stdout_first_line_sha256=hashlib.sha256(launch_line.encode()).hexdigest(),
            nodes=new_patch['inserted_nodes'],source_lines=dict(old_worker=[187,194],old_recovery=[189,202],current_wait_helper='wait_only_patch.py:31–83')),
        historical_protocol=dict(path=str(protocol_path),sha256=sha(protocol_path),parameters=parameters,
            generate_command=protocol['generate_command'],generation_exit_code=0,question_count=8910,
            raw_prediction=dict(path=str(raw),sha256=consistency['sha256'],bytes=raw.stat().st_size),
            repeated_input_groups=2334,conflict_groups=0,raw_fields_note='Old rows record actual retained count/budget in metadata; no saved comparable input tensors, mask IDs or output token IDs were found in these inspected records.'),
        historical_runtime_limit=dict(actual_backend_component_dtypes_packages_and_arch_environment_not_recorded_in_inspected_sidecars=True,
            currently_inspectable_core_files_without_old_run_hash_binding_in_this_protocol=unbound,
            qualifier='These files are available and current old-tree/fresh copies compare equal where audited, but they are not keys of this old generation protocol source map. Do not backfill at-run core/loaded-module identity from current copies.',
            environment_inheritance='Actual recovery launcher lines198–202 inherits os.environ; USE_LLAVA_ARCH_* is neither cleared nor recorded. Fresh llava_llama.py lines28–38 chooses alternate mixins from those switches.',
            no_observed_evidence_that_an_alternate_arch_or_different_precision_was_actually_used=True,
            current_runtime_only=dict(path=str(current_runtime_path),actual_alpha=current_runtime['actual_alpha'],actual_beta=current_runtime['actual_beta'],
                actual_K=current_runtime['actual_visual_token_num'],backend=current_runtime['attention_backend'],torch=current_runtime['torch'],
                transformers=current_runtime['transformers'],parameter_dtypes=current_runtime['parameter_dtypes'])),
        already_measured_scope=['Existing input_and_score_audit: complete8910 canonical/pinned labels and unchanged historical official rescore; not rescored here.',
            'Existing preprocess_identity_comparison:500images/8910fullLLM+CLIPprompts equal for tested current Pillow/tokenizer pin differences; not a reconstruction of the Oct8 whole runtime.',
            'Existing historical repeated-input audit:2334 duplicate groups,0Yes/No conflicts; not an independent full repeat or old/new runtime equivalence proof.'],
        decision=dict(recommend_new_full8910_now=False,classification='Not shown useless in principle, but no observed source/runtime conflict currently justifies immediate full regeneration.',
            reason='Wrapper arithmetic/input/generation and actual wait patch differences are eliminated at source level; missing historical observations alone do not demonstrate an error or a repair.',
            optional_minimum_if_reproducibility_is_required='Reuse an existing fixed canonical panel at the same .5/2/K32/greedy128/three-wait recipe, compare raw text/counts with preserved old outputs and record actual runtime. No GT/score or score-based sample selection. A match closes only that panel; a mismatch is a concrete reason to investigate, not automatically a full-run mandate.',
            no_new_panel_plan_written=True,score_closeness_used=False),
        sources=[dict(path=str(p),sha256=sha(p)) for p in sources])
    assert 'torch' not in sys.modules
    (OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(success=True,minimal_AST_changes=len(removed),same_wait_AST=True,generation_bound_old_helpers=len(old_binding),new_scoring_or_GPU=False)))


if __name__=='__main__':main()
