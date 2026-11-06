"""Fail closed before decoding any oracle predictions or source answers."""
import importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXEC=ROOT/'m3_oracle_exec'
sys.path.insert(0,str(EXEC))
import runtime_gate,e1_admission
V2=ROOT/'m3_oracle_verify_v2'
spec=importlib.util.spec_from_file_location('m3_formal_verifier_v2',V2/'verify.py');verify=importlib.util.module_from_spec(spec);spec.loader.exec_module(verify)

def read(path):return json.loads(Path(path).read_text())
def gate(a):
    run=a.run.resolve();proof=a.proof.resolve();started=a.verify_started.resolve();terminal=a.verify_terminal.resolve()
    expected_paths={run:EXEC/'runtime_001',proof:V2/'runtime_001_verification_author.json',started:V2/'monitor_verify002/started.json',terminal:V2/'monitor_verify002/terminal.json',
                    a.review.resolve():ROOT/'e1_independent_review/m3oracle_panel_review.json',a.review_started.resolve():ROOT/'e1_independent_review/monitor_m3oracle_panel001/started.json',a.review_terminal.resolve():ROOT/'e1_independent_review/monitor_m3oracle_panel001/terminal.json',
                    a.run_started.resolve():EXEC/'monitor_runtime001/started.json',a.run_terminal.resolve():EXEC/'monitor_runtime001/terminal.json'}
    if len(expected_paths)!=9 or any(actual!=expected.resolve() for actual,expected in expected_paths.items()):raise ValueError('wrong accepted run or verifier paths')
    runtime_gate.supervised(started,terminal,V2/'verify.py',proof)
    result=read(proof)
    if (result.get('status')!='passed' or result.get('stage')!='oracle_full' or result.get('n_questions')!=128 or
        result.get('new_generations')!=768 or result.get('scoring_performed') is not False or
        result.get('reference_sources_opened_by_verifier') is not False or result.get('verifier_sha256')!=runtime_gate.sha(V2/'verify.py')):
        raise ValueError('full oracle acceptance required')
    reviewer=ROOT/'e1_independent_review/review_m3oracle_panel.py';review=read(a.review)
    runtime_gate.supervised(a.review_started,a.review_terminal,reviewer,a.review)
    if (review.get('status')!='passed' or review.get('independent_review') is not True or review.get('n_questions')!=128 or review.get('new_generations')!=768 or
        review.get('errors')!=[] or review.get('scoring_performed') is not False or review.get('reference_sources_opened') is not False or
        review.get('formal_proof_sha256')!=runtime_gate.sha(proof) or review.get('formal_started_sha256')!=runtime_gate.sha(started) or
        review.get('formal_terminal_sha256')!=runtime_gate.sha(terminal) or review.get('frozen_sha256')!=runtime_gate.sha(run/'frozen.json') or
        review.get('host_receipt_sha256')!=runtime_gate.sha(run/'host_receipt.json') or review.get('reviewer_sha256')!=runtime_gate.sha(reviewer)):
        raise ValueError('independent oracle review not complete')
    bindingpath=V2/'acceptance_binding_002.json';envpath=V2/'environment_002_v2.json';binding=read(bindingpath);env=read(envpath);vstart=read(started)
    if (binding.get('status')!='passed' or binding.get('formal_proof_sha256')!=runtime_gate.sha(proof) or binding.get('formal_started_sha256')!=runtime_gate.sha(started) or
        binding.get('formal_terminal_sha256')!=runtime_gate.sha(terminal) or binding.get('verifier_sha256')!=runtime_gate.sha(V2/'verify.py') or binding.get('environment_sha256')!=runtime_gate.sha(envpath) or
        binding.get('n_questions')!=128 or binding.get('new_generations')!=768 or binding.get('reference_sources_opened') is not False or binding.get('scoring_performed') is not False or
        review.get('acceptance_binding_sha256')!=runtime_gate.sha(bindingpath) or review.get('verifier_environment_sha256')!=runtime_gate.sha(envpath) or
        env.get('supervisor_started_sha256')!=runtime_gate.sha(started) or env.get('verifier_pid')!=vstart.get('worker_pid') or env.get('python_version')!=binding.get('python_version') or
        env.get('recorded_version_counts')!={binding.get('python_version'):128} or env.get('executable_sha256')!=binding.get('executable_sha256') or
        runtime_gate.sha(env['actual_proc_executable'])!=binding.get('executable_sha256') or env.get('GT_opened') is not False or env.get('reference_token_events_parsed') is not False):
        raise ValueError('actual verifier environment binding mismatch')
    snapshot=V2/'verify_002_sources';manifest=read(snapshot/'manifest.json')
    if (review.get('verifier_snapshot_manifest_sha256')!=runtime_gate.sha(snapshot/'manifest.json') or manifest.get('cpu_proof_sha256')!=runtime_gate.sha(V2/'cpu_check_001/verification.json') or
        any(runtime_gate.sha(path)!=digest or runtime_gate.sha(snapshot/Path(path).name)!=digest for path,digest in manifest.get('source_sha256',{}).items()) or
        manifest.get('source_sha256',{}).get(str(V2/'verify.py'))!=runtime_gate.sha(V2/'verify.py')):raise ValueError('verifier source snapshot mismatch')
    expected={'run_start_sha256':a.run_started,'run_terminal_sha256':a.run_terminal,'launch_sha256':run/'launch.json',
              'exit_sha256':run/'exit.json','panel_events_sha256':run/'panel/panel_events.jsonl',
              'worker_complete_sha256':run/'panel/worker_complete.json'}
    if any(result.get(k)!=runtime_gate.sha(v) for k,v in expected.items()):raise ValueError('accepted oracle evidence changed')
    files=result.get('result_files_sha256',{})
    if len(files)!=128:raise ValueError('accepted question inventory incomplete')
    for path,digest in files.items():
        p=Path(path)
        if p.parent.resolve()!=(run/'panel').resolve() or runtime_gate.sha(p)!=digest:raise ValueError('question journal changed')
    approval=runtime_gate.validate(host=True,require_future=False)
    if result.get('approval_sha256')!=approval['approval_sha256']:raise ValueError('runtime approval changed')
    verify.check_terminal(run,read(a.run_started),read(a.run_terminal),read(run/'launch.json'),read(run/'exit.json'),read(EXEC/'runtime_plan.json'))
    return result
