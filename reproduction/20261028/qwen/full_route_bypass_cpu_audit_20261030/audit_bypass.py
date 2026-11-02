"""Real author K1024 pruner.forward on all16 qualified CPU cache tensors.

No model/processor/generation/selector/GT/scoring or CUDA context is used.
This validates the existing no-pruning branch, not either full LLM route.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path('/media/disk2/YZX/research/EADP_v2')
PANEL = ROOT / 'reproduction/20261028/qwen/gpu_formula_facility_panel_run02'
SOURCE = ROOT / '_reproduction/author_eadp/Qwen_vl/model/pruner.py'
LOCKED = {
    SOURCE: 'bf809614c207142b8a4fc945f29a8828688c88a2896ebd349d55c4a0b1a917e6',
    PANEL / 'run_plan.json': 'd752c876367803c5728ffa84a819fd63b4c129cfd0e2eb78931f7d94dae5d4ad',
    PANEL / 'execution/per_question.jsonl': '9429b7ae99b3ca41bafafb1cc9f16e69460331be5dadc4160db59a60ce3fbaab',
    PANEL / 'execution/runtime.json': 'e95d58b37b79e0506e8217905a760c2c3c18da593b02a5c25ecef5a4e4854cef',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def describe(path):
    return dict(path=str(path), sha256=sha(path), bytes=Path(path).stat().st_size)


def main(output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only invocation required'
    assert not output.exists(), 'Preserve earlier reports; no replacement'
    for path, digest in LOCKED.items():
        assert sha(path) == digest, str(path)
    plan = json.loads((PANEL / 'run_plan.json').read_text())
    records = [json.loads(line) for line in (PANEL / 'execution/per_question.jsonl').read_text().splitlines()]
    expected_positions = [((2*i+1)*5349)//32 for i in range(16)]
    assert len(records) == len(plan['panel']) == 16
    assert [row['position'] for row in records] == [row['position'] for row in plan['panel']] == expected_positions
    assert len({row['index'] for row in records}) == 16
    import numpy as np
    import torch
    torch.set_num_threads(4)
    assert torch.__version__ == '2.8.0+cu128'
    assert not torch.cuda.is_initialized()
    spec = importlib.util.spec_from_file_location('frozen_author_pruner_bypass_only', SOURCE)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    assert Path(source.__file__).resolve() == SOURCE.resolve()
    # The actual native forward is retained. Every non-bypass numerical path
    # raises, proving it was not silently exercised for the all-token control.
    calls = dict(compute_importance=0, sim_visual=0, greedy=0)
    def forbidden(name):
        def stop(*args, **kwargs):
            calls[name] += 1
            raise AssertionError('K1024 entered forbidden selection path: '+name)
        return stop
    pruner = source.VisualTokenPruner(visual_token_num=1024, alpha=.5, beta=2.,
                                     visual_dim=4096, spatial_merge_size=2, device='cpu')
    assert not list(pruner.parameters())
    original_forward = type(pruner).forward
    pruner.compute_importance = forbidden('compute_importance')
    source._sim_visual_impl = forbidden('sim_visual')
    source._greed_select_impl = forbidden('greedy')
    source_tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in source_tree.body if isinstance(node, ast.ClassDef) and node.name == 'VisualTokenPruner')
    forward_ast = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
    rows = []
    for panel, record in zip(plan['panel'], records):
        assert (record['position'], record['index']) == (panel['position'], panel['index'])
        assert record['native_pruner_anchor'] is True and record['native_get_calls'] == 1
        entry = record['cache']
        path = Path(entry['path'])
        assert path == PANEL / 'execution' / ('question_%02d.npz' % len(rows))
        assert sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes']
        with np.load(path, allow_pickle=False) as cache:
            # Arrays are the saved true BF16 bit patterns, not substituted
            # FP32 values or pixels. No ground-truth or prediction is accessed.
            visual_bits = cache['visual_features_BF16_bits']
            global_bits = cache['text_global_mean_BF16_bits']
            sequence_bits = cache['text_sequence_BF16_bits']
            grid_array = cache['image_grid_thw']
        values = dict(visual_features_BF16_bits=visual_bits, text_global_mean_BF16_bits=global_bits,
                      text_sequence_BF16_bits=sequence_bits, image_grid_thw=grid_array)
        for key, value in values.items():
            meta = record['tensors'][key]
            assert list(value.shape) == meta['shape']
            assert hashlib.sha256(value.tobytes()).hexdigest() == meta['sha256']
        assert visual_bits.dtype == global_bits.dtype == sequence_bits.dtype == np.uint16
        assert visual_bits.shape == (1024,4096) and global_bits.shape == (1,4096)
        assert sequence_bits.shape == (1,record['actual_instruction_L'],4096)
        assert grid_array.dtype == np.int64 and grid_array.tolist() == [[1,64,64]]
        to_bf16 = lambda value: torch.from_numpy(value).view(torch.bfloat16)
        features, global_text, dense_text = map(to_bf16, [visual_bits, global_bits, sequence_bits])
        grid = torch.from_numpy(grid_array)
        assert type(pruner).forward is original_forward
        with torch.inference_mode():
            full, splits = pruner(features, global_text, dense_text, grid)
        assert full.device.type == 'cpu' and full.dtype == torch.bfloat16
        assert full.shape == features.shape and splits == [1024]
        full_bits = full.contiguous().view(torch.uint16).numpy()
        assert full_bits.dtype == visual_bits.dtype and full_bits.tobytes() == visual_bits.tobytes()
        assert (calls == dict(compute_importance=0, sim_visual=0, greedy=0))
        rows.append(dict(position=record['position'], index=record['index'], cache=entry,
            grid_thw=[[1,64,64]], visual_input_shape=[1024,4096], visual_output_shape=list(full.shape),
            dtype='torch.bfloat16', native_returned_split_sizes=splits,
            input_BF16_sha256=record['tensors']['visual_features_BF16_bits']['sha256'],
            output_BF16_sha256=hashlib.sha256(full_bits.tobytes()).hexdigest(),
            all1024_rows_original_order_byte_equal=True, selection_calls=dict(calls)))
    assert len(rows) == 16 and not torch.cuda.is_initialized()
    assert all(sha(path) == digest for path,digest in LOCKED.items())
    report = dict(success=True, CPU_only=True, n_cases=16, native_forward_calls=16,
        no_pruning_bypass_checks=16, all_visual_counts=1024, all_grid_thw=[[1,64,64]],
        every_output_byte_equals_cached_native_visual_input=True,
        source_forward_AST_sha256=hashlib.sha256(ast.dump(forward_ast,include_attributes=False).encode()).hexdigest(),
        source_forward_unchanged=True, instantiated_module_has_parameters=False,
        selection_calls=calls, model_loaded=False, processor_called=False, generated_answers=0,
        GT_read=False, accuracy_computed=False, CUDA_initialized=False,
        python=sys.executable, torch=torch.__version__, torch_cuda=torch.version.cuda,
        pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        script=describe(Path(__file__).resolve()), frozen_inputs=[describe(path) for path in LOCKED], rows=rows,
        limits=[
            'Tests only the unchanged pruner.forward K>=N branch on the16 prior true BF16 source caches, not a model generation or environment reconstruction.',
            'Does not run _get_pruned_image_features, _build_pruned_inputs, prepare, attention, generation, DeepStack or position logic.',
            'K1024 is not among the public model registry128/256/512; any model test requires an explicit diagnostic constructor/partial override and must disclose it.',
            'The cached native input feature/get path was measured under accepted author FA2. This CPU bypass neither invokes FA2 nor proves future GPU paths cannot fail.',
            'A paired same-current-stack legacy K1024 versus plain FixedRes FULL is needed to isolate wrapper routes. Historical FULL94.483963 is not that pair.',
            'No paper recipe, reproduction success, full5349 result or accuracy benefit is established.'])
    with output.open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(dict(success=True,n_cases=16,no_pruning_bypass_checks=16,selection_calls=calls,
        CUDA_initialized=False,model_loaded=False,generated_answers=0,report=str(output))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args().output.resolve())
