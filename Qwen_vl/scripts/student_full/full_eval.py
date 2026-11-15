"""Frozen, resumable native DocVQA VAL evaluation; no answers in inference.

Run `freeze`, then `gate`, then `run`. Never select arms using test outcomes.
All seven arms use the same benchmark prompt, greedy decoder, EOS and cap.
"""
from __future__ import annotations
import argparse
import base64
import io
from collections import OrderedDict
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import sqlite3
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
QWEN = HERE.parents[1]
ROOT = QWEN.parent
OUT = QWEN / 'outputs/student_full'
DEFAULT_MANIFEST = OUT / 'docvqa_val_manifest.json'
FREEZE = OUT / 'run_freeze.json'
GATE = OUT / 'dev_gate.json'
DB = OUT / 'predictions.sqlite3'
DISTILL = QWEN / 'outputs/student_distill'
DEV = DISTILL / 'data/dev_manifest.json'
DATASET = 'DocVQA_VAL'
CAP = 128
K = 256
STUDENTS = tuple(f'magnitude_v__seed_{seed}' for seed in (11, 23, 37))
ARMS = ('full', 'b2', 'b1', 'rres512') + STUDENTS
for p in (QWEN / 'scripts/discovery', QWEN / 'scripts/multistream_oracle', QWEN / 'scripts/student_distill', QWEN):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import torch
import common
from model.native_qwen3 import NativeEngine
from experiment_common import resolution
from oracle import generate_selection, tensor_sha256
import student_eval
from student_model import StudentScorer


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): h.update(b)
    return h.hexdigest()


def atomic_json(path, obj, exclusive=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        # O_EXCL prevents silently replacing an existing frozen protocol.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, 'w') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    else:
        tmp = path.with_name(path.name + '.tmp')
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
        os.replace(tmp, path)


def validate_manifest(path):
    payload = json.loads(Path(path).read_text())
    recs = payload['records']
    if not payload.get('complete') or len(recs) != 5349:
        raise ValueError('manifest must contain all 5349 benchmark questions')
    if sorted(int(r['source_row']) for r in recs) != list(range(5349)):
        raise ValueError('manifest source rows are not exactly the full split')
    if len({str(r['index']) for r in recs}) != 5349:
        raise ValueError('original question IDs are not unique')
    for r in recs:
        if any(k not in r for k in ('source_row','index','parent_id','image_path','image_sha256','question')):
            raise ValueError('manifest missing required question/page fields')
    return payload


def source_paths():
    # Independent experiment source files are frozen as a set. Include native
    # implementations, tokenizer/prompt and evaluator provenance as well.
    paths = set(HERE.glob('*.py')) | set(student_eval.source_paths())
    paths.update([QWEN/'VLMEvalKit/vlmeval/vlm/qwen3_vl/prompt.py',
                  QWEN/'VLMEvalKit/vlmeval/dataset/image_base.py',
                  QWEN/'VLMEvalKit/vlmeval/dataset/image_vqa.py',
                  QWEN/'VLMEvalKit/vlmeval/smp/vlm.py',
                  QWEN/'scripts/discovery/instrumented.py'])
    prereg = ROOT/'docs/student_full_prereg.md'
    if prereg.exists(): paths.add(prereg)
    return sorted(p.resolve() for p in paths)


def freeze(manifest):
    manifest = Path(manifest).resolve(); payload = validate_manifest(manifest)
    # Revalidates the complete original pilot freeze, including all checkpoints.
    prev = student_eval.verify_freeze()
    train = json.loads((DISTILL/'train_freeze.json').read_text())
    dev = json.loads(DEV.read_text())['items'][0]
    if sha256(dev['image_path']) != dev['image_sha256']: raise ValueError('dev image changed')
    model_info = train['model']; snap = Path(model_info['snapshot']).resolve()
    if snap.name != model_info['revision'] or not snap.is_dir(): raise ValueError('missing model revision')
    if (snap.parent.parent/'refs/main').read_text().strip() != model_info['revision']:
        raise ValueError('cached model revision differs')
    for name, digest in model_info['metadata_sha256'].items():
        if sha256(snap/name) != digest: raise ValueError('model metadata mismatch: '+name)
    checkpoints = {arm:prev['checkpoints'][arm] for arm in STUDENTS}
    for arm, item in checkpoints.items():
        if sha256(item['path']) != item['sha256']: raise ValueError('checkpoint mismatch: '+arm)
    import transformers
    import PIL
    tsv = Path(os.environ.get('LMUData','/media/disk2/YZX/LMUData'))/'DocVQA_VAL.tsv'
    data = dict(schema_version=1, manifest_path=str(manifest), manifest_sha256=sha256(manifest),
        count=len(payload['records']), original_tsv_path=str(tsv.resolve()), original_tsv_sha256=sha256(tsv),
        prior_eval_freeze=str((DISTILL/'eval_freeze.json').resolve()), prior_eval_freeze_sha256=sha256(DISTILL/'eval_freeze.json'),
        train_freeze=str((DISTILL/'train_freeze.json').resolve()), train_freeze_sha256=sha256(DISTILL/'train_freeze.json'),
        checkpoints=checkpoints, model=model_info,
        source_sha256={str(p):sha256(p) for p in source_paths()},
        local_python_files=[str(p.resolve()) for p in sorted(HERE.glob('*.py'))],
        dev_manifest=str(DEV.resolve()), dev_manifest_sha256=sha256(DEV),
        dev_image_path=dev['image_path'], dev_image_sha256=dev['image_sha256'],
        config=dict(dataset=DATASET, arms=list(ARMS), K=K, full_n_vis=1024, rres=512,
            max_new_tokens=CAP, greedy=True, deepstack=True, position='native_3d_mrope',
            prompt='common.build_message(model,dataset,DocVQA_VAL,row)',
            answer_in_selector=False, teacher_at_inference=False, cache='current page, preprocessed pixel+grid hash',
                    image_materialization='verified source bytes; VLMEvalKit PIL decode and JPEG save',
                    question_preparation='once per question and resolution'),
        environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                         transformers=transformers.__version__, pillow=PIL.__version__, attention_backend=train['environment']['attention_backend']))
    atomic_json(FREEZE, data, exclusive=True)
    print('freeze', FREEZE, sha256(FREEZE), flush=True)


def verify(manifest):
    frozen = json.loads(FREEZE.read_text())
    if str(Path(manifest).resolve()) != frozen['manifest_path'] or sha256(manifest) != frozen['manifest_sha256']:
        raise ValueError('manifest differs from freeze')
    expected = dict(dataset=DATASET, arms=list(ARMS), K=K, full_n_vis=1024, rres=512,
                    max_new_tokens=CAP, greedy=True, deepstack=True, position='native_3d_mrope',
                    prompt='common.build_message(model,dataset,DocVQA_VAL,row)',
                    answer_in_selector=False, teacher_at_inference=False, cache='current page, preprocessed pixel+grid hash',
                    image_materialization='verified source bytes; VLMEvalKit PIL decode and JPEG save',
                    question_preparation='once per question and resolution')
    if frozen['config'] != expected: raise ValueError('runtime config differs from freeze')
    for key in ('prior_eval_freeze','train_freeze','dev_manifest','original_tsv'):
        path = frozen[key if key != 'original_tsv' else 'original_tsv_path']
        if sha256(path) != frozen[key+'_sha256']: raise ValueError(key+' changed')
    if sha256(frozen['dev_image_path']) != frozen['dev_image_sha256']: raise ValueError('dev image changed')
    if [str(p.resolve()) for p in sorted(HERE.glob('*.py'))] != frozen['local_python_files']:
        raise ValueError('experiment python file set changed')
    for path,digest in frozen['source_sha256'].items():
        if sha256(path) != digest: raise ValueError('frozen source changed: '+path)
    for arm, meta in frozen['checkpoints'].items():
        if sha256(meta['path']) != meta['sha256']: raise ValueError('checkpoint changed: '+arm)
    if frozen['environment']['torch'] != torch.__version__ or frozen['environment']['cuda'] != torch.version.cuda:
        raise ValueError('torch/CUDA version changed')
    import transformers
    import PIL
    if frozen['environment']['pillow'] != PIL.__version__: raise ValueError('Pillow changed')
    if frozen['environment']['transformers'] != transformers.__version__: raise ValueError('transformers changed')
    # Rechecks original experiment source, all checkpoint inventory and model freeze.
    student_eval.verify_freeze()
    return frozen


def load_runtime(frozen):
    eng = NativeEngine(common.load_model(common.BASELINE_MODEL, max_new_tokens=CAP))
    student_eval.verify_runtime_model(eng)
    device = next(eng.model.parameters()).device
    models = {}
    for arm in STUDENTS:
        raw = torch.load(frozen['checkpoints'][arm]['path'], map_location='cpu', weights_only=False)
        if raw['teacher'] != 'magnitude_v' or int(raw['seed']) != int(arm.rsplit('_',1)[1]):
            raise ValueError('student checkpoint identity mismatch')
        model = StudentScorer(**raw['config']).to(device).eval()
        model.load_state_dict(raw['model_state_dict'], strict=True)
        for p in model.parameters(): p.requires_grad_(False)
        models[arm] = model
    return eng, models


def input_key(prep):
    if not prep['has_image']: raise ValueError('missing visual input')
    # Hash actual dtype/shape/raw values + 3-D patch grid, not filenames.
    return tuple((str(t.dtype),tuple(t.shape),tensor_sha256(t)) for t in (prep['pv'],prep['gthw']))


class PageCache:
    def __init__(self): self.page = None; self.slots = {}
    def clear_to(self, page):
        if self.page != page: self.page, self.slots = page, {}
    def get(self, eng, prep, side):
        key = input_key(prep)
        if side in self.slots:
            old_key, V, DS = self.slots[side]
            if old_key != key: raise ValueError('same page/resolution has different preprocessed pixels or grid')
            return V, DS, True
        V, DS = eng.encode(prep)
        if prep['n_vis'] != (1024 if side == 1024 else K) or len(DS) != 3:
            raise ValueError('unexpected visual geometry or missing deepstack stream')
        self.slots[side] = (key,V,DS)
        return V,DS,False


@torch.inference_mode()
def predict(eng, models, msg, arm, cache=None, prepared=None):
    if arm not in ARMS: raise ValueError('unknown arm')
    side = 512 if arm == 'rres512' else 1024
    started = time.perf_counter()
    with resolution(eng, side):
        if prepared is not None:
            prep,V,DS,cache_hit = prepared
        else:
            prep = eng.prepare(msg, DATASET)
            if cache is None:
                V, DS = eng.encode(prep); cache_hit = False
            else:
                V, DS, cache_hit = cache.get(eng, prep, side)
        expected = 1024 if side == 1024 else K
        if int(prep['n_vis']) != expected or V.shape[0] != expected or len(DS) != 3 or any(d.shape != V.shape for d in DS):
            raise ValueError('expected aligned visual streams at fixed resolution')
        if arm in ('full','rres512'): keep = None
        elif arm in ('b1','b2'):
            tm, ts = eng.instruction_embeds(msg, DATASET)
            keep = eng.select(arm, K, dict(prep=prep,V=V,DS=DS,K=K,engine=eng,text_mean=tm,text_seq=ts))
        else:
            q, _ = eng.instruction_embeds(msg, DATASET)
            keep, _ = student_eval.select_student(models[arm], V, DS, q)
        result = generate_selection(eng, prep, V, DS, keep, max_tokens=CAP)
        actual = 1024 if arm == 'full' else K
        if result['actual_visual_count'] != actual or result['ds_lengths'] != [actual]*3:
            raise ValueError('actual count/DS differs from intended budget')
        result['keep_sha256'] = tensor_sha256(result.pop('keep_idx'))
        result['cache_hit'] = cache_hit
        torch.cuda.synchronize()
        result['elapsed_seconds'] = time.perf_counter() - started
        return result


def gate(manifest):
    frozen = verify(manifest)
    if GATE.exists(): raise FileExistsError('gate already exists; preserve its result')
    eng, models = load_runtime(frozen)
    item = json.loads(DEV.read_text())['items'][0]
    # Qwen's actual VQA builder, with independent development image supplied as dump.
    previous = getattr(eng.vlm,'dump_image_func',None)
    eng.vlm.set_dump_image(lambda row: item['image_path'])
    try:
        questions = [item['question'], 'What is the main heading shown on the page?']
        msgs = [eng.vlm._build_vqa_prompt({'question':q},DATASET) for q in questions]
    finally:
        if previous is None: del eng.vlm.dump_image_func
        else: eng.vlm.set_dump_image(previous)
    checks = []; cache = PageCache(); cache.clear_to(item['image_sha256'])
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); gate_start=time.perf_counter()
    try:
        # Verify the two different questions produce exactly the same pixels,
        # grid and visual streams at both resolutions.
        for side in (1024,512):
            with resolution(eng,side):
                p0=eng.prepare(msgs[0],DATASET); p1=eng.prepare(msgs[1],DATASET)
                if input_key(p0) != input_key(p1): raise AssertionError('same-page pixel/grid mismatch')
                v0,d0,hit0=cache.get(eng,p0,side)
                v1,d1=eng.encode(p1)
                if not torch.equal(v0,v1) or any(not torch.equal(a,b) for a,b in zip(d0,d1)):
                    raise AssertionError('cached/fresh visual stream mismatch')
                checks.append(dict(check='same_page_fresh_features_bitwise',side=side,passed=True))
        for qi,msg in enumerate(msgs):
            prepared = {}
            for side in (1024,512):
                with resolution(eng,side):
                    prep=eng.prepare(msg,DATASET); v,ds,hit=cache.get(eng,prep,side)
                    prepared[side]=(prep,v,ds,hit)
            for arm in ARMS:
                cached=predict(eng,models,msg,arm,prepared=prepared[512 if arm=='rres512' else 1024])
                fresh=predict(eng,models,msg,arm,None)
                fields=('text','token_ids','truncated','actual_visual_count','ds_lengths','keep_sha256','positions_sha256','prefill_length','cache_length','decode_positions')
                differences=[k for k in fields if cached[k] != fresh[k]]
                checks.append(dict(check='cached_vs_fresh_generation',question=qi,arm=arm,passed=not differences,differences=differences))
                if differences: raise AssertionError(f'gate mismatch {qi} {arm}: {differences}')
                print('gate', qi, arm, 'ok', flush=True)
        torch.cuda.synchronize()
        atomic_json(GATE,dict(pass_all=True,run_freeze_sha256=sha256(FREEZE),checks=checks,
            prompt_source='Qwen3 _build_vqa_prompt', development_questions=questions,
            wall_seconds=time.perf_counter()-gate_start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            note='Two dev questions; seven arms cached and fresh; not a whole-benchmark runtime estimate'),exclusive=True)
    except BaseException as e:
        atomic_json(GATE,dict(pass_all=False,run_freeze_sha256=sha256(FREEZE),checks=checks,
            error=repr(e),traceback=traceback.format_exc()),exclusive=True)
        raise
    print('gate passed',len(checks),flush=True)



def materialize_image(row, item):
    """Reproduce VLMEvalKit PIL decode/save without trusting shared images.

    The resolved `image` column is the original base64 even for reference
    rows. Save to this experiment's isolated directory using the source page
    extension, which preserves the default PIL encoder semantics.
    """
    from PIL import Image
    raw = base64.b64decode(row['image'], validate=True)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != item['image_sha256']: raise ValueError('benchmark source image bytes changed')
    suffix=Path(item['image_path']).suffix.lower()
    formats=Image.registered_extensions()
    if suffix not in formats: raise ValueError('unsupported benchmark image extension')
    image=Image.open(io.BytesIO(raw))
    if image.mode in ('RGBA','P','LA'): image=image.convert('RGB')
    buffer=io.BytesIO(); image.save(buffer,format=formats[suffix]); data=buffer.getvalue()
    processed=hashlib.sha256(data).hexdigest()
    path=OUT/'images'/f"{digest}{suffix}";path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if sha256(path)!=processed: raise ValueError('isolated materialized image hash mismatch')
    else:
        tmp=path.with_name(path.name+'.tmp')
        tmp.write_bytes(data);os.replace(tmp,path)
    return path,processed

def connect_db():
    OUT.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(DB); db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    db.execute('''CREATE TABLE IF NOT EXISTS results (
        row_index INTEGER NOT NULL, arm TEXT NOT NULL, status TEXT NOT NULL,
        prediction TEXT, gen_ids TEXT, truncated INTEGER, n_vis INTEGER,
        positions_sha256 TEXT, keep_sha256 TEXT, durations_json TEXT, error TEXT,
        PRIMARY KEY(row_index,arm))''')
    db.execute('CREATE TABLE IF NOT EXISTS sessions (started_unix REAL PRIMARY KEY, finished_unix REAL, wall_seconds REAL, new_success INTEGER)')
    db.execute('CREATE TABLE IF NOT EXISTS page_inputs (source_sha256 TEXT PRIMARY KEY, materialized_sha256 TEXT NOT NULL, path TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS error_events (row_index INTEGER, arm TEXT, created_unix REAL, error TEXT)')
    digest=sha256(FREEZE)
    old=db.execute("SELECT value FROM meta WHERE key='run_freeze_sha256'").fetchone()
    if old and old[0]!=digest: raise ValueError('database belongs to a different run freeze')
    db.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('run_freeze_sha256',?)",(digest,));db.commit()
    return db


def run(manifest):
    frozen=verify(manifest)
    gate_data=json.loads(GATE.read_text())
    if not gate_data.get('pass_all') or gate_data.get('run_freeze_sha256') != sha256(FREEZE):
        raise ValueError('successful matching development gate required')
    manifest_data=validate_manifest(manifest)
    groups=OrderedDict()
    for row in manifest_data['records']:
        groups.setdefault((row['image_path'],row['image_sha256']),[]).append(row)
    eng,models=load_runtime(frozen)
    dataset=common.build_dataset(DATASET)
    if len(dataset.data)!=5349: raise ValueError('runtime dataset size mismatch')
    eng.vlm.set_dump_image(dataset.dump_image)
    db=connect_db(); cache=PageCache(); start=time.perf_counter(); completed=0
    started_unix=time.time(); db.execute('INSERT INTO sessions VALUES(?,?,?,?)',(started_unix,None,None,0));db.commit()
    for page,records in groups.items():
        cache.clear_to(page)
        image_path=None
        for item in records:
            idx=int(item['source_row'])
            done={r[0] for r in db.execute("SELECT arm FROM results WHERE row_index=? AND status='success'",(idx,))}
            if all(a in done for a in ARMS): continue
            try:
                row=dataset.data.iloc[idx]
                if str(row['index']) != str(item['index']) or str(row['question']) != str(item['question']):
                    raise ValueError('dataset index/question differs from manifest')
                if image_path is None:
                    image_path,image_digest=materialize_image(row,item)
                    old=db.execute('SELECT materialized_sha256 FROM page_inputs WHERE source_sha256=?',(item['image_sha256'],)).fetchone()
                    if old and old[0]!=image_digest: raise ValueError('materialized image hash changed on resume')
                    db.execute('INSERT OR IGNORE INTO page_inputs VALUES(?,?,?)',(item['image_sha256'],image_digest,str(image_path)));db.commit()
                eng.vlm.set_dump_image(lambda _row: str(image_path))
                msg=common.build_message(eng.vlm,dataset,DATASET,row)
                prepared={}
                needed={512 if arm=='rres512' else 1024 for arm in ARMS if arm not in done}
                for side in needed:
                    with resolution(eng,side):
                        prep=eng.prepare(msg,DATASET); v,ds,hit=cache.get(eng,prep,side)
                        prepared[side]=(prep,v,ds,hit)
            except BaseException:
                err=traceback.format_exc()
                db.execute('INSERT INTO error_events VALUES(?,?,?,?)',(idx,'__prepare__',time.time(),err))
                db.execute('UPDATE sessions SET finished_unix=?,wall_seconds=?,new_success=? WHERE started_unix=?',(time.time(),time.perf_counter()-start,completed,started_unix));db.commit()
                raise
            for arm in ARMS:
                if arm in done: continue
                try:
                    output=predict(eng,models,msg,arm,prepared=prepared[512 if arm=='rres512' else 1024])
                    vals=(idx,arm,'success',output['text'],json.dumps(output['token_ids']),
                        int(output['truncated']),int(output['actual_visual_count']),output['positions_sha256'],
                        output['keep_sha256'],json.dumps(dict(predict_elapsed_seconds=output['elapsed_seconds'],cache_hit=output['cache_hit'], timing_scope='selection plus prefill/decode only; excludes shared image materialization, preprocess and vision encode')),None)
                    db.execute('INSERT OR REPLACE INTO results VALUES(?,?,?,?,?,?,?,?,?,?,?)',vals); db.commit()
                    completed+=1
                    if completed % 20 == 0: print('new generations',completed,'row',idx,'arm',arm,'elapsed',round(time.perf_counter()-start,1),flush=True)
                except BaseException:
                    err=traceback.format_exc()
                    db.execute('INSERT OR REPLACE INTO results VALUES(?,?,?,?,?,?,?,?,?,?,?)',(idx,arm,'error',None,None,None,None,None,None,None,err))
                    db.execute('INSERT INTO error_events VALUES(?,?,?,?)',(idx,arm,time.time(),err))
                    db.execute('UPDATE sessions SET finished_unix=?,wall_seconds=?,new_success=? WHERE started_unix=?',(time.time(),time.perf_counter()-start,completed,started_unix));db.commit()
                    raise
    db.execute('UPDATE sessions SET finished_unix=?,wall_seconds=?,new_success=? WHERE started_unix=?',
        (time.time(),time.perf_counter()-start,completed,started_unix));db.commit()
    counts=dict(db.execute('SELECT status, count(*) FROM results GROUP BY status').fetchall())
    print('complete',counts,'expected',5349*len(ARMS),'new elapsed',time.perf_counter()-start,flush=True)
    if counts.get('success',0)!=5349*len(ARMS) or counts.get('error',0): raise RuntimeError('incomplete evaluation')
    db.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','gate','run'))
    p.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST); args=p.parse_args()
    {'freeze':freeze,'gate':gate,'run':run}[args.command](args.manifest)

if __name__ == '__main__': main()
