#!/usr/bin/env python3
"""DocVQA VAL scoring: official VLMEvalKit ANLS and paired parent bootstrap.

This program never chooses an arm, seed, checkpoint, or subset by outcome.
Partial results are explicitly marked and use paired complete rows only.
"""
from __future__ import annotations
import argparse
import csv
import ast
import hashlib
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
QWEN = HERE.parents[1]
ROOT = QWEN.parent
sys.path.insert(0, str(QWEN / 'VLMEvalKit'))
from vlmeval.dataset.utils.vqa_eval import hit_calculate, process_line

OUT = QWEN / 'outputs/student_full'
TSV = Path('/media/disk2/YZX/LMUData/DocVQA_VAL.tsv')
ARMS = ('full', 'b2', 'b1', 'rres512', 'magnitude_v__seed_11',
        'magnitude_v__seed_23', 'magnitude_v__seed_37')
STUDENTS = ARMS[4:]
REFS = ARMS[:4]
DRAWS = 10000
SEED = 20261109


def anls(answer_field, prediction):
    """Official process_line computes distances, hit_calculate converts to ANLS."""
    line = {'answer': answer_field, 'prediction': prediction}
    return float(hit_calculate([process_line(line, method='anls')], 'DocVQA_VAL')[0])


def literal(answer_field, prediction):
    try:
        aliases = ast.literal_eval(answer_field)
    except (SyntaxError, ValueError):
        aliases = answer_field
    aliases = aliases if isinstance(aliases, list) else [aliases]
    return float(any(prediction == str(a) for a in aliases))


def paired_bootstrap(values_a, values_b, groups, draws=DRAWS, seed=SEED, batch=256):
    """Question-weighted paired bootstrap, sampling source parents jointly.

    values_{a,b} have equal length; one shared group index preserves the
    pairing, including the varying number of questions per sampled parent.
    """
    a, b = np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(groups) != len(a) or len(a) == 0:
        raise ValueError('nonempty equally sized paired data required')
    names, inv = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    n = len(names)
    counts = np.bincount(inv, minlength=n).astype(float)
    sums = np.bincount(inv, weights=b - a, minlength=n)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for start in range(0, draws, batch):
        end = min(start + batch, draws)
        idx = rng.integers(0, n, size=(end - start, n))
        samples[start:end] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return {'delta': float((b-a).mean()),
            'ci95': [float(x) for x in np.percentile(samples, [2.5, 97.5])],
            'n_questions': len(a), 'n_parents': n,
            'improved': int(np.sum(b > a)), 'worsened': int(np.sum(b < a)),
            'tied': int(np.sum(b == a))}


def self_check():
    # ANLS = max similarity, similarity below .5 becomes zero; exactly .5 passes.
    cases = [(["cat"], 'cats', .75), (["cat"], 'dog', 0.),
             (["ab"], 'ac', .5), (["abc"], 'axy', 0.),
             (["cat", "dog"], 'DOG', 1.), (["Mixed Case"], 'mixed case', 1.)]
    for aliases, pred, expected in cases:
        actual = anls(repr(aliases), pred)
        assert abs(actual - expected) < 1e-12, (aliases, pred, actual, expected)
        # Independently call the repository's exact official evaluator pipeline.
        official = process_line({'answer': repr(aliases), 'prediction': pred}, method='anls')
        assert actual == hit_calculate([official], 'DocVQA_VAL')[0]
    assert literal(repr(['A', 'b']), 'A') == 1
    assert literal(repr(['A', 'b']), 'a') == 0
    assert literal(repr([' A ']), 'A') == 0
    assert literal(repr(['A']), ' A ') == 0
    # A and B differ by a constant on each question: shared resampling is exact,
    # while independent resampling would spuriously create uncertainty.
    a = np.array([.1, .4, .9, .8]); b = a + .1
    p = paired_bootstrap(a,b,['p','p','q','r'],draws=1000,seed=5)
    assert p['n_questions'] == 4 and p['n_parents'] == 3
    assert np.allclose(p['ci95'], [.1,.1]) and p['improved'] == 4
    # Varied per-parent group sizes: compare shared-parent indices and
    # question-weighted denominators directly to a vectorized reference.
    a = np.array([0.,0.,0.,0.]); b = np.array([1.,1.,0.,0.]); g=['p','p','q','r']
    p = paired_bootstrap(a,b,g,draws=1000,seed=5,batch=31)
    rng=np.random.default_rng(5); ix=rng.integers(0,3,size=(1000,3))
    ref=np.array([2.,0.,0.])[ix].sum(1)/np.array([2.,1.,1.])[ix].sum(1)
    assert np.allclose(p['ci95'],np.percentile(ref,[2.5,97.5]))
    print('self-check: official ANLS, threshold, aliases, case, literal and shared-parent bootstrap passed')


def load_manifest(path, tsv):
    raw = json.loads(path.read_text())
    digest = hashlib.sha256()
    with tsv.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != raw['provenance']['source_tsv_sha256']:
        raise ValueError('source TSV hash differs from frozen manifest')
    rows = raw['records']
    if not raw.get('complete') or len(rows) != 5349:
        raise ValueError('manifest missing complete 5349 records')
    indices = [int(r['source_row']) for r in rows]
    if sorted(indices) != list(range(5349)):
        raise ValueError('manifest source rows are not precisely 0..5348')
    by_index = {int(r['source_row']):r for r in rows}
    if len({str(r['index']) for r in rows}) != 5349:
        raise ValueError('manifest question IDs not unique')
    csv.field_size_limit(sys.maxsize)
    result=[]
    with tsv.open(newline='', encoding='utf-8') as f:
        for i, row in enumerate(csv.DictReader(f, delimiter='\t')):
            if i not in by_index:
                raise ValueError('unexpected source TSV row')
            rec=by_index[i]
            if str(row['index']) != str(rec['index']) or str(row['ucsf_document_id']) != str(rec['parent_id']):
                raise ValueError(f'source mismatch at row {i}')
            if ast.literal_eval(row['answer']) != rec['answers'] or row['question'] != rec['question']:
                raise ValueError(f'source question/answers mismatch at row {i}')
            result.append({'row_index':i, 'index':str(row['index']),
                           'parent_id':str(row['ucsf_document_id']), 'answer':row['answer']})
    if len(result) != 5349 or len({r['parent_id'] for r in result}) != 938:
        raise ValueError('source TSV expected 5349 questions / 938 parents')
    return result


def read_database(path):
    uri='file:' + str(path.resolve()) + '?mode=ro'
    con=sqlite3.connect(uri, uri=True)
    try:
        con.execute('BEGIN')  # one consistent snapshot while inference writes
        rows=con.execute('SELECT row_index, arm, status, prediction, truncated, n_vis, error FROM results').fetchall()
        con.rollback()
        return rows
    finally:
        con.close()


def analyze(records, dbrows):
    lookup={r['row_index']:r for r in records}
    decoded={arm:{} for arm in ARMS}
    counts={arm:Counter() for arm in ARMS}
    errors={arm:[] for arm in ARMS}
    observed=set()
    for row_index, arm, status, pred, truncated, n_vis, error in dbrows:
        if arm not in decoded or row_index not in lookup:
            raise ValueError(f'unexpected result {row_index} {arm}')
        key=(row_index,arm)
        if key in observed: raise ValueError(f'duplicate result {key}')
        observed.add(key)
        counts[arm]['seen']+=1
        if status != 'success':
            counts[arm]['error' if status=='error' else 'unexpected_status']+=1
            errors[arm].append({'row_index':row_index,'status':status,'error':error})
            continue
        expected = 1024 if arm == 'full' else 256
        if not isinstance(pred,str) or n_vis != expected or truncated not in (0,1):
            counts[arm]['invalid_success']+=1
            errors[arm].append({'row_index':row_index,'status':'invalid_success',
                                'n_vis':n_vis,'truncated':truncated})
            continue
        rec=lookup[row_index]
        score=anls(rec['answer'],pred)
        decoded[arm][row_index] = {'anls':score, 'literal_exact':literal(rec['answer'],pred),
                                    'truncated':bool(truncated), 'prediction':pred,
                                    'parent_id':rec['parent_id'], 'index':rec['index']}
        counts[arm]['success']+=1
        counts[arm]['truncated']+=bool(truncated)
    arms={}
    for arm in ARMS:
        d=decoded[arm]; n=len(d)
        arms[arm]={'seen':counts[arm]['seen'], 'success':n, 'pending':5349-counts[arm]['seen'],
                   'errors':counts[arm]['error'], 'unexpected_status':counts[arm]['unexpected_status'],
                   'invalid_success':counts[arm]['invalid_success'], 'truncated':counts[arm]['truncated'],
                   'anls':float(np.mean([v['anls'] for v in d.values()])) if n else None,
                   'literal_exact':float(np.mean([v['literal_exact'] for v in d.values()])) if n else None,
                   'error_details':errors[arm]}
    pairs={}
    comparisons=[(ref,student) for student in STUDENTS for ref in REFS] + [('b1','b2')]
    for ref, student in comparisons:
            ids=sorted(set(decoded[student]) & set(decoded[ref]))
            if not ids:
                pairs[f'{student} - {ref}']=None
                continue
            a=[decoded[ref][i]['anls'] for i in ids]
            b=[decoded[student][i]['anls'] for i in ids]
            groups=[lookup[i]['parent_id'] for i in ids]
            pairs[f'{student} - {ref}']=paired_bootstrap(a,b,groups)
    question_scores=[]
    for rec in records:
        i=rec['row_index']
        scores={arm:decoded[arm][i]['anls'] for arm in ARMS if i in decoded[arm]}
        changes={f'{target} - {ref}':scores[target]-scores[ref]
                 for ref,target in comparisons if ref in scores and target in scores}
        question_scores.append({'row_index':i,'index':rec['index'],'parent_id':rec['parent_id'],
                                'anls':scores,'paired_anls_changes':changes})
    complete=all(arms[a]['success']==5349 and arms[a]['seen']==5349 for a in ARMS)
    student_means=[arms[a]['anls'] for a in STUDENTS]
    seed_summary = (dict(mean=float(np.mean(student_means)), range=[min(student_means),max(student_means)])
                    if complete else None)
    return {'complete':complete, 'state':'complete' if complete else 'partial',
            'expected_questions':5349,'expected_parents':938,'expected_results':5349*len(ARMS),
            'metric':'VLMEvalKit process_line(anls) then hit_calculate(DocVQA_VAL)',
            'scale':'all ANLS, deltas and intervals are on 0-to-1 scale; multiply by 100 for percentage points',
            'literal_exact_definition':'case-sensitive equality to any original alias, without stripping',
            'bootstrap':{'unit':'ucsf_document_id parent, paired; question-weighted',
                         'draws':DRAWS,'seed':SEED,'interval':'percentile 95%',
                         'improved_worsened':'strictly higher/lower ANLS per paired question'},
            'arms':arms,'magnitude_v_seeds_mean_and_range':seed_summary,
            'paired_changes':pairs,'question_scores':question_scores}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--self-check',action='store_true')
    p.add_argument('--partial',action='store_true',help='allow writing explicitly partial analysis')
    p.add_argument('--manifest',type=Path,default=OUT/'docvqa_val_manifest.json')
    p.add_argument('--tsv',type=Path,default=TSV)
    p.add_argument('--database',type=Path,default=OUT/'predictions.sqlite3')
    p.add_argument('--output',type=Path,default=OUT/'analysis.json')
    args=p.parse_args()
    if args.self_check:
        self_check(); return
    records=load_manifest(args.manifest,args.tsv)
    analysis=analyze(records,read_database(args.database))
    if not analysis['complete'] and not args.partial:
        raise SystemExit('results incomplete; use --partial for explicitly partial analysis')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=args.output.with_suffix(args.output.suffix+'.tmp')
    tmp.write_text(json.dumps(analysis,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(args.output)
    print(json.dumps({'complete':analysis['complete'], 'state':analysis['state'],
                      'counts':{k:v['success'] for k,v in analysis['arms'].items()},
                      'output':str(args.output)},ensure_ascii=False))


if __name__ == '__main__':
    main()
