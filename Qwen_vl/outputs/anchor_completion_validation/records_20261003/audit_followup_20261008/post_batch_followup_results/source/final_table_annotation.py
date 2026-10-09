"""CPU final presentation annotation; frozen render/build/generation sources stay intact."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLE = HERE.parents[1] / 'submission_tables_20261009'
PLAN = json.loads((HERE / 'followup.plan.json').read_text())
RENDER = TABLE / 'render_tables.py'
assert hashlib.sha256(RENDER.read_bytes()).hexdigest() == PLAN['source_and_input_sha256'][str(RENDER)]
data = json.loads((TABLE / 'latest_tables.json').read_text())
before_rows = json.dumps(data['rows'], sort_keys=True)
note = ('Local VizWiz primary scores use official leave-one-annotator-out accuracy. '
        'K128 (NeXT nominal640) uses the predeclared alpha0/beta1 public-default pair, '
        'not verified paper argv; other budgets retain their recorded recipes. '
        'The released min(matches/3) formula is auxiliary only.')
if note not in data['notes']:
    data['notes'].append(note)
data['sources_sha256'][str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
assert json.dumps(data['rows'], sort_keys=True) == before_rows
(TABLE / 'latest_tables.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
for name in ['latest_tables.html', 'latest_tables.tex']:
    path = TABLE / name
    text = path.read_text()
    anchor = 'Completed VizWiz K128 followup: alpha0/beta1 public defaults, not proven paper argv.'
    if note not in text:
        assert anchor in text, name
        text = text.replace(anchor, anchor + ' ' + note)
        path.write_text(text)
text = RENDER.read_text()
anchor = "        fig.text(.025,.12/height,'\\n'.join(footer)"
assert text.count(anchor) == 1
insert = """        if model != 'qwen':
            footer = [
                '— Missing or unresolved protocol. * VizWiz validation; local primary scores use official leave-one-out accuracy.',
                'VizWiz K128 (NeXT nominal640): alpha0/beta1 public defaults for EADP/ours; not verified paper argv.',
                'Other budgets retain their recorded recipes. Released min(matches/3) scores are auxiliary only.',
                'SQA: shipped CQM-A, 2017 image questions. POPE: mean F1 (2910/3000/3000). NeXT budgets are nominal.',
                'Avg7 = (GQA + SQA-IMG + TextVQA + POPE-F1 + MME/20 + MMB-EN + MMB-CN)/7; MME is perception raw.',
                'Local Avg7 stays blank for missing/unresolved MMB. Reported baselines are references, not local reproductions.'
            ]
"""
text = text.replace(anchor, insert + anchor)
exec(compile(text, str(RENDER) + '[final-annotation]', 'exec'), {'__name__': '__main__', '__file__': str(RENDER)})
assert json.dumps(json.loads((TABLE / 'latest_tables.json').read_text())['rows'], sort_keys=True) == before_rows
print('Final annotation complete; all numerical cells unchanged; frozen sources unchanged.')
