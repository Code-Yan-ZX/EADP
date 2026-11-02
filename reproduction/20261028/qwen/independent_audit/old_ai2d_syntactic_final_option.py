"""One fixed syntax-only diagnostic on the preserved OLD60 unresolved answers."""
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
PLAN = HERE / 'old_ai2d_final_option_plan.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    plan = json.loads(PLAN.read_text())
    source = Path(plan['source_bound_report']['path'])
    assert sha(source) == plan['source_bound_report']['sha256']
    original = json.loads(source.read_text())
    rows = original['unresolved_answers']
    assert len(rows) == 60
    pattern = re.compile(plan['pattern'], re.IGNORECASE)
    matches = []
    for row in rows:
        match = pattern.search(row['raw_prediction'])
        if match is None:
            continue
        option = match.group(1).upper()
        if option not in row['valid_option_letters']:
            continue
        matches.append(dict(index=row['index'], captured_option=option, GT=row['GT'],
                            GT_agrees=option == row['GT'], matched_text=match.group(0),
                            original_can_infer=row['inferred']))
    result = dict(scope='OLD60 unresolved only; syntax diagnostic, not a judge or new main metric',
        source_bound_report=plan['source_bound_report'], old_raw=original['raw'],
        predeclared_plan_sha256=sha(PLAN), script_sha256=sha(__file__),
        pattern=plan['pattern'], case_handling=plan['case_handling'],
        unresolved_count=60, matched_count=len(matches),
        matched_GT_agreement_count=sum(r['GT_agrees'] for r in matches),
        matched_GT_disagreement_count=sum(not r['GT_agrees'] for r in matches),
        unmatched_count=60-len(matches), matches=matches,
        main_score_changed=False, API_or_GPU_used=False,
        limits=['Single strict last-line final/answer-label syntax, fixed before inspecting the matching counts; no regex revisions based on the result.',
                'No semantic interpretation of unmatched text, and no claim to reproduce the author judge.',
                'The main2302/3088 score and its complete denominator remain unchanged.'])
    (HERE / 'old_ai2d_final_option_syntax.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['unresolved_count','matched_count','matched_GT_agreement_count','matched_GT_disagreement_count','unmatched_count']}))


if __name__ == '__main__':
    main()
