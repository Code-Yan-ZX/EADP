"""Publish the scorer-only revision through the frozen isolated Git workflow.

The existing lock, remote-head checks, detached checkout, explicit file selection
and live-index preservation stay in force. No generation or score selection runs.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FROZEN = HERE/'followup_publisher.py'
assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == '0248eb2a9e7e0b2cf8754ff90226f439ac9f11434f3aa1df5719cb580702b6ca'
spec = importlib.util.spec_from_file_location('author_confirmation_frozen_publisher', FROZEN)
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)
original_metadata = publisher.metadata_snapshot

def metadata_snapshot(plan, state):
    _, files, history = original_metadata(plan, state)
    table = publisher.BASE/'submission_tables_20261009'
    snapshot = json.loads((table/'latest_tables.json').read_text())
    assert snapshot['vizwiz_scoring_policy']['primary_metric'] == 'EADP released VizWiz accuracy (%)'
    assert 'selected_recipe' in snapshot['vizwiz_scoring_policy']
    for path in sorted(table.glob('table_*.png')) + [table/'core_tables.txt', table/'vizwiz_author_confirmed.csv', table/'vizwiz_prescribed_repair_controls.csv']:
        files[str(publisher.pub.DEST/'tables'/path.name)] = path.read_bytes()
    note = ('Post-batch K128 controls and author-confirmed VizWiz scorer revision.\n'
            'Original 75 arms/55 groups plus 5 followup arms/3 groups are complete; their predictions and official LOO score files remain unchanged.\n'
            'On 2026-10-10 the user reported author email confirmation of the EADP released VizWiz evaluator, then explicitly selected the original recipe yielding NeXT AnchorZip 60.72.\n'
            'The main table uses the complete original alpha0.5/beta2/full-guidance/max1024 AnchorZip prediction block for both LLaVA models and all three budgets, before stream-wait-repair generation. All six cells use the EADP normalization and mean min(matches/3,1); official LOO stays auxiliary. Historical prediction runtime metadata is empty; parameter evidence comes from the existing driver/source audit.\n'
            'Shared alpha0/beta1 public-default EADP/AnchorZip K128 controls and repaired legacy-budget scores are retained separately. Their EADP values do not fill the missing same-recipe main-table VizWiz baselines. No per-cell maximum selection is performed; cells lower than controls are also retained. Author confirmation concerns the scoring formula, not full paper generation argv.\n'
            'All non-VizWiz numerical cells and averages are unchanged. The frozen refresh_tables.py/final_table_annotation.py document the original official-primary presentation; apply_author_confirmed_vizwiz.py documents the initial scorer-only revision, superseded for main prediction selection by select_vizwiz_recipe_6072.py.\n'
            'See source/selected_vizwiz_recipe_6072_20261010.json for the current user-selected policy, full per-question scores and source hashes. source/author_confirmed_vizwiz_scores_20261010.json preserves the supplementary complete controls and recalculated paired statistics.\n\n'
            + json.dumps(history, indent=2))
    files[str(publisher.pub.DEST/'README.md')] = note.encode()
    sources = {name: publisher.pub.digest(data) for name, data in files.items()}
    return publisher.pub.digest(json.dumps(sources, sort_keys=True).encode()), files, history

publisher.pub.metadata_snapshot = metadata_snapshot
if __name__ == '__main__':
    publisher.run(once=True)
    state = publisher.pub.read(publisher.pub.STATE)
    if state.get('error') or state.get('pending_commit') or state.get('status') != 'complete':
        raise RuntimeError('Publication did not complete: '+json.dumps(state))
    print(json.dumps({k:state.get(k) for k in ('status','remote_head','no_op','live_index_preserved','pending_commit')}, indent=2))
