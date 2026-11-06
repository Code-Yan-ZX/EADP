"""Fixed full-panel and previously reviewed transitive source inventory."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
LOCAL=('contract.py','plan.json','budget_plan.json','full_imports.py','source_inventory.py','panel_gate.py','panel_worker.py','panel_runner.py','panel_launch.py','panel_verify.py','validate_panel_record.py','preflight.py','cpu_checks.py','probe_smoke.py','test_contract.py','test_panel_gate.py','test_panel_worker.py','test_panel_runner.py','test_panel_launch.py','test_validate_panel_record.py','test_preflight_lifecycle.py')
def sources():
 proof=json.loads((BASE/'e2_e3_smoke/cpu_check_011/verification.json').read_text())
 if proof.get('status')!='passed' or proof.get('source_sha256_before')!=proof.get('source_sha256_after'):raise ValueError('smoke CPU source proof invalid')
 return sorted({*(str((HERE/name).resolve()) for name in LOCAL),*proof['source_sha256_after']})
