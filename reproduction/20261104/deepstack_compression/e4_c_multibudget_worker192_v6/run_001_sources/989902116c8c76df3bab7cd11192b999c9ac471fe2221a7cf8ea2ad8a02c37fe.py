"""Run the complete candidate and execution synthetic CPU contracts."""
import subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
for path in (BASE/'e4_c_budget_pilot_v2',HERE):
 result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(path),'-p','test_*.py','-v'],check=False)
 if result.returncode:raise SystemExit(result.returncode)
