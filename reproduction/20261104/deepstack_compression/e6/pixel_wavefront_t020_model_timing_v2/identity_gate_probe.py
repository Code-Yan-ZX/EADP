"""Check actual completed identity gates; CPU only, no GPU authority."""
import json,os
from pathlib import Path
import admission
D=Path(__file__).resolve().parent
bundle=admission.require_identity();out=D/'identity_gate_probe001.json'
with out.open('x') as f:json.dump({'status':'passed','identity':bundle,'GPU_authorized':False,'model_loaded':False,'GT_opened':False,'scoring_performed':False},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
