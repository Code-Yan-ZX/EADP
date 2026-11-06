"""Load actual controller and sampler module graphs in separate CPU processes."""
import argparse,os,runpy
from pathlib import Path
D=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=('controller','sampler'));a=p.parse_args()
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('probe requires hidden GPU')
runpy.run_path(str(D/(a.mode+'.py')),run_name='_modelmemory_import_probe')
print('import_probe_passed',a.mode)
