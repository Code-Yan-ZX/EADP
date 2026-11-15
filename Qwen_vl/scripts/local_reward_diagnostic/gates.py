"""Real frozen-backbone gate on a fixed page outside the primary panel."""
import json
from batch_runner import batch_gate
from layoutlite_runner import LayoutLiteRunner, diagnostic_masks
from run import ROOT, OUT, SEEDS, atomic_json

def main():
    source=json.loads((ROOT/'Qwen_vl/outputs/binding_oracle/data/combined_manifest.json').read_text())
    image=source['records'][0]['source_image']
    runner=LayoutLiteRunner()
    page=runner.encode(image)
    masks=diagnostic_masks(page['scores'],seeds=SEEDS)
    result=batch_gate(runner,page,[m['keep_idx'] for m in masks])
    result['image']=image
    atomic_json(OUT/'batch_gates.json',result)
    print(json.dumps(result,indent=2))
    assert result['pass_all'],'Real native/batching token identity gate failed'

if __name__=='__main__':main()
