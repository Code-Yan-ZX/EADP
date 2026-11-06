"""CPU-only rehash and freeze of all 96 already-accepted selector archives."""
import json,os
from pathlib import Path
import inventory

HERE=Path(__file__).resolve().parent
def main():
    rows=inventory.inventory(hash_files=True);prior=inventory.require_scatter_review()
    result={'status':'passed','scope':'fixed 32-image x 3-budget archive identities; no GPU work',
            'cases':rows,'case_count':len(rows),'inventory_sha256':inventory.sha(HERE/'inventory.py'),
            'freezer_sha256':inventory.sha(__file__),'scatter_review_sha256':prior['independent_review_sha256'],
            'GT_opened':False,'predictions_opened':False,'scoring_performed':False,'gpu_used':False}
    with (HERE/'archives002.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
