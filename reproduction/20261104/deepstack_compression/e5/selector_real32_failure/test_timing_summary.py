import unittest
from timing_summary import summarize

def panel(failed=False):
    events=[{'status':'graph_setup','budget':128,'result':{'status':'passed'}}]
    for i in range(2):
        ok=not(failed and i==1)
        ms=[{'implementation':m,'repeat':j,'status':'measured' if m=='author' or ok else 'unavailable','equals_author':True,'gpu_event_ms':float(j+1+i)} for m in ('author','fused','graph') for j in range(5)]
        events.append({'status':'case_completed','image_index':i,'budget':128,'result':{'status':'measured','candidate_matches_author':{'fused':ok,'graph':ok},'measurements':ms}})
    return events
class TestSummary(unittest.TestCase):
    def test_paired_exclusion_and_median(self):
        r=summarize(panel(True),2,(128,))['128']
        self.assertEqual(r['all_available']['author']['median_ms'],3.5)
        self.assertEqual(r['paired']['author']['median_ms'],3)
        self.assertEqual(r['all_available']['graph']['n'],1)
        self.assertEqual(r['excluded_image_indices'],[1])
    def test_duplicate_and_unverified_rejected(self):
        e=panel();e.append(e[-1]);self.assertRaises(ValueError,summarize,e,2,(128,))
        e=panel(True);e[-1]['result']['measurements'][5]['status']='measured';self.assertRaises(ValueError,summarize,e,2,(128,))
if __name__=='__main__':unittest.main()
