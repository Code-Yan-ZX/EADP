import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import score_oracle,paired
class NativeTests(unittest.TestCase):
    def test_spreadsheet_preserves_denominator_and_empty_rule(self):
        rows=[]
        for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
            for i,text in enumerate(('', 'yes')):
                rows.append({'task':task,'qid':str(i),'arms':{arm:{'answer':text} for arm in paired.ARMS}})
        native=score_oracle.native_predictions(rows)
        self.assertEqual(len(native),3*2*7)
        for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
            self.assertEqual(native[(task,'0','EADP')],'nan')
            self.assertEqual(native[(task,'1','EADP')],'yes')
if __name__=='__main__':unittest.main()
