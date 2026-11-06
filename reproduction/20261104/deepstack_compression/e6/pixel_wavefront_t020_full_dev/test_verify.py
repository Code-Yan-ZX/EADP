import copy
import unittest
from verify import validate_events

class JournalTests(unittest.TestCase):
    def test_complete_and_failed_or_missing(self):
        anchors=[{'task':'T','qid':str(i)} for i in range(3)]
        rows=[{'generated_count':2} for _ in anchors];events=[]
        for a in anchors:
            events.extend([{'event':'arm_started',**a},{'event':'forward_complete','index':0,**a},
                           {'event':'forward_complete','index':1,**a},{'event':'arm_recorded','status':'completed',**a}])
        validate_events(events,anchors,rows)
        failed=copy.deepcopy(events);failed[3]['status']='failed'
        with self.assertRaises(ValueError):validate_events(failed,anchors,rows)
        with self.assertRaises(ValueError):validate_events(events[:-1],anchors,rows)
        bad=copy.deepcopy(events);bad[2]['index']=0
        with self.assertRaises(ValueError):validate_events(bad,anchors,rows)

if __name__=='__main__':unittest.main()
