import copy,unittest
from full_contract import split_schedule,COMPONENTS
from test_full_contract import payload
from full_verify import group_events,check_event_identity
from pilot_verify import check_events
class Tests(unittest.TestCase):
 def test_32_images_1920_complete_and_missing_arm(self):
  schedules=split_schedule(payload());events=[];fps={c:c for c in COMPONENTS}
  for i,s in enumerate(schedules):
   ev=[{'status':'image_started','image_index':i},{'status':'capture_generated','image_index':i},{'status':'capture_cleanup','passed':True,'capture_completed':True,'image_index':i},{'status':'archive_written','image_index':i},{'status':'capture_identity_passed','identity':{'status':'passed'},'image_index':i},{'status':'reference_fingerprints','sha256':fps,'image_index':i}];seen=set()
   for j in s['jobs']:
    c=j['component']
    if c not in seen:
     ev.extend({'status':'warmup_validated','component':c,'index':n,'full_output_equal':True,'output_sha256':fps[c],'image_index':i} for n in range(3));seen.add(c)
    ev.append({'status':'measured','job':j,'full_output_equal':True,'output_sha256':fps[c],'gpu_event_ms':1.,'host_wall_ms':1.,'scope':'excludes decoder DS addition','image_index':i})
   ev.append({'status':'image_completed','image_index':i});events.extend(ev)
  chunks=group_events(events)
  for i,chunk in enumerate(chunks):
   row={'task':'D','qid':str(i),'group_id':'g'+str(i)}
   for e in chunk:
    if e['status'] in ('capture_generated','capture_identity_passed'):e.update(row)
   check_event_identity(chunk,row)
  wrong=copy.deepcopy(chunks[8]);next(e for e in wrong if e['status']=='capture_generated')['qid']='other'
  with self.assertRaises(ValueError):check_event_identity(wrong,{'task':'D','qid':'8','group_id':'g8'})
  for s,chunk in zip(schedules,chunks):check_events(chunk[1:-1],s,fps)
  broken=copy.deepcopy(chunks[12][1:-1]);broken.pop(next(i for i,e in enumerate(broken) if e['status']=='measured'))
  with self.assertRaises(ValueError):check_events(broken,schedules[12],fps)
if __name__=='__main__':unittest.main()
