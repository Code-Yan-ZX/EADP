"""Annotate all latency samples that overlap conservative monitoring windows."""

def annotate(records,snapshots,time_namespace):
 if not isinstance(time_namespace,str) or not time_namespace:raise ValueError('missing time namespace')
 if not snapshots:raise ValueError('missing monitoring snapshots')
 windows=[]
 for snapshot in snapshots:
  pair=snapshot.get('health_snapshot_window_monotonic_ns')
  if (snapshot.get('time_namespace')!=time_namespace or not isinstance(pair,list) or len(pair)!=2 or
      any(type(x) is not int or x<0 for x in pair) or pair[0]>pair[1]):raise ValueError('invalid or incompatible monitoring clock')
  windows.append(pair)
 result=[]
 for record in records:
  start=record.get('t0_before_image_read_ns');end=record.get('generation_complete_ns')
  if type(start) is not int or type(end) is not int or start<0 or start>end:raise ValueError('invalid latency window')
  overlap=[i for i,(a,b) in enumerate(windows) if a<=end and start<=b]
  result.append(dict(record,monitor_overlap=bool(overlap),monitor_snapshot_indices=overlap))
 return {'records':result,'n_records':len(records),'n_monitor_overlaps':sum(r['monitor_overlap'] for r in result),
         'analysis_scope':'all samples retained; health windows conservatively bracket GPU snapshots'}


def annotate_events(events,snapshots,time_namespace,started,terminal):
 """Require a complete monitored interval; preserve every event unchanged otherwise.

 The caller must independently bind the actual started, terminal and complete
 health journal by SHA. Windows bracket health calls only, so nearby residual
 interference is not ruled out.
 """
 if (terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[] or
     type(started.get('worker_pid')) is not int or started['worker_pid']!=terminal.get('worker_pid')):
  raise ValueError('monitor did not complete cleanly')
 interval=started.get('snapshot_seconds');poll=started.get('poll_seconds')
 if not isinstance(interval,(int,float)) or isinstance(interval,bool) or interval<=0 or not isinstance(poll,(int,float)) or isinstance(poll,bool) or poll<=0:
  raise ValueError('invalid supervisor intervals')
 if len(snapshots)<2 or snapshots[-1].get('event')!='terminal_snapshot':raise ValueError('incomplete monitoring snapshot journal')
 # Validate all windows, clocks, chronological order and conservative gaps.
 annotate([],snapshots,time_namespace)
 windows=[x['health_snapshot_window_monotonic_ns'] for x in snapshots]
 if any(b[0]<a[1] for a,b in zip(windows,windows[1:])):raise ValueError('overlapping or reordered monitor snapshots')
 max_gap_ns=int((interval+2*poll+30)*1e9)
 if any(b[0]-a[1]>max_gap_ns for a,b in zip(windows,windows[1:])):raise ValueError('monitor snapshot coverage gap')
 output=[];count=0;starts=[];ends=[]
 for event in events:
  if event.get('status')=='generated':
   bounds=event.get('result',{}).get('boundaries',{})
   if bounds.get('time_namespace')!=time_namespace or bounds.get('clock_implementation')!='clock_gettime(CLOCK_MONOTONIC)':
    raise ValueError('worker and monitor clocks not proven comparable')
   value=annotate([bounds],snapshots,time_namespace)['records'][0]
   starts.append(bounds['t0_before_image_read_ns']);ends.append(bounds['generation_complete_ns'])
   output.append(dict(event,monitor_overlap=value['monitor_overlap'],monitor_snapshot_indices=value['monitor_snapshot_indices']))
   count+=1
  else:output.append(dict(event))
 if not count or windows[0][0]>min(starts) or windows[-1][1]<max(ends):raise ValueError('monitor snapshots do not cover measurement interval')
 return {'events':output,'n_events':len(events),'n_generated':count,'n_monitor_overlaps':sum(x.get('monitor_overlap',False) for x in output),
         'analysis_scope':'all events retained; health windows bracket calls only; residual effects not excluded'}
