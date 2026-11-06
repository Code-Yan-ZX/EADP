"""Independent half-open window oracle; unobserved work is never inferred absent."""
def overlap(start,end,windows,*,worker_namespace,observer_namespace,coverage_start,coverage_end):
 if any(type(v) is not int for v in (start,end)) or start>=end:raise ValueError('invalid request interval')
 if not isinstance(worker_namespace,str) or not worker_namespace or worker_namespace!=observer_namespace:return {'recorded_overlap':'unknown','reason':'time_namespace_unmatched_or_missing','unobserved_interference':'unknown'}
 if type(coverage_start) is not int or type(coverage_end) is not int or coverage_start>start or coverage_end<end:return {'recorded_overlap':'unknown','reason':'coverage_incomplete','unobserved_interference':'unknown'}
 matches=[]
 for i,window in enumerate(windows):
  if not isinstance(window,(list,tuple)) or len(window)!=2 or any(type(v) is not int for v in window) or window[0]>=window[1]:raise ValueError('invalid observation window')
  if window[0]<end and start<window[1]:matches.append(i)
 return {'recorded_overlap':bool(matches),'matching_indices':matches,'unobserved_interference':'unknown'}
