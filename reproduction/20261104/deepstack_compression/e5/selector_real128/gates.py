"""Per-candidate and per-case admission; failures remain visible."""
SYNTHETIC=('ties','zeros','near_ties','seeded_random','boundary_one','boundary_all')
def synthetic_eligibility(rows):
 if [r.get('case') for r in rows]!=[*SYNTHETIC,'graph_changed_input']:raise ValueError('incomplete synthetic inventory')
 for i,r in enumerate(rows):
  n=32 if r['case']=='seeded_random' else 8;k=r.get('k');author=r.get('author_ids');graph_ids=r.get('graph_ids')
  expected_k={'ties':4,'zeros':4,'near_ties':4,'seeded_random':8,'boundary_one':1,'boundary_all':8,'graph_changed_input':4}[r['case']]
  if k!=expected_k or not isinstance(author,list) or not isinstance(graph_ids,list) or len(author)!=k or len(graph_ids)!=k or any(type(x) is not int or not 0<=x<n for x in author+graph_ids) or len(set(author))!=k or len(set(graph_ids))!=k or r.get('graph_equal') is not (author==graph_ids):raise ValueError('synthetic author/graph record invalid')
  if r['case'] in ('ties','zeros','boundary_one','boundary_all') and author!=list(range(k)):raise ValueError('synthetic author tie rule mismatch')
  if i<6:
   fused_ids=r.get('fused_ids')
   if not isinstance(fused_ids,list) or len(fused_ids)!=k or any(type(x) is not int or not 0<=x<n for x in fused_ids) or len(set(fused_ids))!=k or r.get('fused_equal') is not (author==fused_ids):raise ValueError('synthetic fused record invalid')
  else:
   before=r.get('before_graph_ids')
   if not isinstance(before,list) or len(before)!=k or r.get('different_from_original') is not (before!=graph_ids):raise ValueError('synthetic changed-input record invalid')
 fused=all(r.get('fused_equal') is True for r in rows[:6]);graph=all(r.get('graph_equal') is True for r in rows) and rows[-1].get('different_from_original') is True
 return {'fused':fused,'graph':graph}

def endpoint_eligible(identity,reconstruction,visual,author,cross,capture_order_equal):
 flags=('e4_common_metadata_equal','e4_order_equal','e4_spatial_equal','e4_pruned_hash_equal')
 return (identity.get('status')=='passed' and reconstruction.get('status')=='passed' and visual is True and all(author.get(k) is True for k in flags) and cross.get('available') is True and cross.get('importance_equal_256') is True and cross.get('similarity_equal_256') is True and capture_order_equal is True)

def reconstruction_claim(result,reported):
 if result is None:
  if reported.get('status')!='failed':raise ValueError('reported successful reconstruction despite mapping failure')
  return False
 if reported.get('status') not in ('passed','mismatch') or reported.get('sequence_equal')!=result['sequence_equal'] or reported.get('mapped_indices')!=result['mapped_indices']:raise ValueError('instruction mapping record inconsistent')
 if reported.get('status')!=('passed' if reported.get('sequence_equal') is True and reported.get('mean_equal') is True else 'mismatch'):raise ValueError('reconstruction status inconsistent')
 return result['sequence_equal'] and reported.get('mean_equal') is True
