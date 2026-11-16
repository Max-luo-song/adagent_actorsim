"""Aligned geometry/observation contract. Raw neural outputs are immutable evidence."""
import hashlib,json
import numpy as np
SCHEMA='aligned_box_observation_v1'

def fingerprint(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()

def build(arrays,origin,evidence,refinement,info,vehicle):
 raw=np.asarray(arrays['boxes']);scores=np.asarray(arrays['scores']);labels=np.asarray(arrays['labels']);candidate=np.asarray(arrays['candidate_boxes']);cs=np.asarray(arrays['candidate_scores']);cl=np.asarray(arrays['candidate_labels'])
 decoded=np.asarray(origin['decoded_gravity_boxes'],dtype=np.float32).reshape(-1,9).copy();decoded[:,2]-=decoded[:,5]/2
 if raw.ndim!=2 or raw.shape[1]!=9 or candidate.shape!=decoded.shape or any(np.asarray(x).shape!=(len(raw),) for x in [scores,labels]):raise ValueError('Malformed or unaligned neural boxes')
 if origin['decode_input_has_special_ego'] or not np.allclose(candidate,decoded,rtol=0,atol=1e-4):raise ValueError('Decoder origin/query mapping changed')
 if evidence['timestamp']!=float(info['timestamp']) or evidence['scene_token']!=info['scene_token']:raise ValueError('Stale scene evidence')
 if not all(np.isfinite(a).all() for a in [raw,candidate,scores]) or np.any(raw[:,3:6]<=0):raise ValueError('Invalid box geometry')
 geometry=raw.copy();qualified=np.zeros(len(raw),dtype=bool);rows=[]
 proposed=np.asarray(arrays['geometry_candidate_boxes']);queries={r['index']:r for r in evidence['queries']};refinements={r['candidate_index']:r for r in refinement['candidates']}
 if proposed.shape!=candidate.shape or len(origin['selected_track_ids'])!=len(candidate):raise ValueError('Candidate/geometry/identity count mismatch')
 ground=-float(np.asarray(info['model_lidar_frame']['virtual_to_ego'])[2,3]);vertical_margin=vehicle.clearance_margin_m
 for i,b in enumerate(raw):
  matches=np.flatnonzero(np.all(np.abs(candidate-b)<=1e-4,axis=1)&(cl==labels[i])&(np.abs(cs-scores[i])<=1e-5))
  if len(matches)!=1:raise ValueError('Retained-to-decoder mapping missing/ambiguous; no fabricated observation')
  j=int(matches[0]);q=queries.get(j);r=refinements.get(j)
  if q is None:raise ValueError('Missing per-query evidence')
  if q['neural_id']!=origin['selected_track_ids'][j]:raise ValueError('Evidence query identity differs')
  visible=q['visible_center'] is True;surface=q['support_pixels']>0 and q['reliable_pixels']>0
  intersects_height=bool(b[2]<=ground+vehicle.body_height_m+vertical_margin and b[2]+b[5]>=ground-vertical_margin)
  corrected=False
  if visible and surface and r is not None and r['accepted'] is True:
   if not np.allclose(r['original_box'],candidate[j],rtol=0,atol=1e-4) or not np.allclose(r['box'],proposed[j],rtol=0,atol=1e-4):raise ValueError('Geometry refinement mapping changed')
   if not np.allclose(proposed[j,2:],candidate[j,2:],rtol=0,atol=1e-5):raise ValueError('Unregistered dimension/height/heading/velocity rewrite')
   geometry[i]=proposed[j];corrected=True
  qualified[i]=visible and surface
  rows.append(dict(index=i,candidate_index=j,neural_id=int(q['neural_id']),visible_center=visible,support_pixels=int(q['support_pixels']),reliable_pixels=int(q['reliable_pixels']),protected_history=q['protected_observation'],vertical_overlap_possible=intersects_height,current_observation_qualified=bool(qualified[i]),geometry_center_refined=corrected,reason='current_visible_surface_supported' if qualified[i] else 'unqualified_geometry_kept_as_soft_risk'))
 packet=dict(schema=SCHEMA,scene_token=info['scene_token'],timestamp=float(info['timestamp']),raw_boxes_sha256=fingerprint(raw),scores_sha256=fingerprint(scores),labels_sha256=fingerprint(labels),geometry_sha256=fingerprint(geometry),rows=rows,raw_neural_modified=False,physical_geometry_certified=False,free_space_asserted=False)
 return dict(arrays,validated_memory_boxes=geometry,validated_memory_mask=qualified,box_observation_contract=packet)

def validate(arrays,info):
 p=arrays['box_observation_contract'];g=np.asarray(arrays['validated_memory_boxes']);m=np.asarray(arrays['validated_memory_mask'])
 if p['schema']!=SCHEMA or p['scene_token']!=info['scene_token'] or p['timestamp']!=float(info['timestamp']):raise ValueError('Stale box contract')
 if m.shape!=(len(arrays['boxes']),) or m.dtype.kind!='b' or g.shape!=np.shape(arrays['boxes']):raise ValueError('Bad qualified geometry layout')
 for k,a in [('raw_boxes',arrays['boxes']),('scores',arrays['scores']),('labels',arrays['labels']),('geometry',g)]:
  if fingerprint(np.asarray(a))!=p[k+'_sha256']:raise ValueError('Box contract arrays changed')
 if len(p['rows'])!=len(m) or any(r['index']!=i or r['current_observation_qualified']!=bool(m[i]) for i,r in enumerate(p['rows'])):raise ValueError('Box qualification mismatch')
 return g,m
