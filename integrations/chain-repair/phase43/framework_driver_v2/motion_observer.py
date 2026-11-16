"""Physical-support motion history is separate from geometry confirmation.

Weak classification cannot promote existence or move confirmed anchors. A weak
query with current depth support may contribute a motion sample for a later
already-retained object. Ambiguous associations do not create a certificate.
"""
import numpy as np
from time_local_motion import qualify
class MotionObserver:
 def __init__(self):self.scene=None;self.time=None;self.records=[]
 def observe(self,report,info):
  now=float(info['timestamp']);scene=info['scene_token'];cfg=report.get('config',{});lease=float(cfg.get('max_gap_seconds',.75));margin=float(cfg.get('obstacle_margin_m',.5));association=float(cfg.get('association_distance_m',2.));r=np.asarray(info['explicit_l2g_r_mat']);t=np.asarray(info['explicit_l2g_t']);required=max(3,int(cfg.get('confirmation_observations',3)))
  if self.time is not None and (scene!=self.scene or now<=self.time or now-self.time>lease):self.records=[]
  self.time=now;self.scene=scene;self.records=[x for x in self.records if now-x['time']<=lease]
  if info.get('declared_static_epoch') is not None:
   self.records=[]
   return {},dict(depth_supported_queries=0,qualified_motion_queries=0,ambiguous_associations_rejected=0,geometry_anchors_updated=False,weak_classification_promoted_to_hard=False,scene_specific_rules=False,declared_static_snapshot=True)
  evidence={x['index']:x for x in report.get('global_evidence',{}).get('queries',[])};observations=[]
  for row in sorted(report.get('obstacles',[]),key=lambda x:x['raw_score'],reverse=True):
   index=row['candidate_index'];q=evidence.get(index,{})
   if index<0 or not q.get('protected_observation') or q.get('untrusted_geometry') or row['raw_score']<cfg.get('candidate_floor',.05):continue
   box=np.asarray(row.get('neural_box',row['box']),dtype=float)
   if box.shape!=(9,) or not np.isfinite(box).all() or np.any(box[3:6]<=0):continue
   world=r@box[:3]+t;velocity=r@np.r_[box[7:9],0.]
   if any(x['label']==row['label'] and np.linalg.norm(x['world'][:2]-world[:2])<min(.5,float(min(box[3:5]))/2) for x in observations):continue
   observations.append(dict(index=index,label=row['label'],box=box.copy(),world=world,velocity=velocity))
  pairs=[]
  for j,record in enumerate(self.records):
   predicted=record['history'][-1][1]+record['velocities'][-1]*(now-record['time'])
   for i,obs in enumerate(observations):
    ratio=obs['box'][3:6]/record['box'][3:6];distance=float(np.linalg.norm((obs['world']-predicted)[:2]))
    if obs['label']==record['label'] and (.5<=ratio).all() and (ratio<=2).all() and distance<=association:pairs.append((distance,j,i))
  used_records=set();used_observations=set();certificates={};ambiguous=0
  for distance,j,i in sorted(pairs):
   if j in used_records or i in used_observations:continue
   competitors=[d for d,k,n in pairs if (k==j and n!=i) or (n==i and k!=j)]
   if competitors and min(competitors)-distance<margin:ambiguous+=1;continue
   record=self.records[j];obs=observations[i];record['history']=(record['history']+[(now,obs['world'])])[-required:];record['velocities']=(record['velocities']+[obs['velocity']])[-required:];record['time']=now;record['box']=obs['box'];used_records.add(j);used_observations.add(i)
   cert=qualify(record['history'],record['velocities'],[True]*len(record['history']),now,cfg,r)
   if cert is not None:certificates[obs['index']]=cert
  for i,obs in enumerate(observations):
   if i not in used_observations:self.records.append(dict(time=now,label=obs['label'],box=obs['box'],history=[(now,obs['world'])],velocities=[obs['velocity']]))
  return certificates,dict(depth_supported_queries=len(observations),qualified_motion_queries=len(certificates),ambiguous_associations_rejected=ambiguous,geometry_anchors_updated=False,weak_classification_promoted_to_hard=False,scene_specific_rules=False)
