import collections
import numpy as np
from common import *
def smooth(x):return np.convolve(np.pad(x,(2,2),mode='edge'),np.ones(5)/5,mode='valid')
def analyze(scene):
 b=ASSETS/str(scene);raw=read(b/'raw_metadata/raw_metadata.json');old=read(b/'core_interval.json');arr=np.load(b/'poses_and_clock.npz');t=arr['timestamp_raw_s'];poses=arr['relative_ego_to_sr_world'];arc=arr['cumulative_travel_m'];events=old['events'];tracks=collections.defaultdict(dict)
 for i,row in enumerate(raw['frames']):
  for obj in row['object_labels']:
   if obj['waymo_type'] in (1,2,4):tracks[obj['id']][i]=obj
 risk=[];peak_details=[]
 for e in events:
  if e['kind']!='actor_closing_corridor':continue
  track=tracks[e['actor_id']];ids=np.array(sorted(track));local=np.array([track[int(i)]['center_vehicle'][:2] for i in ids]);closing=-np.gradient(smooth(local[:,0]),t[ids]);gap=np.array([local[j,0]-track[int(i)]['size_lwh'][0]/2-1.5 for j,i in enumerate(ids)]);ttc=np.divide(np.maximum(gap,0),closing,out=np.full(len(ids),np.inf),where=closing>.1)
  selected=(ids>=max(10,e['start']))&(ids<=e['end'])&(gap>0)&(abs(local[:,1])<=2.5)&np.isfinite(ttc)
  if not np.any(selected):continue
  js=np.flatnonzero(selected);j=min(js,key=lambda j:(ttc[j],ids[j]));risk.append((float(ttc[j]),int(ids[j]),e['actor_id']))
  k=min(js,key=lambda j:(gap[j],ids[j]));d=min(js,key=lambda j:(np.linalg.norm(local[j]),ids[j]));peak_details.append({'actor_id':e['actor_id'],'episode_frames':[max(10,e['start']),e['end']],'minimum_ttc_frame':int(ids[j]),'minimum_ttc_s':float(ttc[j]),'minimum_longitudinal_gap_frame':int(ids[k]),'minimum_gap_m':float(gap[k]),'minimum_center_distance_frame':int(ids[d]),'minimum_center_distance_m':float(np.linalg.norm(local[d]))})
 stratum='H' if scene in (3,233) else 'N';primary=min(risk) if risk else None
 if stratum=='H' and primary is None:raise ValueError('High-interaction original has no post-warmup risk peak')
 end=old['core_interval_frames'][1] if stratum=='H' else None;rows=[]
 for i in range(len(t)):
  motion=i>=10 and arc[i]>=6 and t[-1]-t[i]>=10
  duration=float(t[end]-t[i]) if end is not None else None
  frac=duration/float(t[end]-t[10]) if end is not None else None
  peak=i<=primary[1] if primary is not None else None
  retained=bool(peak and (duration>=3 or frac>=.75)) if stratum=='H' else True
  rows.append({'frame':i,'cumulative_travel_m':float(arc[i]),'remaining_seconds':float(t[-1]-t[i]),'base_motion_window_pass':bool(motion),'retained_interaction_seconds':duration,'retained_fraction_of_post10_interval':frac,'primary_peak_not_passed':peak,'interaction_retention_pass':retained,'pre_map_candidate':bool(motion and retained)})
 result={'scene':scene,'stratum':stratum,'task_start_frame':10,'source_event_envelope_end':end,'primary_peak_frame':primary[1] if primary else None,'primary_min_ttc_s':primary[0] if primary else None,'primary_actor_id':primary[2] if primary else None,'all_actor_peaks':peak_details,'end_unchanged_from_v1':True,'retention_duration_operator':'OR: duration>=3.0s or post10_fraction>=0.75, plus primary peak coverage mandatory','candidate_frames_before_map':[r['frame'] for r in rows if r['pre_map_candidate']],'all_frames':rows,'sources':[asset(b/'raw_metadata/raw_metadata.json'),asset(b/'core_interval.json'),asset(b/'poses_and_clock.npz')],'no_map_or_patch_performance_used':True}
 freeze(b/'interaction_registration_v2.json',result);print('REGISTERED',scene,stratum,'peak',result['primary_peak_frame'],'TTC',result['primary_min_ttc_s'],'pre-map starts',result['candidate_frames_before_map'],flush=True)
 return asset(b/'interaction_registration_v2.json')
if __name__=='__main__':freeze(OUT/'core_registration_frozen.json',{'rules':asset(OUT/'preregistration.json'),'scenes':{str(s):analyze(s) for s in SCENES}})
