"""Exogenous start-window feasibility only; no scene/window is promoted here."""
import argparse
import numpy as np
from shapely.geometry import shape
from scene_contracts import *
from start_diagnostics import footprint
RULE={'minimum_source_frame':10,'displacement_sensitivity_m':[6.,8.],'displacement_kind':'cumulative planar path length; origin displacement and initial-axis forward distance also reported','required_controls':100,'dt_s':.1,'extra_clearance_gate_m':.05,'core_interaction_interval':'not_registered; preservation cannot be certified','deployment':'diagnostic_only_original_start_and_pool_unchanged'}

def trajectory_stats(poses):
 p=np.asarray(poses)
 if p.ndim!=3 or p.shape[1:]!=(4,4) or not np.isfinite(p).all():raise ValueError('Invalid source pose sequence')
 xy=p[:,:2,3];arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))];offset=xy-xy[0];forward=offset@p[0,:2,0]
 return arc,np.linalg.norm(offset,axis=1),forward

def windows(arc,gaps,required_controls=100,threshold=6.,minimum_frame=10,extra_gap=.05):
 arc=np.asarray(arc);gaps=np.asarray(gaps);n=len(arc)
 if len(gaps)!=n or required_controls<25:raise ValueError('Invalid declared evaluation window')
 shift=[i for i in range(minimum_frame,n) if arc[i]>=threshold]
 legal=[i for i in shift if np.isfinite(gaps[i]) and gaps[i]>=extra_gap]
 full=[i for i in legal if n-1-i>=required_controls]
 return {'first_displacement_candidate':shift[0] if shift else None,'first_footprint_candidate':legal[0] if legal else None,'first_full_window_start':full[0] if full else None,'full_window_starts':full}

def scan(scene):
 bundle=ASSETS/str(scene);poses=np.load(bundle/'poses_and_clock.npz')['relative_ego_to_sr_world'];arc,offset,forward=trajectory_stats(poses);mapfile=MAPS/str(scene)/'map.json';mapping=read(mapfile) if mapfile.exists() else None;vehicle=read(ROOT/'waymo23-native-import/configs/vehicle.json');gaps=np.full(len(poses),np.nan);body_cover=np.zeros(len(poses),bool)
 if mapping:
  verify(mapping['support']);poly=shape(mapping['geometry'])
  for i,pose in enumerate(poses):
   body=footprint(pose,vehicle['half_length_m'],vehicle['half_width_m']);body_cover[i]=poly.covers(body);gaps[i]=body.distance(poly.boundary)-vehicle['clearance_margin_m']-mapping['uncertainty_m'] if body_cover[i] else -1.
 records=[{'frame':i,'cumulative_distance_m':float(arc[i]),'origin_displacement_m':float(offset[i]),'initial_forward_distance_m':float(forward[i]),'remaining_controls':len(poses)-1-i,'body_covered':bool(body_cover[i]) if mapping else None,'gap_after_existing_margin_m':float(gaps[i]) if np.isfinite(gaps[i]) else None} for i in range(len(poses))]
 sensitivity={}
 for threshold in RULE['displacement_sensitivity_m']:
  row=windows(arc,gaps,RULE['required_controls'],threshold,RULE['minimum_source_frame'],RULE['extra_clearance_gate_m'])
  for key in ('first_displacement_candidate','first_footprint_candidate','first_full_window_start'):
   frame=row[key];row[key+'_record']=records[frame] if frame is not None else None
  if not mapping:row['status']='waiting_road_map_not_a_rejection'
  elif row['first_full_window_start'] is None:row['status']='no_start_preserving_declared_100_control_window'
  else:row['status']='geometric_window_available_core_interval_unregistered'
  sensitivity[str(threshold)]=row
 result={'schema':'effective_start_feasibility_scan_v1','scene':scene,'frames':len(poses),'last_source_frame':len(poses)-1,'rule':RULE,'poses':asset(bundle/'poses_and_clock.npz'),'map':asset(mapfile) if mapping else None,'all_geometric_starts':[records[i] for i in range(len(poses)) if np.isfinite(gaps[i]) and gaps[i]>=RULE['extra_clearance_gate_m']],'sensitivity':sensitivity,'all_frames':records,'core_interaction_preserved':None,'core_interval_missing':True,'GPU_executed':False,'start_map_margin_or_validation_pool_modified':False}
 write(DIAGNOSTICS/'effective_start_v1'/str(scene)/'report.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scenes',type=int,nargs='+',default=[172,552,621,31,233]);a=p.parse_args();summary={'rule':RULE,'scenes':{}}
 for scene in a.scenes:
  r=scan(scene);summary['scenes'][str(scene)]={k:v for k,v in r.items() if k!='all_frames'}
 write(DIAGNOSTICS/'effective_start_v1/summary.json',summary)
 for scene,r in summary['scenes'].items():print(scene,r['frames'],[(d,v['first_displacement_candidate'],v['first_footprint_candidate'],v['first_full_window_start'],v['status']) for d,v in r['sensitivity'].items()])
