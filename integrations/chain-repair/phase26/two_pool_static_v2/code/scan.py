import numpy as np
from shapely.geometry import Polygon,shape,box
from common import *

def body_polygon(pose,half_length=1.5,half_width=.8):
 points=np.array([[-half_length,-half_width,0],[half_length,-half_width,0],[half_length,half_width,0],[-half_length,half_width,0]])@pose[:3,:3].T+pose[:3,3]
 return Polygon(points[:,:2])

def qualified(frame_record,footprint_pass):
 return bool(frame_record['base_motion_window_pass'] and frame_record['interaction_retention_pass'] and footprint_pass)

def scan(scene):
 b=ASSETS/str(scene);path=MAPS/str(scene)/'map.json';m=read(path);verify(m['support']);poly=shape(m['geometry']);npz=np.load(b/'poses_and_clock.npz');poses=npz['relative_ego_to_sr_world'];registered=read(b/'interaction_registration_v2.json');records=[]
 for r,pose in zip(registered['all_frames'],poses):
  body=body_polygon(pose);covered=bool(poly.covers(body));gap=float(body.distance(poly.boundary)) if covered else None;foot=bool(covered and gap>=.5-1e-9)
  records.append({**r,'body_covered':covered,'minimum_body_boundary_distance_m':gap,'full_body_plus_half_meter_pass':foot,'complete_admission_pass':qualified(r,foot)})
 first=lambda predicate:next((r for r in records if predicate(r)),None)
 first_foot_any=first(lambda r:r['frame']>=10 and r['full_body_plus_half_meter_pass']);first_foot=first(lambda r:r['frame']>=10 and r['cumulative_travel_m']>=6 and r['full_body_plus_half_meter_pass']);valid=first(lambda r:r['complete_admission_pass']);first_base=first(lambda r:r['base_motion_window_pass'])
 if valid:reason='admitted'
 elif first_foot is None:reason='no_full_body_plus_0p5m_road_support_after_6m'
 elif not any(r['base_motion_window_pass'] and r['full_body_plus_half_meter_pass'] for r in records):reason='footprint_only_available_after_remaining_window_below_10s'
 else:reason='full_footprint_and_window_exist_but_interaction_peak_or_retention_not_preserved'
 points=[r['frame'] for r in (first_base,first_foot,valid) if r is not None]
 if not points:points=[10]
 # Add the pre-map candidate with most body coverage for diagnostic localization only.
 candidates=registered['candidate_frames_before_map']
 if candidates:points.append(max(candidates,key=lambda i:poly.intersection(body_polygon(poses[i]).buffer(.5)).area))
 support=np.load(m['support']['path']);road=support['road_votes'];seen=support['qualified_frame_votes'];known=support['known_road'];origin=support['origin'];step=float(support['step']);diagnostics=[]
 for frame in sorted(set(points)):
  body=body_polygon(poses[frame]);outer=body.buffer(.5,quad_segs=32);ring=outer.difference(body);sections={}
  for name,geom in [('body',body),('margin_ring',ring)]:
   totals={'zero_qualified':0.,'insufficient_road_support':0.,'rejected_with_model_nonroad_votes':0.,'admitted_road':0.};areas=[]
   lo=np.floor((np.array(geom.bounds[:2])-origin)/step-.5).astype(int);hi=np.ceil((np.array(geom.bounds[2:])-origin)/step+.5).astype(int)
   for y in range(max(0,lo[1]),min(seen.shape[0]-1,hi[1])+1):
    for x in range(max(0,lo[0]),min(seen.shape[1]-1,hi[0])+1):
     px,py=origin+np.array([x,y])*step;cell=box(px-step/2,py-step/2,px+step/2,py+step/2);area=geom.intersection(cell).area
     if area<=1e-10:continue
     category='admitted_road' if known[y,x] else ('zero_qualified' if seen[y,x]==0 else ('rejected_with_model_nonroad_votes' if seen[y,x]>road[y,x] else 'insufficient_road_support'))
     totals[category]+=area
     if category!='admitted_road':areas.append({'row':y,'column':x,'area_m2':area,'qualified_votes':int(seen[y,x]),'road_votes':int(road[y,x]),'category':category})
   sections[name]={'area_m2':geom.area,'four_state_intersection_area_m2':totals,'unadmitted_intersecting_cells':areas}
  diagnostics.append({'frame':frame,'footprint_statistics':sections})
 result={'schema':'four_scene_full_static_admission_v2','scene':scene,'stratum':registered['stratum'],'map':asset(path),'core_registration':asset(b/'interaction_registration_v2.json'),'first_body_and_ring_qualified_frame_from10':first_foot_any,'first_footprint_qualified_frame_after6m':first_foot,'first_valid_start':valid,'primary_peak_frame':registered['primary_peak_frame'],'source_event_end':registered['source_event_envelope_end'],'status':'admitted' if valid else 'rejected','reason':reason,'diagnostics':diagnostics,'all_frames':records,'no_closed_loop_performance_used':True,'remaining_seconds_use_raw_hardware_clock':True,'ego_annotation_limitation':read(b/'ego_masks/manifest.json')['low_light_review_limitation']}
 write(OUT/'scans'/str(scene)/'scan.json',result);print('SCAN',scene,result['status'],'foot6m',first_foot['frame'] if first_foot else None,'valid',valid['frame'] if valid else None,reason,flush=True);return result

if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--scene',type=int,required=True);a=p.parse_args();scan(a.scene)
