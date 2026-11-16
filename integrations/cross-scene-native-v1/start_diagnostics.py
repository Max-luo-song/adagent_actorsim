"""Read-only start-footprint audit; never modifies maps, margins or eligibility."""
import argparse,math
import numpy as np
from PIL import Image
from shapely.geometry import Polygon,box,shape,mapping
from scipy.ndimage import minimum_filter
from scene_contracts import *
from boundaries import camera_projection
LABELS=('zero_qualified_observation','road_only_insufficient_support','rejected_with_nonroad_evidence','admitted_by_current_votes')
GEOMETRY_DIAGNOSTIC={'translation_m':.5,'rotation_deg':5.,'deployment':'counterfactual_only_no_map_change','bucket_merge':'mixed road/nonroad evidence becomes unknown; nonroad flag retained'}

def classify(road,seen,minimum=2,fraction=.8):
 road=np.asarray(road);seen=np.asarray(seen)
 if np.any(road>seen) or np.any(road<0) or np.any(seen<0):raise ValueError('Invalid vote counts')
 known=(road>=minimum)&(road>=fraction*np.maximum(seen,1));labels=np.full(road.shape,2,np.uint8);labels[seen==0]=0;labels[(seen>0)&(road<minimum)&(road==seen)]=1;labels[known]=3
 return labels

def camera_axes(source):
 rows=[]
 for cid,c in source['cameras'].items():
  R=np.asarray(c['camera_waymo_to_ego'])[:3,:3]@np.asarray(c['opencv_to_waymo']);axis=R[:,2];rows.append({'camera':int(cid),'optical_axis_ego':axis.tolist(),'azimuth_deg':float(np.degrees(np.arctan2(axis[1],axis[0]))),'rear_facing_optical_axis':bool(axis[0]<-.7)})
 return rows

def footprint(pose,half_length,half_width):
 points=np.array([[-half_length,-half_width,0],[half_length,-half_width,0],[half_length,half_width,0],[-half_length,half_width,0]]);return Polygon((points@pose[:3,:3].T+pose[:3,3])[:,:2])

def cells_for_regions(body,ring,origin,step,grid_shape):
 region=body.union(ring);x0,y0,x1,y1=region.bounds;rows=[]
 for j in range(math.floor((y0-origin[1])/step)-1,math.ceil((y1-origin[1])/step)+2):
  for i in range(math.floor((x0-origin[0])/step)-1,math.ceil((x1-origin[0])/step)+2):
   center=np.asarray(origin)+np.array([i,j])*step;cell=box(center[0]-step/2,center[1]-step/2,center[0]+step/2,center[1]+step/2);b=cell.intersection(body).area;r=cell.intersection(ring).area
   if b+r<=1e-12:continue
   if not 0<=j<grid_shape[0] or not 0<=i<grid_shape[1]:raise ValueError('Footprint lies outside frozen vote grid')
   rows.append({'i':i,'j':j,'x':float(center[0]),'y':float(center[1]),'body_area_m2':float(b),'ring_area_m2':float(r)})
 return rows

def region_stats(cells,states,road,seen,region):
 weights=np.array([c[region+'_area_m2'] for c in cells]);active=weights>1e-12;area=float(weights.sum());nonroad=seen-road
 result={'intersecting_cells':int(active.sum()),'area_m2':area,'states':{}}
 for value,name in enumerate(LABELS):
  keep=active&(states==value);a=float(weights[keep].sum());result['states'][name]={'cells':int(keep.sum()),'area_m2':a,'area_percent':100*a/area if area else 0.}
 result['cells_with_any_retained_nonroad_vote']=int((active&(nonroad>0)).sum());result['nonroad_evidence_area_percent']=100*float(weights[nonroad>0].sum())/area if area else 0.
 result['qualified_frame_votes_total']=int(seen[active].sum());result['road_frame_votes_total']=int(road[active].sum());return result

def cause_breakdown(cells,region):
 weights=np.array([c[region+'_area_m2'] for c in cells]);active=weights>1e-12;zero=np.array([c['qualified_votes']==0 for c in cells]);e={k:np.array([c['evidence'][k] for c in cells]) for k in ('visible_frames','unoccluded_frames','confident_frames','static_semantic_frames','cross_camera_conflict_frames')}
 flags={'zero_votes_never_projected_in_sampled_views':zero&(e['visible_frames']==0),'zero_votes_visible_but_always_occluded':zero&(e['visible_frames']>0)&(e['unoccluded_frames']==0),'zero_votes_unoccluded_but_never_confident':zero&(e['unoccluded_frames']>0)&(e['confident_frames']==0),'zero_votes_confident_but_no_static_class':zero&(e['confident_frames']>0)&(e['static_semantic_frames']==0),'zero_votes_cross_camera_conflict_only':zero&(e['static_semantic_frames']>0)}
 for cell in cells:
  cell['evidence']['road_candidate_frames_before_conflict']=cell['road_votes']+cell['evidence']['cross_camera_conflict_frames'];cell['evidence']['nonroad_candidate_frames_before_conflict']=cell['nonroad_votes']+cell['evidence']['cross_camera_conflict_frames']
 flags['any_nonroad_evidence_even_if_conflict_rejected']=np.array([c['evidence']['nonroad_candidate_frames_before_conflict']>0 for c in cells])
 return {name:{'cells':int((active&flag).sum()),'area_m2':float(weights[flag].sum()),'area_percent':100*float(weights[flag].sum())/float(weights.sum())} for name,flag in flags.items()}

def blind_area_topology(road_polygon,body,expanded):
 from shapely.ops import unary_union
 parts=list(road_polygon.geoms) if road_polygon.geom_type=='MultiPolygon' else [road_polygon];shells=unary_union([Polygon(part.exterior) for part in parts])
 return {'body_inside_known_road_outer_shell':bool(shells.covers(body)),'expanded_inside_known_road_outer_shell':bool(shells.covers(expanded)),'diagnostic_only_no_holes_filled_in_actual_map':True}

class PoseBuckets:
 def __init__(self):self.rows=[]
 def index(self,camera,pose):
  for i,row in enumerate(self.rows):
   if row['camera']!=camera:continue
   distance=np.linalg.norm(pose[:3,3]-row['pose'][:3,3]);rotation=pose[:3,:3]@row['pose'][:3,:3].T;angle=np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2,-1.,1.)))
   if distance<GEOMETRY_DIAGNOSTIC['translation_m'] and angle<GEOMETRY_DIAGNOSTIC['rotation_deg']:return i
  self.rows.append({'camera':camera,'pose':pose.copy()});return len(self.rows)-1

def replay(scene,data,source,cells):
 bundle=ASSETS/str(scene);raw=read(bundle/'raw_metadata/raw_metadata.json');semantic=read(bundle/'semantic19/semantic_manifest.json');catalog={(r['frame'],r['camera']):r for r in semantic['maps']};poses=np.load(bundle/'poses_and_clock.npz')['relative_ego_to_sr_world'];inverse_origin=np.linalg.inv(np.loadtxt(Path(source['raw_data'])/'ego_pose/000.txt'));construction=read(MAPS/str(scene)/'construction.json');planes={r['frame']:r['ground_plane'] for r in construction['frames']};xy=np.array([[c['x'],c['y']] for c in cells]);N=len(cells);params=data['parameters']
 counters={k:np.zeros(N,np.int32) for k in ('in_range_frames','visible_frames','unoccluded_frames','confident_frames','static_semantic_frames','cross_camera_conflict_frames','replayed_road_votes','replayed_qualified_votes')};buckets=PoseBuckets();bucket_votes={};per_camera={}
 for frame in data['source_frames']:
  record=source['frames_data'][frame];verify(record['lidar']);lidar=np.memmap(record['lidar']['path'],dtype='<f4',mode='r').reshape(-1,14);pose=poses[frame];plane=np.asarray(planes[frame]);normal=pose[:3,:3]@np.r_[-plane[:2],1.];offset=plane[2]+normal@pose[:3,3];z=(offset-normal[0]*xy[:,0]-normal[1]*xy[:,1])/normal[2];world=np.c_[xy,z];local=(world-pose[:3,3])@pose[:3,:3];near=np.linalg.norm(local[:,:2],axis=1)<=params['max_local_range_m'];cloud=lidar[:,3:6]@pose[:3,:3].T+pose[:3,3]
  masks={k:np.zeros(N,bool) for k in ('visible','unoccluded','confident','static','yes','no')};views=[];counters['in_range_frames']+=near
  for camera in raw['frames'][frame]['camera_images']:
   cid=camera['camera'];row=catalog[(frame,cid)];capture=inverse_origin@np.asarray(camera['pose']);cal=source['cameras'][str(cid)];arrays=[]
   for key in ('file','confidence_file','margin_file'):
    file=bundle/'semantic19'/row[key];verify({'path':str(file),'sha256':row['sha256' if key=='file' else key.replace('_file','_sha256')]});arrays.append(np.asarray(Image.open(file)))
   labels,confidence,margin=arrays;h,w=labels.shape;uv,depth,visible=camera_projection(world,capture,cal,(w,h));visible&=near;lp,ld,lv=camera_projection(cloud,capture,cal,(w,h));buffer=np.full(h*w,np.inf,np.float32);np.minimum.at(buffer,lp[lv,1]*w+lp[lv,0],ld[lv]);buffer=minimum_filter(buffer.reshape(h,w),size=5,mode='constant',cval=np.inf);u,v=uv.T
   unoccluded=visible&(depth<=buffer[v,u]+params['occlusion_tolerance_m']);confident=unoccluded&(confidence[v,u]>=params['semantic_confidence_u8'])&(margin[v,u]>=params['semantic_margin_u8']);static=confident&(labels[v,u]<=9);yes=static&(labels[v,u]==0);no=static&(labels[v,u]!=0)
   for name,value in zip(('visible','unoccluded','confident','static','yes','no'),(visible,unoccluded,confident,static,yes,no)):masks[name]|=value
   optical=capture@np.asarray(cal['camera_waymo_to_ego']);views.append((buckets.index(cid,optical),yes,no));per_camera.setdefault(str(cid),np.zeros(N,np.int32));per_camera[str(cid)]+=visible
  conflict=masks['yes']&masks['no'];qualified=masks['static']&~conflict;road=masks['yes']&qualified
  for key,name in (('visible_frames','visible'),('unoccluded_frames','unoccluded'),('confident_frames','confident'),('static_semantic_frames','static')):counters[key]+=masks[name]
  counters['cross_camera_conflict_frames']+=conflict;counters['replayed_road_votes']+=road;counters['replayed_qualified_votes']+=qualified
  for bucket,yes,no in views:
   if bucket not in bucket_votes:bucket_votes[bucket]=[np.zeros(N,bool),np.zeros(N,bool)]
   bucket_votes[bucket][0]|=yes&qualified;bucket_votes[bucket][1]|=no&qualified
  print('Start audit',scene,'frame',frame,flush=True)
 geometry_road=np.zeros(N,np.int32);geometry_seen=np.zeros(N,np.int32);mixed=np.zeros(N,np.int32)
 for yes,no in bucket_votes.values():
  mixed+=yes&no;geometry_road+=yes&~no;geometry_seen+=yes^no
 return counters,{'road_votes':geometry_road,'qualified_votes':geometry_seen,'mixed_buckets_unknown':mixed,'parameters':GEOMETRY_DIAGNOSTIC,'camera_bucket_count':len(buckets.rows)},per_camera

def run(scene):
 folder=MAPS/str(scene);mapfile=folder/'map.json'
 if not mapfile.exists():return {'scene':scene,'status':'waiting_map_assets','GPU_executed':False}
 data=read(mapfile);verify(data['support']);source=read(ASSETS/str(scene)/'source_manifest.json');grid=np.load(data['support']['path']);pose=np.load(ASSETS/str(scene)/'poses_and_clock.npz')['relative_ego_to_sr_world'][10];vehicle=read(ROOT/'waymo23-native-import/configs/vehicle.json');body=footprint(pose,vehicle['half_length_m'],vehicle['half_width_m']);margin=vehicle['clearance_margin_m']+data['uncertainty_m'];expanded=body.buffer(margin,resolution=32);ring=expanded.difference(body);cells=cells_for_regions(body,ring,grid['origin'],float(grid['step']),grid['known_road'].shape);indices=([c['j'] for c in cells],[c['i'] for c in cells]);road=grid['road_votes'][indices].astype(int);seen=grid['qualified_frame_votes'][indices].astype(int);states=classify(road,seen,data['parameters']['min_independent_frames'],data['parameters']['min_road_fraction']);counters,geometry,per_camera=replay(scene,data,source,cells)
 if not np.array_equal(counters['replayed_road_votes'],road) or not np.array_equal(counters['replayed_qualified_votes'],seen):raise ValueError('ROI replay differs from frozen full-grid vote counts')
 for i,cell in enumerate(cells):
  cell.update(state=LABELS[states[i]],road_votes=int(road[i]),qualified_votes=int(seen[i]),nonroad_votes=int(seen[i]-road[i]),evidence={k:int(v[i]) for k,v in counters.items()},camera_visible_frames={k:int(v[i]) for k,v in per_camera.items()})
 poly=shape(data['geometry']);covered=poly.covers(body);raw_gap=body.distance(poly.boundary) if covered else None;newstates=classify(geometry['road_votes'],geometry['qualified_votes'],data['parameters']['min_independent_frames'],data['parameters']['min_road_fraction'])
 result={'schema':'start_footprint_evidence_audit_v1','scene':scene,'source_frame':10,'map':asset(mapfile),'support':data['support'],'camera_axes':camera_axes(source),'regions':{r:region_stats(cells,states,road,seen,r) for r in ('body','ring')},'geometry_only_counterfactual':{'parameters':GEOMETRY_DIAGNOSTIC,'regions':{r:region_stats(cells,newstates,geometry['road_votes'],geometry['qualified_votes'],r) for r in ('body','ring')},'camera_pose_buckets':geometry['camera_bucket_count']},'body_covers':covered,'body_to_map_boundary_distance_m':raw_gap,'required_clearance_m':margin,'margin_only_failure':bool(covered and raw_gap<margin+.05),'zero_qualified_is_not_proof_of_never_visible':True,'fourth_state_is_grid_acceptance_not_physical_margin_failure':True,'same_grid_cell_may_intersect_both_regions':True,'body_polygon':mapping(body),'expanded_polygon':mapping(expanded),'cells':cells,'map_or_policy_modified':False,'GPU_executed':False}
 result['blind_area_topology']=blind_area_topology(poly,body,expanded)
 result['cause_breakdown']={region:cause_breakdown(cells,region) for region in ('body','ring')}
 destination=DIAGNOSTICS/'start_support_v1'/str(scene);write(destination/'report.json',result);plot(result,destination);return result

def plot(report,destination):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.patches import Rectangle
 colors=['#888888','#e6aa32','#c93434','#43a55b'];fig,ax=plt.subplots(figsize=(8,7))
 for c in report['cells']:ax.add_patch(Rectangle((c['x']-.25,c['y']-.25),.5,.5,color=colors[LABELS.index(c['state'])],alpha=.7));ax.text(c['x'],c['y'],str(c['road_votes'])+'/'+str(c['qualified_votes']),ha='center',va='center',fontsize=7)
 for key,style,label in (('expanded_polygon','--','Body + 0.5m'),('body_polygon','-','Body')):
  p=np.asarray(report[key]['coordinates'][0]);ax.plot(p[:,0],p[:,1],style,c='black',label=label)
 ax.autoscale();ax.set_aspect('equal');ax.legend();ax.set_title('Scene '+str(report['scene'])+': road/qualified votes\nGray zero | Amber insufficient road-only | Red rejected nonroad | Green admitted');fig.tight_layout();fig.savefig(destination/'footprint_states.png',dpi=150);plt.close(fig)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scene',type=int,required=True);a=p.parse_args();r=run(a.scene);print(__import__('json').dumps({k:v for k,v in r.items() if k not in ('cells','body_polygon','expanded_polygon')},ensure_ascii=False,indent=2))
