"""Canonical actor boxes and edited asset for post-control evaluation only."""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand
Path = _HugsimPath

import argparse,json
pass  # Path is provided by the portable boundary above
import numpy as np
from scene_data import ROOT,write_json,sha

def export(bundle):
 bundle=Path(bundle);raw=_hugsim_expand(json.loads((bundle/'raw_metadata/raw_metadata.json').read_text()));poses=np.load(bundle/'poses_and_clock.npz')['relative_ego_to_sr_world'];rows=[]
 for frame in raw['frames']:
  i=frame['frame'];pose=poses[i];actors=[]
  for obj in frame['object_labels']:
   center=pose[:3,:3]@np.asarray(obj['center_vehicle'])+pose[:3,3];yaw=obj['heading_vehicle'];direction=pose[:3,:3]@np.array([np.cos(yaw),np.sin(yaw),0.]);l,w,h=obj['size_lwh'];heading=float(np.arctan2(direction[1],direction[0]));kind={1:'vehicle',2:'pedestrian',3:'sign',4:'cyclist'}.get(obj['waymo_type'],'unknown')
   actors.append({'id':obj['id'],'source_class':kind,'waymo_type':obj['waymo_type'],'box_xyzwlhyaw':[float(center[0]),float(center[1]),float(center[2]),w,l,h,heading],'native_foreground_name':'car' if obj['waymo_type']==1 else kind,'native_car_mapping':'Waymo Vehicle includes vehicle subtypes; no unsupported subtype classification'})
  rows.append({'frame':i,'simulation_time_s':i*.1,'hardware_time_micros':frame['timestamp_micros'],'objects':actors})
 work=Path(__file__).parent.parent/'native_alignment_v1';insert_path=work/'insert_request.json';insert=_hugsim_expand(json.loads(insert_path.read_text()));cal_path=Path(insert['asset_calibration']);cal=_hugsim_expand(json.loads(cal_path.read_text()));bounds=np.asarray(cal['support']['bounds_raw']);matrix=np.asarray(insert['asset_to_world']);corners=np.array([[bounds[a,0],bounds[b,1],bounds[c,2],1.] for a in [0,1] for b in [0,1] for c in [0,1]]);world=(matrix@corners.T).T[:,:3];center=world.mean(axis=0);width,length,height=cal['support']['wlh'];axis=matrix[:3,0];yaw=float(np.arctan2(axis[1],axis[0]));bottom=float(world[:,2].min());asset={'id':insert['object_id'],'class':'car','box_xyzwlhyaw':[float(center[0]),float(center[1]),float(center[2]),width,length,height,yaw],'support_corners_sr_world':world.tolist(),'ground_bottom_z':bottom,'stationary':True,'active_source_frames':[0,198],'insert_sha256':sha(insert_path),'calibration_sha256':sha(cal_path),'dimension_authority':cal['rule']['dimensions_source'],'independent_physical_calibration':cal['independent_physical_calibration']}
 result={'schema':'scene551_actor_reference_v1','frame':'sr_world_first_ego = native scorer imu world','frames':rows,'edited_asset':asset,'clock_policy':'same source frame-index clock as current simulator; true sensor times are retained separately','source':'original Waymo3D annotations plus calibrated edit support','actor_interpolation_policy':'integer source frames; no assumed extrapolation','ground_truth_used_by_optimizer':False,'formal_evaluation_admitted':False}
 write_json(bundle/'evaluation_reference.json',result);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);a=p.parse_args();export(a.bundle)
if __name__=='__main__':main()
