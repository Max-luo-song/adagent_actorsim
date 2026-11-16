"""Scene data preparation for evaluation. Never imported by the MPC."""

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

import argparse,ast,hashlib,json,datetime,os
pass  # Path is provided by the portable boundary above
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'code/HUGSIM').is_dir())
SOURCE=Path(_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/adscenegraph'))
CASE=_hugsim_expand(json.loads(Path(__file__).with_name('cases.json').read_text()))[os.environ['GENERALIZATION_SCENE']]
SCENE=int(CASE['scene'])
RAW=Path(CASE['raw'])
OMNI=SOURCE/'Omniscenegraph'
CHECKPOINT=Path(CASE['checkpoint'])
CV_TO_SR=np.array([[0,0,1],[-1,0,0],[0,-1,0]],dtype=float)
CLASSES=['road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation','terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle']
LIDAR_COLUMNS=['origin_x','origin_y','origin_z','point_x','point_y','point_z','flow_x','flow_y','flow_z','flow_class','geometric_ground','intensity','elongation','laser_id']
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write_json(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def sr_to_native(xyz):return np.asarray(xyz)@CV_TO_SR

def ply(path,xyz):
 xyz=np.asarray(xyz)
 if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():raise ValueError('Invalid PLY geometry')
 with Path(path).open('xb') as f:
  f.write(('ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nend_header\n'%len(xyz)).encode());np.asarray(xyz,dtype='<f4').tofile(f)
def dense_route(relative):
 source=ROOT/'code/HUGSIM/sim/utils/sim_utils.py';tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='dense_cam_poses')
 scope={'np':np,'SCR':Rotation};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
 poses=np.tile(np.eye(4),(len(relative),1,1));poses[:,:3,:3]=np.einsum('ij,njk,kl->nil',CV_TO_SR.T,relative[:,:3,:3],CV_TO_SR);poses[:,:3,3]=sr_to_native(relative[:,:3,3])
 return scope['dense_cam_poses'](poses,[0]*len(poses))[0]
def read_lidar(path):
 path=Path(path)
 if path.stat().st_size%56:raise ValueError('Incomplete lidar record '+str(path))
 data=np.memmap(path,dtype='<f4',mode='r').reshape(-1,14)
 if not np.isfinite(data).all():raise ValueError('Nonfinite source lidar')
 if not np.isin(data[:,10],[0,1]).all() or not np.isin(data[:,13],range(5)).all():raise ValueError('Unexpected lidar labels')
 return data

def prepare(out):
 import torch
 from PIL import Image
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 fs=sorted((RAW/'ego_pose').glob('*.txt'));ids=[int(f.stem) for f in fs]
 if ids!=list(range(CASE['frames'])):raise ValueError('Source pose coverage changed')
 poses=np.stack([np.loadtxt(f) for f in fs]);relative=np.linalg.inv(poses[0])@poses
 if not np.allclose(relative[0],np.eye(4),atol=1e-8):raise ValueError('Bad scene origin')
 np.savez(out/'poses_and_clock.npz',relative_ego_to_sr_world=relative,timestamp_assumed_s=np.arange(len(fs))*.1)
 dense=dense_route(relative);np.save(out/'native_dense_route.npy',dense)
 print('Exporting reconstructed static geometry...',flush=True)
 states=torch.load(CHECKPOINT,map_location='cpu',mmap=True,weights_only=False)['models']
 road=states['RoadNodes']['_means'].detach().cpu().numpy();bg=states['Background']['_means'].detach().cpu().numpy()
 ply(out/'ground.ply',sr_to_native(road));ply(out/'background_unclassified.ply',sr_to_native(bg))
 np.save(out/'background_sr_world.npy',bg);np.save(out/'road_sr_world.npy',road)
 actors={};groups={}
 for name,semantic in [('RigidNodes','Vehicle'),('DeformableNodes','Cyclist'),('SMPLNodes','Pedestrian'),('TrafficLightNodes','TrafficLight')]:
  if name not in states:continue
  state=states[name]
  for field in ['instances_trans','instances_quats','instances_size','instances_fv']:actors[name+'__'+field]=state[field].detach().cpu().numpy()
  groups[name]={'source_class':semantic,'instances':int(actors[name+'__instances_trans'].shape[1]),'native_foreground_included':name=='RigidNodes','coordinate_frame':'sr_world_first_ego','geometry':'trained reconstruction boxes, not independent physical truth'}
 np.savez_compressed(out/'actors.npz',**actors)
 cameras={};records=[]
 for cam in range(5):
  intrinsic=np.loadtxt(RAW/'intrinsics'/('%d.txt'%cam));extrinsic=np.loadtxt(RAW/'extrinsics'/('%d.txt'%cam));cameras[str(cam)]={'intrinsics_9':intrinsic.tolist(),'camera_waymo_to_ego':extrinsic.tolist(),'opencv_to_waymo':CV_TO_SR.tolist(),'distortion_preserved':True}
 for i in ids:
  lp=RAW/'lidar'/('%03d.bin'%i);data=read_lidar(lp);files={};dimensions={}
  for cam in range(5):
   paths={kind:RAW/folder/('%03d_%d.%s'%(i,cam,ext)) for kind,folder,ext in [('image','images','jpg'),('road','road_masks','png'),('sky','sky_masks','png')]}
   with Image.open(paths['image']) as im:shape=im.size
   for kind,path in paths.items():
    if not path.is_file():raise ValueError('Missing source '+str(path))
    if kind!='image':
     with Image.open(path) as mask:
      if mask.size!=shape:raise ValueError('Source mask/image shape mismatch')
    files[str(cam)+':'+kind]={'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}
   dimensions[str(cam)]=list(shape)
  records.append({'frame':i,'timestamp_assumed_s':i*.1,'lidar':{'path':str(lp),'sha256':sha(lp),'points':len(data),'laser_counts':np.bincount(data[:,13].astype(int),minlength=5).tolist(),'flow_classes':np.unique(data[:,9]).astype(int).tolist(),'geometric_ground_points':int(np.count_nonzero(data[:,10]))},'camera_files':files,'image_size_wh':dimensions})
  if i%25==0:print('Checked source frame',i,flush=True)
 metadata={'schema':'scene_evaluation_data_v1','scene':SCENE,'source_checkpoint':str(CHECKPOINT),'source_checkpoint_sha256':sha(CHECKPOINT),'raw_data':str(RAW),'frames':len(ids),'cameras':cameras,'lidar_columns':LIDAR_COLUMNS,'lidar_kind':'recorded Waymo laser point clouds; not millimeter-wave radar and not a closed-loop virtual lidar','frames_data':records,'actors':groups,'road_points':len(road),'unclassified_background_points':len(bg),'ground_semantic_origin':'RoadNodes initialized with LIDAR_AND_IMAGE: geometric PMF ground intersect image road class0','road_semantics_are_model_predictions':True,'full_cityscapes_classes':CLASSES,'clock_authority':'existing scene adapter assumes 10Hz; original hardware timestamps absent from processed scene','route_definition':'native dense_cam_poses on source ego route; rc=(nearest native xz route index+1)/(0.9*N), clip at1','route_source':'explicit adapted source ego route, not original native ground_param.pkl','native_scorer_semantic_predicate':'ground class0; background classes>1 and !=10; foreground cars only','full_semantic_maps_complete':False,'scene_and_actor_semantics_verified':False,'coordinate_and_clock_verified':False,'ground_truth_used_by_optimizer':False,'missing':['full19class_image_maps','background19class_projection_and_occlusion_validation','clock_authority_or_explicit_assumed_clock_admission','edited_asset_box_and_runtime_adopted_plan_bundle'],'native_evaluation_admitted':False,'formal_evaluation_admitted':False}
 for name in ['instances_info.json','frame_instances.json']:
  f=RAW/'instances'/name;metadata[name]={'path':str(f),'sha256':sha(f)}
 metadata['artifacts']={f.name:sha(f) for f in out.iterdir() if f.is_file()}
 write_json(out/'source_manifest.json',metadata);print('Prepared:',out,flush=True);return metadata

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
 out=args.output or Path(__file__).parent/'runs'/('scene%d_'%SCENE+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
 prepare(out)
if __name__=='__main__':main()
