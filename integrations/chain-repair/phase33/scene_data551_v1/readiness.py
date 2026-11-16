"""Report completed evidence and unmet native data requirements explicitly."""

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

import argparse,json,ast
pass  # Path is provided by the portable boundary above
import numpy as np
from scene_data import sha,write_json,OMNI

def check(bundle):
 bundle=Path(bundle);source=_hugsim_expand(json.loads((bundle/'source_manifest.json').read_text()));raw=_hugsim_expand(json.loads((bundle/'raw_metadata/raw_metadata.json').read_text()));reference=_hugsim_expand(json.loads((bundle/'evaluation_reference.json').read_text()))
 for name,digest in source['artifacts'].items():
  if sha(bundle/name)!=digest:raise ValueError('Changed prepared data '+name)
 if not raw['raw_processed_pose_identity_all_frames'] or not raw['raw_processed_image_identity_all_995_images'] or not raw['raw_processed_calibration_identity']:raise ValueError('Original data identity not verified')
 checks={'source_199_frames':source['frames']==199,'all_5_camera_calibrations':len(source['cameras'])==5,'all_recorded_lidar_frames':len(source['frames_data'])==199,'reconstructed_road_geometry':source['road_points']>0,'source_hardware_clock':len(raw['frames'])==199,'raw_3d_actor_annotations':sum(len(r['objects']) for r in reference['frames'])==raw['original_3d_labels'],'edited_vehicle_reference':bool(reference['edited_asset']['stationary']),'native_dense_route':len(np.load(bundle/'native_dense_route.npy'))>199,'full_19class_semantic_maps':False,'point_and_background_semantics':False,'semantic_coverage_and_occlusion_review':False,'runtime_adopted_plan_native_bundle':False}
 sm=bundle/'semantic19/semantic_manifest.json';pm=bundle/'point_semantics/point_semantics_manifest.json'
 if sm.exists():
  semantic=_hugsim_expand(json.loads(sm.read_text()));checks['full_19class_semantic_maps']=len(semantic.get('maps',[]))==995 and semantic.get('source_checkpoint_sha256')==source['source_checkpoint_sha256']
 if pm.exists():
  point=_hugsim_expand(json.loads(pm.read_text()));checks['point_and_background_semantics']=len(point.get('frames',[]))==199
 result={'schema':'native_scene551_preparation_readiness_v1','scene':551,'checks':checks,'source_frames':199,'source_images':995,'recorded_lidar_points':sum(f['lidar']['points'] for f in source['frames_data']),'road_gaussians':source['road_points'],'background_gaussians':source['unclassified_background_points'],'original_3d_annotations':raw['original_3d_labels'],'hardware_clock':raw['clock'],'full_semantics_model_weights_available':(OMNI/'third_party/SegFormer/pretrained/segformer.b5.1024x1024.city.160k.pth').is_file(),'recorded_lidar_is_closed_loop_sensor':False,'optimizer_receives_scene_ground_truth':False,'cpu_data_preparation_complete':all(v for k,v in checks.items() if k not in ['full_19class_semantic_maps','point_and_background_semantics','semantic_coverage_and_occlusion_review','runtime_adopted_plan_native_bundle']),'native_evaluation_admitted':False,'closed_loop_launch_ready':False,'missing':[k for k,v in checks.items() if not v]}
 write_json(bundle/'readiness.json',result);print(json.dumps(result,indent=2));return result

def main():
 p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);a=p.parse_args();check(a.bundle)
if __name__=='__main__':main()
