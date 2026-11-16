
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

import argparse,sys
pass  # Path is provided by the portable boundary above
from common_eval import *
from source_loader import load

def run(scene,stage,device):
 os.environ['GENERALIZATION_SCENE']=str(scene);bundle=OUT/'assets'/str(scene)
 if stage=='data':
  from scene_data import prepare,RETARGET_AUDIT
  if (bundle/'source_manifest.json').exists():
   m=read(bundle/'source_manifest.json')
   if m['scene']!=scene:raise ValueError('Mismatched prepared scene')
   from pipeline import verify_bundle
   return verify_bundle(scene,bundle)
  if bundle.exists():raise ValueError('Partial data preparation retained: '+str(bundle))
  m=prepare(bundle);write(bundle/'retarget_audit.json',RETARGET_AUDIT);return m
 if stage=='raw':
  if (bundle/'raw_metadata/raw_metadata.json').exists():
   from pipeline import verify_raw
   verify_raw(bundle,read(bundle/'source_manifest.json'));return read(bundle/'raw_metadata/raw_metadata.json')
  module=load('_raw_metadata_generic','raw_metadata.py');r=module.export(bundle)
  r['schema']='registered_scene_raw_metadata_v1';r['raw_processed_image_identity_all_images']=r.pop('raw_processed_image_identity_all_995_images');write(bundle/'raw_metadata/raw_metadata.json',r)
  if r['scene_index']!=scene or len(r['frames'])!=read(bundle/'source_manifest.json')['frames']:raise ValueError('Raw identity mismatch')
  return r
 if stage=='semantics':
  sys.path.insert(1,str(SOURCE));module=load('_semantic_export_generic','export_semantics.py');return module.export(bundle,device)
 if stage=='points':
  if (bundle/'point_semantics/point_semantics_manifest.json').exists():return read(bundle/'point_semantics/point_semantics_manifest.json')
  sys.path.insert(1,str(SOURCE));module=load('_point_semantics_generic','point_semantics.py');return module.transfer(bundle)
 raise ValueError(stage)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('scene',type=int);p.add_argument('stage',choices=['data','raw','semantics','points']);p.add_argument('--device',default='cuda');a=p.parse_args();r=run(a.scene,a.stage,a.device);print('ASSET_STAGE_DONE',a.scene,a.stage,flush=True)
