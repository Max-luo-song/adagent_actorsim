"""Admit replacement source assets by coverage only, before any controller score."""
import ast,importlib.util,sys,types,time
import numpy as np
from scene_contracts import *
from projection_domain import REVISION,POINT_FOLDER,wrap_point_projector
SOURCE=P26.parent/'phase33/scene_data551_v1'
EXTRA={s:P26.parent/'phase14/prepared'/str(s)/'source_report.json' for s in (31,233)}

def context(scene):
 report=read(REPORTS[scene]);cached=ASSETS/str(scene)/'source_manifest.json';meta=read(cached) if cached.exists() else None;raw=Path(meta['raw_data'] if meta else report['data_dir']);checkpoint=Path(report['checkpoint']);count=len(list((raw/'ego_pose').glob('*.txt')))
 if report['scene']!=scene or count!=report['frames'] or not checkpoint.exists():raise ValueError('Replacement scene identity/coverage mismatch')
 return raw,checkpoint,count

def load(scene,name,file):
 raw,checkpoint,count=context(scene);path=SOURCE/file;tree=ast.parse(path.read_text())
 class Retarget(ast.NodeTransformer):
  def visit_Constant(self,node):
   if file=='point_semantics.py' and node.value=='point_semantics':return ast.copy_location(ast.Constant(value=POINT_FOLDER),node)
   if type(node.value) is int and node.value in (551,199,995):return ast.copy_location(ast.Constant(value={551:scene,199:count,995:count*5}[node.value]),node)
   return node
 module=types.ModuleType(name);module.__file__=str(path);sys.modules[name]=module;exec(compile(ast.fix_missing_locations(Retarget().visit(tree)),str(path),'exec'),module.__dict__)
 if file=='scene_data.py':module.RAW=raw;module.CHECKPOINT=checkpoint
 if file=='point_semantics.py':module.project=wrap_point_projector(module.project,module.CV_TO_SR)
 return module

def stage(scene,kind,device='cuda'):
 raw,checkpoint,count=context(scene);bundle=ASSETS/str(scene)
 # Export helpers import scene_data by its original module name. Bind the
 # retargeted data module before semantic/point modules execute.
 data=load(scene,'scene_data','scene_data.py');sys.path.insert(0,str(SOURCE))
 if kind=='data':
  if not (bundle/'source_manifest.json').exists():
   if bundle.exists():bundle.rename(bundle.with_name(bundle.name+'_incomplete_'+str(time.time_ns())))
   return data.prepare(bundle)
  m=read(bundle/'source_manifest.json')
  if m['scene']!=scene or m['frames']!=count or m['raw_data']!=str(raw) or m['source_checkpoint']!=str(checkpoint):raise ValueError('Replacement scene cache mismatch')
  for name,h in m['artifacts'].items():verify({'path':str(bundle/name),'sha256':h})
  return m
 if kind=='raw':
  file=bundle/'raw_metadata/raw_metadata.json'
  if file.exists():
   m=read(file)
   if m['scene_index']!=scene or len(m['frames'])!=count or not m['raw_processed_pose_identity_all_frames'] or not m['raw_processed_calibration_identity'] or not m['raw_processed_image_identity_all_images']:raise ValueError('Replacement raw identity is unverified')
   return m
  partial=bundle/'raw_metadata'
  if partial.exists():partial.rename(bundle/('raw_metadata_incomplete_'+str(time.time_ns())))
  module=load(scene,'_cross_raw','raw_metadata.py');result=module.export(bundle);result['schema']='registered_scene_raw_metadata_v1';result['raw_processed_image_identity_all_images']=result.pop('raw_processed_image_identity_all_995_images');write(file,result);return result
 if kind=='semantics':
  file=bundle/'semantic19/semantic_manifest.json'
  if file.exists():
   m=read(file)
   if len(m['maps'])!=count*5 or m['source_checkpoint_sha256']!=read(bundle/'source_manifest.json')['source_checkpoint_sha256']:raise ValueError('Wrong/incomplete cached semantic identity')
   for row in m['maps']:
    for key in ('file','confidence_file','margin_file'):verify({'path':str(bundle/'semantic19'/row[key]),'sha256':row['sha256' if key=='file' else key.replace('_file','_sha256')]})
   return m
  return load(scene,'_cross_semantics','export_semantics.py').export(bundle,device)
 if kind=='points':
  file=bundle/POINT_FOLDER/'point_semantics_manifest.json'
  if file.exists():
   m=read(file)
   if len(m['frames'])!=count or m.get('projection_revision')!=REVISION or m.get('scene')!=scene:raise ValueError('Incomplete or stale point semantics')
   for binding in ('source_manifest','image_semantic_manifest','projection_adapter'):verify(m[binding])
   return m
  partial=bundle/POINT_FOLDER
  if partial.exists():partial.rename(bundle/('point_semantics_incomplete_'+str(time.time_ns())))
  result=load(scene,'_cross_points','point_semantics.py').transfer(bundle);result.update(scene=scene,projection_revision=REVISION,source_manifest=asset(bundle/'source_manifest.json'),image_semantic_manifest=asset(bundle/'semantic19/semantic_manifest.json'),projection_adapter=asset(LOCAL/'projection_domain.py'),legacy_point_semantics_reused=False);write(file,result);return result
 raise ValueError(kind)
