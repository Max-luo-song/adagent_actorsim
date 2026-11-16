"""A versioned native data view; legacy point semantics are never scored as v2."""
import numpy as np
from scene_contracts import *
from projection_domain import REVISION,POINT_FOLDER

def ensure_view(scene):
 source=ASSETS/str(scene);points=source/POINT_FOLDER;manifest=points/'point_semantics_manifest.json'
 if not manifest.exists():raise ValueError('Corrected principal-branch point semantics are required for native scoring: '+str(manifest))
 meta=read(manifest)
 if meta.get('scene')!=scene or meta.get('projection_revision')!=REVISION:raise ValueError('Wrong point semantics revision/scene')
 for key in ('source_manifest','image_semantic_manifest','projection_adapter'):verify(meta[key])
 if meta['source_manifest']!=asset(source/'source_manifest.json'):raise ValueError('Point semantics belong to another source')
 with np.load(points/'background_semantics.npz') as data:labels=data['class_id']
 if len(labels)!=len(np.load(source/'background_sr_world.npy',mmap_mode='r')) or not np.isin(labels,[*range(19),255]).all():raise ValueError('Invalid corrected point semantics geometry')
 view=NATIVE_ASSETS/str(scene);view.mkdir(parents=True,exist_ok=True)
 for original in source.iterdir():
  if not original.is_file():continue
  link=view/original.name
  if link.exists() or link.is_symlink():
   if link.resolve()!=original.resolve():raise ValueError('Native asset view mismatch')
  else:link.symlink_to(original.resolve())
 link=view/'point_semantics'
 if link.exists() or link.is_symlink():
  if link.resolve()!=points.resolve():raise ValueError('Native scoring view must not select legacy point semantics')
 else:link.symlink_to(points.resolve(),target_is_directory=True)
 write(view/'view_manifest.json',{'scene':scene,'projection_revision':REVISION,'source_manifest':meta['source_manifest'],'point_semantics':asset(manifest),'native_formula_changed':False,'legacy_point_semantics_selected':False})
 return view
