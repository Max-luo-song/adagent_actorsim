"""Frozen full-scene experiment scope; controller v3 remains immutable."""

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

pass  # Path is provided by the portable boundary above
import hashlib,json

def validate_scope(task,scope,spec,arm):
 if task.get('driver')!='framework_memory_corridor551_geometry_v4' or task.get('validation')!='full_scene_19p8s_v1':raise ValueError('Wrong full-scene driver/scope')
 if spec.frames!=199 or float(spec.fps)!=10. or spec.source.get('scene')!=551:raise ValueError('Unexpected source scene/frame clock')
 if arm['start_frame']!=0 or scope['steps']!=spec.frames-1 or scope['max_wall_seconds']!=43200:raise ValueError('Full-scene coverage/deadline changed')
 if arm.get('initial_speed_mps')!=8. or arm.get('arm')!='edited':raise ValueError('Driving conditions changed')
 return True

def verify(bundle):
 bundle=Path(bundle).resolve();pins=_hugsim_expand(json.loads((bundle/'snapshot_manifest.json').read_text()))
 for name,expected in pins.items():
  f=bundle/name
  if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=expected:raise RuntimeError('Frozen full-scene adapter changed: '+name)
 scope=_hugsim_expand(json.loads((bundle/'experiment_scope.json').read_text()))
 if scope['allowed_controls']!=[30,198] or scope['expected_source_frames']!=199 or scope['wall_caps']!={'30':7200,'198':43200}:raise RuntimeError('Full-scene admission changed')
 return True

def bindings(bundle):
 """Import without allocating CUDA; reject shadowed renderer/editor modules."""
 import sys,importlib
 bundle=Path(bundle).resolve();bridge=bundle/'bridge'
 sys.path[:0]=[str(bridge/'closed_loop_integration'),str(bridge/'unified'),str(bridge/'framework_backend'),str(bundle/'framework_repo')]
 result={}
 for name,folder in [('scene_edit_session','unified'),('scene_api_fwbackend','framework_backend'),('backend','framework_backend'),('managed_assets','framework_backend'),('sensor_backend','closed_loop_integration')]:
  m=importlib.import_module(name);actual=Path(m.__file__).resolve()
  if actual.parent!=bridge/folder:raise RuntimeError('Full-scene module shadowed: '+name+' '+str(actual))
  result[name]=str(actual)
 backend=sys.modules['backend']
 if backend.F!=bundle/'framework_repo' or not (backend.ROOT/'code/HUGSIM').is_dir():raise RuntimeError('Invalid frozen renderer roots')
 return result
