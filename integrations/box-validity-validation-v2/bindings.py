"""Bind frozen off/v15 and observation-contract repair to identical live worlds."""

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

import ast,importlib.util,sys,types
pass  # Path is provided by the portable boundary above
import numpy as np
ROOT=Path(__file__).resolve().parents[2];LOCAL=Path(__file__).resolve().parent
PARENT=ROOT/'integrations/light-tracking-curvature-v15-r2';REPAIR=ROOT/'integrations/box-validity-repair-v1';CANON=ROOT/'integrations/light-execution-horizon-v17-r2';MAP=ROOT/'integrations/light-map551-v4';OLD=ROOT/'integrations/light-cutin-diagnostic';CROSS=ROOT/'integrations/cross-scene-validation-v2'
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def bind(work):
 work=Path(work);common=load('common',OLD/'common.py');plan=common.read(work/'plan.json');common.BUNDLE=Path(plan['scene_bundle']);common.configure()
 code=REPAIR if plan['validation_arm']=='repair' else PARENT
 sys.path[:0]=[str(code),str(LOCAL),str(CANON),str(MAP),str(OLD),str(CROSS)]
 import environment,tracker,local_boot,map_reference,joint_safety,reference_adapter,risk_memory,observation_evidence
 if Path(joint_safety.__file__).resolve()!=code/'joint_safety.py':raise ValueError('Controller binding mismatch; use a fresh episode process')
 switch=load('controller_switch',ROOT/'integrations/patch-v15-validation-r2-v1/controller_switch.py');switch.MODE='off' if plan['validation_arm']=='off' else 'on'
 source=(CANON/'runtime.py').read_text();changes=[]
 if plan['scene']!=551:
  adapter=load('_frozen_cross_bindings',CROSS/'bindings.py');tree,changes=adapter.runtime_tree(source,plan['scene'],plan['navigation_command'],plan['rigid_actor_count']);source=ast.unparse(tree)
  from support_map import install
  install(map_reference)
 tree=switch.rewrite_runtime(source)
 if plan['validation_arm']=='repair':
  from live_contract import rewrite
  tree=rewrite(ast.unparse(tree))
 runtime=types.ModuleType('runtime');runtime.__file__=str(CANON/'runtime.py');sys.modules['runtime']=runtime
 if plan['validation_arm']=='repair':
  from live_contract import attach
  runtime.attach_box_observation=attach
 def verify_original_actor_arrays(root,tracks):
  ref=Path(common.read(Path(root)/'plan.json')['actor_reference']);ref.parent.mkdir(parents=True,exist_ok=True)
  if not ref.exists():np.savez_compressed(ref,**tracks)
  with np.load(ref) as z:
   if set(z.files)!=set(tracks) or any(not np.array_equal(z[k],tracks[k]) for k in z.files):raise ValueError('Original actor arrays changed between arms')
  return {'passed':True,'all_original_actor_arrays_identical':True,'reference':common.asset(ref)}
 runtime.verify_original_actor_arrays=verify_original_actor_arrays
 exec(compile(tree,str(CANON/'runtime.py'),'exec'),runtime.__dict__)
 runtime.TERMINATION=dict(runtime.TERMINATION,simulation_observation_limit_s=plan['controls']*.1)
 scoring=load('scoring',MAP/'scoring.py')
 import native_scores
 native_scores.SCENE_BUNDLE=common.BUNDLE
 diagnostics=load('diagnostics',OLD/'diagnostics.py')
 return common,runtime,scoring,diagnostics,changes
