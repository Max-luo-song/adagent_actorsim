"""Narrow scene plumbing around immutable controller, worker, tracker and scorer."""

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
OLD=ROOT/'integrations/light-cutin-diagnostic';MAP=ROOT/'integrations/light-map551-v4';CANON=ROOT/'integrations/light-execution-horizon-v17-r2'

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def runtime_tree(source,scene,command,expected_count):
 # Replace only declared scene metadata, navigation and the actor-identity gate.
 tree=ast.parse(source);changes=[]
 class Retarget(ast.NodeTransformer):
  def visit_keyword(self,node):
   self.generic_visit(node)
   if node.arg=='scene' and isinstance(node.value,ast.Constant) and node.value.value==551:node.value=ast.Constant(scene);changes.append('scene')
   if node.arg in ('command','nav_command') and isinstance(node.value,ast.Constant) and node.value.value==2:node.value=ast.Constant(command);changes.append(node.arg)
   return node
  def visit_If(self,node):
   self.generic_visit(node)
   text=ast.unparse(node.test)
   if text=="name == 'B'":
    return ast.parse("report['actor_sync']=verify_original_actor_arrays(root,tracks)").body[0]
   if text=="report['rigid_actor_count'] != 15":node.test.comparators[0]=ast.Constant(expected_count);changes.append('actor_count')
   return node
 tree=ast.fix_missing_locations(Retarget().visit(tree))
 if sorted(changes)!=sorted(['scene','command','nav_command','actor_count']):raise ValueError('Runtime adaptation source contract changed: '+str(changes))
 return tree,changes

def bind(work,mode):
 work=Path(work);sys.path[:0]=[str(LOCAL),str(CANON),str(MAP),str(OLD)]
 common=load('common',OLD/'common.py');plan=common.read(work/'plan.json');common.BUNDLE=Path(plan['scene_bundle']);common.configure()
 sys.path[:0]=[str(LOCAL),str(CANON),str(MAP),str(OLD)]
 import environment,tracker,local_boot
 import map_reference
 from support_map import install
 install(map_reference)
 code=Path(plan['candidate_code']);sys.path[:0]=[str(code),str(LOCAL),str(CANON),str(MAP),str(OLD)]
 import joint_safety,reference_adapter,risk_memory,observation_evidence
 if Path(joint_safety.__file__).resolve()!=code/'joint_safety.py':raise ValueError('Wrong frozen controller')
 switch=load('controller_switch',ROOT/'integrations/patch-v15-validation-r2-v1/controller_switch.py');switch.MODE=mode
 source=(CANON/'runtime.py').read_text();tree,changes=runtime_tree(source,plan['scene'],plan['navigation_command'],plan['rigid_actor_count'])
 # The existing audited off/on switch changes exactly one controller import.
 source=ast.unparse(tree);tree=switch.rewrite_runtime(source)
 runtime=types.ModuleType('runtime');runtime.__file__=str(CANON/'runtime.py');sys.modules['runtime']=runtime
 def verify_original_actor_arrays(root,tracks):
  root=Path(root);ref=Path(common.read(root/'plan.json')['actor_reference']);ref.parent.mkdir(parents=True,exist_ok=True)
  if not ref.exists():np.savez_compressed(ref,**tracks)
  with np.load(ref) as z:
   if set(z.files)!=set(tracks) or any(not np.array_equal(z[k],tracks[k]) for k in z.files):raise ValueError('Scene actor geometry changed across independent arms')
  return dict(passed=True,all_original_actor_arrays_identical=True,reference=common.asset(ref))
 runtime.verify_original_actor_arrays=verify_original_actor_arrays
 exec(compile(tree,str(CANON/'runtime.py'),'exec'),runtime.__dict__)
 runtime.TERMINATION=dict(runtime.TERMINATION,simulation_observation_limit_s=plan['controls']*.1)
 scoring=load('scoring',MAP/'scoring.py')
 import native_scores
 native_scores.SCENE_BUNDLE=common.BUNDLE
 diagnostics=load('diagnostics',OLD/'diagnostics.py')
 return common,runtime,scoring,diagnostics,changes
