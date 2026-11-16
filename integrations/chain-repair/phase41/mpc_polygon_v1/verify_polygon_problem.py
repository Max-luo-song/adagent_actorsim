
# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

from bootstrap import *
import hashlib,json,pickle,numpy as np
from run_driver import configure
from occupancy_protocol import unpack
from runtime_configuration import load_vehicle_config
import empty_controller
configure();rows=[];rng=np.random.default_rng(714)
class Captured(Exception):pass
for scene,indices in [('phase39',(0,1,2,3,4,8,9)),('phase37',(2,))]:
 source=CR/('phase39/runs/mpc_ten551_20261107T112411Z_83f645be/closed_loop/uniad_worker' if scene=='phase39' else 'phase37/runs/uncertainty_closed551_20261107T044006Z_bc47a553/closed_loop/uniad_worker')
 for index in indices:
  folder=source/('step_%03d'%index)
  with (folder/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
  with np.load(folder/'perception.npz') as z:occ,contract=unpack(z);reference=z['trajectory'].copy()
  info['occupancy_contract']=contract;report=_hugsim_expand(json.loads((folder/'guard_memory_report.json').read_text()));report['cruise_speed_preference_mps']=8.;record=_hugsim_expand(json.loads((folder/'optimization.json').read_text()));x=np.asarray(record['selected']['controls']).ravel();functions=[]
  for compiled in (False,True):
   captured={}
   def capture(fun,x0,**kwargs):
    captured.update(objective=fun,constraints=kwargs['constraints'][0]['fun']);raise Captured()
   original=empty_controller.minimize;empty_controller.minimize=capture
   try:empty_controller.optimize(reference,occ,info,reliability=report,vehicle=load_vehicle_config(),maxiter=220,_runtime_policy=dict(compiled_road=True,compiled_polygon=compiled,consensus_search=True,adaptive_consensus=True,scaled_slsqp=False))
   except Captured:pass
   finally:empty_controller.minimize=original
   assert captured;functions.append(captured)
  maximum=0.;count=0;bitwise=True
  for _ in range(2):
   base=x+rng.normal(0.,1e-5,len(x))
   for j in range(len(x)+1):
    q=base.copy()
    if j<len(x):q[j]+=np.sqrt(np.finfo(float).eps)
    for name in ('objective','constraints'):
     a=np.asarray(functions[0][name](q));b=np.asarray(functions[1][name](q));np.testing.assert_array_equal(a,b);bitwise=bitwise and a.tobytes()==b.tobytes();count+=a.size
  rows.append(dict(source=scene,step=index,scalar_values_compared=count,exactly_equal=True,bitwise_equal=bitwise))
summary=dict(passed=True,rows=rows,scope='Actual saved objectives and full constraints at selected controls plus finite-difference perturbations; no new GPU execution.')
(HERE/'polygon_problem_equivalence.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
