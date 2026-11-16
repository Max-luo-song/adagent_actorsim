
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
import json,pickle,time,numpy as np
from run_driver import configure
from runtime_configuration import load_vehicle_config
configure();folder=CR/'phase31/runs/depth_consistency551_20261105T051937Z_fc796426/closed_loop/uniad_worker/step_070'
with (folder/'observation.pkl').open('rb') as stream:obs,info=pickle.load(stream)
del obs
with np.load(folder/'perception.npz') as saved:plan=saved['trajectory'].copy();occupancy=saved['occupancy'].copy()
rel=_hugsim_expand(json.loads((folder/'guard_memory_report.json').read_text()));v=load_vehicle_config();started=time.perf_counter();result=h.optimize(plan,occupancy,info,warm=None,maxiter=220,reliability=rel,vehicle=v);wall=time.perf_counter()-started
(HERE/'cpu_replay_result.json').write_text(json.dumps(result,indent=2)+chr(10));chosen=result.get('selected');assert chosen is not None and chosen['feasible'] and chosen['minimum_constraint']>=-1e-5
states=np.asarray(chosen['states']);assert float(states[:,3].max())>2.1
summary=dict(passed=True,selected=True,maximum_speed_mps=float(states[:,3].max()),minimum_constraint=chosen['minimum_constraint'],guard_speed_policy=result['guard']['speed_policy'],wall_seconds=wall,gpu_executed=False,new_closed_loop_executed=False)
(HERE/'cpu_replay_summary.json').write_text(json.dumps(summary,indent=2)+chr(10));print(json.dumps(summary,indent=2),flush=True)
