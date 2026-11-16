
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
import argparse,json,pickle,time,numpy as np
from run_driver import configure
from runtime_configuration import load_vehicle_config
from occupancy_protocol import unpack
parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,nargs='+',default=[0]);args=parser.parse_args();configure();v=load_vehicle_config();source=CR/'phase34/runs/global_control551_v2_20261106T141946Z_4f9bf305/closed_loop/uniad_worker';out=HERE/'depth_behavior_replays/551';rows=[]
for index in args.steps:
 folder=source/('step_%03d'%index)
 with (folder/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
 with np.load(folder/'perception.npz') as z:occ,contract=unpack(z);plan=z['trajectory'].copy()
 info['occupancy_contract']=contract;report=_hugsim_expand(json.loads((out/(folder.name+'_guard.json')).read_text()));start=time.perf_counter();result=h.optimize(plan,occ,info,reliability=report,vehicle=v,maxiter=120);selected=result.get('selected');(out/(folder.name+'_result.json')).write_text(json.dumps(result,indent=2)+'\n');row=dict(step=index,selected=selected is not None,first_action=(selected or {}).get('controls',[None])[0],minimum_constraint=(selected or {}).get('minimum_constraint'),forecast_recovery=result.get('forecast_recovery'),solve_seconds=time.perf_counter()-start);rows.append(row);(out/'solve_summary.json').write_text(json.dumps(dict(rows=rows,scope='Fresh CPU solves on old sensor observations; not a new closed loop.',new_gpu_executed=False),indent=2)+'\n');print(json.dumps(row),flush=True)
