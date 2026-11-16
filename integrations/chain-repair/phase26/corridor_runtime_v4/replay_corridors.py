"""Fresh offline counterfactual solves; never edit a running GPU bundle."""

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
import sys,argparse,json,pickle,time,hashlib
import numpy as np
sys.path.insert(0,str(HERE.parent/'bridge/closed_loop_integration'))
from run_driver import configure
configure()
from geometry_acceleration import accelerated
accelerated.install()
from empty_controller import optimize
from runtime_configuration import load_vehicle_config
from occupancy_protocol import unpack
parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,nargs='+',default=[0,7,29]);args=parser.parse_args();v=load_vehicle_config();source=HERE.parent/'bridge/closed_loop_integration/runs/20261107T145401Z/uniad_worker';out=HERE/'candidate_replays';out.mkdir(exist_ok=True)
for index in args.steps:
 folder=source/f'step_{index:03d}';record=_hugsim_expand(json.loads((folder/'optimization.json').read_text()));report=_hugsim_expand(json.loads((folder/'guard_memory_report.json').read_text()));report['cruise_speed_preference_mps']=record['guard']['native_alignment']['cruise_preference_mps'];report['unobserved_motion_budget_mps']=v.max_speed_mps
 with (folder/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
 with np.load(folder/'perception.npz') as z:plan=z['trajectory'].copy();occ,contract=unpack(z)
 info['occupancy_contract']=contract;previous=_hugsim_expand(json.loads((source/f'step_{index-1:03d}/optimization.json').read_text())).get('selected') if index else None;warm=None if previous is None else np.array(previous['controls']);pins={q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in HERE.glob('*.py')};print('START',index,flush=True);start=time.monotonic();result=optimize(plan,occ,info,warm=warm,reliability=report,vehicle=v,maxiter=220,_warm_elapsed_s=.1);wall=time.monotonic()-start
 assert all(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest for name,digest in pins.items()),'Implementation changed during replay'
 (out/f'step_{index:03d}_result.json').write_text(json.dumps(result,indent=2)+'\n');selected=result.get('selected');summary=dict(step=index,seconds=wall,selected=None if selected is None else selected['seed'],forecast_recovery=result.get('forecast_recovery'),minimum_constraint=None if selected is None else selected['minimum_constraint'],trials=len(result.get('trials',[])),new_gpu_executed=False,implementation_sha256=pins)
 (out/f'step_{index:03d}_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
 for r in result.get('trials',[]):print(r['seed'],r['feasible'],round(r['objective'],4),r['lateral_extent_m'],r['native_alignment_proxy'].get('road_loss'),r['native_alignment_proxy'].get('road_peak_violation_m2'),flush=True)
