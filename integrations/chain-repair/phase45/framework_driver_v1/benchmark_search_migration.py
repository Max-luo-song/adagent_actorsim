"""Fresh sequential paired solves on the same immutable framework observations."""

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
import argparse,hashlib,json,pickle,time
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,nargs='+',default=[10,11]);parser.add_argument('--variants',nargs='+',choices=['baseline','candidate'],default=['baseline','candidate']);args=parser.parse_args()
from migration_contract import validate
validate(HERE)
import sys
sys.path.insert(0,str(HERE.parent/'bridge/closed_loop_integration'))
import run_driver
run_driver.configure()
from geometry_acceleration import accelerated
accelerated.install()
from proposal_geometry_audit import install
install(run_driver.optimized_controller.PlanningGuard)
from empty_controller import optimize
from occupancy_protocol import unpack
from runtime_configuration import load_vehicle_config
reference=HERE.parent/'motion_envelope_v2';source=HERE.parent/'bridge/closed_loop_integration/runs/20261107T064934Z/uniad_worker';v=load_vehicle_config()
def digest(f):return hashlib.sha256(f.read_bytes()).hexdigest()
for index in args.steps:
 folder=source/f'step_{index:03d}';guard_path=reference/f'envelope_evidence/step_{index:03d}_guard.json';paths=[folder/'sensor_metadata.pkl',folder/'perception.npz',guard_path]
 if index:paths.append(source/f'step_{index-1:03d}/optimization.json')
 pins={str(f):digest(f) for f in paths}
 for variant in args.variants:
  out=HERE/'search_benchmarks'/variant;out.mkdir(parents=True,exist_ok=True);target=out/f'step_{index:03d}_result.json'
  if target.exists():raise FileExistsError('Refusing to overwrite a completed measurement: '+str(target))
  with (folder/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
  with np.load(folder/'perception.npz') as z:plan=z['trajectory'].copy();occ,contract=unpack(z)
  info['occupancy_contract']=contract;report=_hugsim_expand(json.loads(guard_path.read_text()));previous=_hugsim_expand(json.loads((source/f'step_{index-1:03d}/optimization.json').read_text())).get('selected') if index else None;warm=None if previous is None else np.array(previous['controls']);policy=_hugsim_expand(json.loads((HERE/'runtime_policy.json').read_text()));policy['adaptive_consensus']=variant=='candidate'
  print('START',index,variant,flush=True);start=time.monotonic();cpu=time.process_time();result=optimize(plan,occ,info,warm=warm,maxiter=220,reliability=report,vehicle=v,_warm_elapsed_s=.1,_runtime_policy=policy);wall=time.monotonic()-start;cpu=time.process_time()-cpu
  assert all(digest(Path(f))==d for f,d in pins.items()),'Input changed during solve'
  target.write_text(json.dumps(result,indent=2)+'\n');selected=result.get('selected');summary=dict(step=index,variant=variant,wall_seconds=wall,process_cpu_seconds=cpu,selected=selected is not None,forecast_recovery=result.get('forecast_recovery'),objective=None if selected is None else selected['objective'],minimum_constraint=None if selected is None else selected['minimum_constraint'],first_control=None if selected is None else selected['controls'][0],trials=len(result.get('trials',[])),iteration_limit=220,input_sha256=pins,search_audit=result.get('guard',{}).get('search_schedule'),dac_proxy=None if selected is None else selected['native_alignment_proxy']['dac_proxy'],new_gpu_executed=False)
  (out/f'step_{index:03d}_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
