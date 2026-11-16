"""Replay normal/caution driver state; no legacy teacher labels trusted blindly."""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

import argparse,copy,json,pickle
from types import SimpleNamespace
import numpy as np
from paths import *
from run_cutin import configure
EPISODES={'train_search':PHASE26/'bridge/closed_loop_integration/runs/20261107T145401Z','train_envelope':PHASE26/'bridge/closed_loop_integration/runs/20261107T124054Z','test_cutin':CR/'phase43/runs/cutin551_20261109T040010Z_1411461'}
def run(key):
 source=EPISODES[key];driver=configure();h=driver.h;out=HERE/'data_inputs'/key;out.mkdir(parents=True,exist_ok=False)
 from occupancy_protocol import unpack
 from forecast_memory import frame_health
 from drivable_area import from_saved
 from runtime_configuration import load_vehicle_config
 report=_hugsim_expand(json.loads((source/'run_report.json').read_text()));vehicle=load_vehicle_config();saved={}
 def capture(reference,occupancy,info,*args,**kwargs):saved.update(reference=copy.deepcopy(reference),occupancy=copy.deepcopy(occupancy),info=copy.deepcopy(info),kwargs=copy.deepcopy(kwargs));return copy.deepcopy(original)
 driver.base.boundary_optimize=capture
 for index,trace in enumerate(report['trace']):
  folder=source/'uniad_worker'/('step_%03d'%index);original=_hugsim_expand(json.loads((folder/'optimization.json').read_text()))
  with (folder/'sensor_metadata.pkl').open('rb') as handle:info=pickle.load(handle)
  with np.load(folder/'perception.npz') as z:
   plan=z['trajectory'].copy();occupancy,contract=unpack(z);road=from_saved(z);candidates=[z['candidate_'+k].copy() for k in ('boxes','scores','labels')]
   if 'geometry_candidate_boxes' in z:candidates[0]=z['geometry_candidate_boxes'].copy()
  info['occupancy_contract']=contract;info['perception_health']=frame_health(info['timestamp'],_hugsim_expand(json.loads((folder/'occupancy_head_audit.json').read_text())),plan,candidates,road is not None)
  reliability=_hugsim_expand(json.loads((folder/'guard_memory_report.json').read_text()));warm=None
  if index:
   last=_hugsim_expand(json.loads((source/'uniad_worker'/('step_%03d'%(index-1))/'optimization.json').read_text())).get('selected')
   if last and report['trace'][index-1]['control_decision']['state_after'] in ('normal','caution','recovering','degraded'):warm=np.asarray(last['controls'])
  state=trace['control_decision']['state_before']
  if state not in ('normal','caution'):raise ValueError('Recovery state requires an actual state snapshot')
  h._active_recovery_controller=SimpleNamespace(state=state);saved.clear();h.optimize(plan,occupancy,info,warm=warm,reliability=reliability,vehicle=vehicle,_reuse_verified_warm=False)
  if not saved:continue
  saved.update(source_episode=key,source_step=index,source_result=original,source_path=str(source))
  with (out/('step_%03d.pkl'%index)).open('wb') as handle:pickle.dump(saved,handle,protocol=4)
 print(key,len(list(out.glob('*.pkl'))),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('episode',choices=EPISODES);run(parser.parse_args().episode)
