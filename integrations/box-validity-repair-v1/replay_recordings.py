"""Frozen-observation counterfactuals, NOT new closed-loop scores."""

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

import argparse,importlib.util,json,pickle,sys,time
pass  # Path is provided by the portable boundary above
import numpy as np
LOCAL=Path(__file__).resolve().parent;ROOT=LOCAL.parents[1];P26=ROOT/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26';OUT=P26/'box_validity_repair_v1';sys.path.insert(0,str(ROOT/'integrations/cross-scene-validation-v2'))
from bindings import bind
from evidence_packet import build
CASES={'551_original':P26/'patch_task_discovery_v1/pair/episodes/original_on/A','551_cutin':P26/'patch_task_discovery_v1/pair/episodes/edited_on/B','172_v15':P26/'cross_scene_validation_v2/cases/172/v15/A','172_native':P26/'cross_scene_validation_v2/cases/172/off/A','552_v15':P26/'cross_scene_validation_v2/cases/552/v15/A'}

def read(p):return _hugsim_expand(json.loads(Path(p).read_text()))
def run(case):
 folder=CASES[case];work=folder.parent;report=read(folder/'run_report.json')
 if case.startswith('551_'):
  sys.path[:0]=[str(LOCAL),str(ROOT/'integrations/light-map551-v4'),str(ROOT/'integrations/light-cutin-diagnostic')]
  import common
  common.configure()
  sys.path[:0]=[str(LOCAL),str(ROOT/'integrations/light-map551-v4'),str(ROOT/'integrations/light-cutin-diagnostic')]
 else:common,_,_,_,_=bind(work,'on')
 # All frozen parent dependencies and scene geometry are already bound; replace only the candidate class.
 sys.path.insert(0,str(LOCAL));spec=importlib.util.spec_from_file_location('_new_observation_candidate',LOCAL/'joint_safety.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 from scene_api import SceneSpec
 from tracker import Tracker
 from live_env import Motion
 from reference_adapter import control
 v=SceneSpec.read(work/'scene_manifest.json').vehicle;planner=module.JointReference(v,read(work/'map/map.json'));tracker=Tracker(v);rows=[]
 for i,o in enumerate(report['observations']):
  step=folder/f'uniad_worker/step_{i:03d}'
  if not (step/'perception.npz').exists():continue
  with np.load(step/'perception.npz') as z:arrays={k:z[k].copy() for k in ['trajectory','boxes','scores','labels','candidate_boxes','candidate_scores','candidate_labels','geometry_candidate_boxes']}
  with (step/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
  packet=build(arrays,read(step/'neural_query_origin.json'),read(step/'global_evidence.json'),read(step/'geometry_refinement.json'),info,v)
  x,z,yaw,speed,steer=o['motion_state'];m=Motion(v.dt,v.wheelbase_m,speed);m.x=x;m.z=z;m.yaw=yaw;m.steer=steer;m.timestamp=o['timestamp']
  if i:
   m.last_accel=report['trace'][i-1]['action']['acc_mps2'];m.last_rate=report['trace'][i-1]['action']['steer_rate_radps']
  began=time.monotonic();adopted,selection=planner.process(arrays['trajectory'],packet,{},info,m)
  final=i==len(report['observations'])-1 and report['status']=='completed'
  if final:action=None;gate={};not_admitted=False
  else:
   nominal,c=control(tracker,planner,adopted,info,m);action,gate=planner.admit(m,nominal);not_admitted=action is None
  rows.append({'index':i,'actual_recorded_speed':speed,'selection':selection['mode'],'hard_boxes':len(planner.actor_boxes()[0]),'qualified':int(packet['validated_memory_mask'].sum()),'weak_retained':len(planner._weak_geometry),'no_action':not_admitted,'action':action,'gate':gate.get('selected'),'adopted_changed_vs_raw':bool(np.max(np.abs(adopted-arrays['trajectory']))>1e-5),'seconds':time.monotonic()-began})
  print(case,i,'qualified',rows[-1]['qualified'],'hard',rows[-1]['hard_boxes'],'no_action',not_admitted,flush=True)
  (OUT/(case+'_replay.json')).write_text(json.dumps({'case':case,'rows':rows,'same_observation_counterfactual_only':True,'actual_closed_loop_gain_not_measured':True,'GPU_executed':False},indent=2)+'\n')
 result={'case':case,'frames':len(rows),'no_action_frames':[x['index'] for x in rows if x['no_action']],'terminal_old_reason':report['termination_reason'],'terminal_candidate_no_action':rows[-1]['no_action'],'rows':rows,'same_observation_counterfactual_only':True,'actual_closed_loop_gain_not_measured':True,'GPU_executed':False}
 (OUT/(case+'_replay.json')).write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':p=argparse.ArgumentParser();p.add_argument('case',choices=CASES);run(p.parse_args().case)
