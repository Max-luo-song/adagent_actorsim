
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

import json,sys,numpy as np
pass  # Path is provided by the portable boundary above
p=Path('outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26/bridge/agent_json').resolve();sys.path.insert(0,str(p));import agent_bootstrap
from agent_catalog import Catalog
from scene_edit_session import saf
from ego_trajectory import compile_trajectory
from ego_execution import TrackingDriver,execution_plan,motion_pose
from ego_attitude import pose_for_sample
job=_hugsim_expand(json.loads((p/'examples/ego_driving_level.json').read_text()));args=job['steps'][2]['request']['args'];spec=saf.SceneSpec.read(Catalog().scene('552'))
c=compile_trajectory(args['trajectory'],spec);plan=execution_plan(c,spec.vehicle,args);driver=TrackingDriver(plan,spec.vehicle)
trace=[]
for i in range(40):
 row=driver.advance(i);nominal=motion_pose(plan['anchor'],driver.motion);pose=pose_for_sample(c,nominal,i+1)
 target=np.asarray(c['samples'][i+1]['ego_to_world']);delta=np.linalg.inv(target)@pose
 row.update(position_error_m=float(np.linalg.norm(delta[:3,3])),heading_error_degrees=float(abs(np.degrees(np.arctan2(delta[1,0],delta[0,0])))),
  body_up_error=float(np.linalg.norm(pose[:3,2]-c['gravity_up_source'])),ego_to_world=pose.tolist(),motion_ego_to_world=nominal.tolist());trace.append(row)
report=dict(passed=max(x['position_error_m'] for x in trace)<1 and max(x['heading_error_degrees'] for x in trace)<10 and max(x['body_up_error'] for x in trace)<1e-8,
 controls=40,attitude_policy=c['attitude_policy'],max_position_error_m=max(x['position_error_m'] for x in trace),final_position_error_m=trace[-1]['position_error_m'],
 max_heading_error_degrees=max(x['heading_error_degrees'] for x in trace),max_body_up_error=max(x['body_up_error'] for x in trace),gpu_executed=False,renderer_executed=False,trace=trace)
Path('/tmp/level_driving_cpu.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='trace'},indent=2));assert report['passed']
