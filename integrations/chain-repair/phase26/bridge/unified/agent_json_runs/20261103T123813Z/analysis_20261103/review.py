
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

pass  # Path is provided by the portable boundary above
import json,hashlib
import numpy as np
from PIL import Image
OUT=Path(__file__).resolve().parent;RUN=OUT.parent
read=lambda p:_hugsim_expand(json.loads((RUN/p).read_text()))
a=read('job_results/acceptance.json');steps=read('job_results/steps.json')
before=steps[1]['response']['result'];edit=steps[2]['response']['result'];after=steps[3]['response']['result']
checks=dict(five_steps_completed=a['passed'] and a['completed_steps']==a['requested_steps']==5,
 all_requests_matched=all(s['matched'] and s['response']['status']=='ok' for s in steps),
 one_load_two_renders_no_reset=a['scene_loads']==1 and a['actual_renders']==2 and a['scene_resets']==0,
 first_insert_not_replayed=steps[2]['response']['replayed'] is False,
 exactly_one_revision=edit['receipt']['previous_revision']==0 and edit['receipt']['edit_revision']==after['edit_revision']==1 and before['edit_revision']==0,
 same_generation=before['history_token']==after['history_token']==edit['generation'],
 same_frame_pose=before['source_frame']==after['source_frame']==80 and np.array_equal(before['ego_to_world'],after['ego_to_world']) and before['timestamp_s']==after['timestamp_s']==8.,
 actual_preview_kept_editable=all(s['response']['context']['phase']=='pre_run' for s in steps if s['action']=='preview'),
 managed_car_present_only_after=not before['managed_objects'] and set(after['managed_objects'])=={'agent_car'},
 correct_reference_asset=edit['receipt']['command']['asset'].endswith('/data/3DRealCar/2024_06_04_13_44_39'),
 owner_closed=steps[4]['response']['result']['closed'] is True,
 no_training_or_driving=all(s['response']['training_performed'] is False and s['response']['control_executed'] is False for s in steps),
 actual_gpu_backend=a['gpu_render_executed'] and a['execution_backend']=='framework_renderer')
counts={}
for cam in before['cameras']:
 for name,packet in [('before',before),('after',after)]:
  c=packet['cameras'][cam];rgb=np.array(Image.open(c['rgb']));depth=np.load(c['depth'])
  checks[name+'_'+cam+'_hashes']=hashlib.sha256(rgb.tobytes()).hexdigest()==c['rgb_sha256'] and hashlib.sha256(depth.tobytes()).hexdigest()==c['depth_sha256']
  checks[name+'_'+cam+'_finite_depth']=np.isfinite(depth).all().item() and (depth>=0).all().item()
 x=np.array(Image.open(before['cameras'][cam]['rgb']));y=np.array(Image.open(after['cameras'][cam]['rgb']))
 counts[cam]=int(np.any(x!=y,axis=-1).sum())
 checks[cam+'_camera_unchanged']=np.array_equal(before['cameras'][cam]['K'],after['cameras'][cam]['K']) and np.array_equal(before['cameras'][cam]['camera_to_world'],after['cameras'][cam]['camera_to_world'])
checks['front_edit_visible']=counts['CAM_FRONT']>0
checks['source_implementation_matches']=all(Path(p).is_file() and hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in read('implementation.json').items())
files={str(f.relative_to(RUN)):hashlib.sha256(f.read_bytes()).hexdigest() for f in RUN.rglob('*') if f.is_file() and OUT not in f.parents}
(OUT/'original_sha256.json').write_text(json.dumps(files,indent=2))
summary=dict(passed=all(checks.values()),checks=checks,rgb_changed_pixels=counts,sealed_files=len(files),
 reset_executed=False,native_actor_edit_executed=False,committed_observation_executed=False,
 closed_loop_driving=False,formal_evaluation_admitted=False)
(OUT/'review.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));assert summary['passed']
