
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
from scipy.spatial.transform import Rotation

OUT=Path(__file__).resolve().parent;RUN=OUT.parent
report=_hugsim_expand(json.loads((RUN/'report.json').read_text()))
cases=_hugsim_expand(json.loads((RUN/'commands.json').read_text()))['cases']
checks={};checks['four_gpu_cases_completed']=report['complete'] and report['gpu_render_executed'] and len(cases)==4
checks['original_visual_quality_flag_preserved']=report['visual_quality_accepted'] is False
checks['source_implementation_matches']=all(Path(p).is_file() and hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in report['implementation_sha256'].items())
commands={name:c for name,c in cases}
for name,command in cases:
 folder=RUN/name;journal=_hugsim_expand(json.loads((folder/'journal.json').read_text()))
 checks[name+'_one_exact_insert']=len(journal)==1 and journal[0]['command']==command and journal[0]['edit_revision']==1
 checks[name+'_three_camera_images']=all((folder/f'{cam}.png').is_file() for cam in ('CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT'))
 checks[name+'_target_layers_present']=all((folder/f'{kind}.png').is_file() for kind in ('target','target_gray','target_alpha','without_target'))
 checks[name+'_target_visible']=bool((np.asarray(Image.open(folder/'target_alpha.png'))>51).any())
base=np.asarray(commands['source_original']['asset_to_world']);supported=np.asarray(commands['source_supported_view']['asset_to_world'])
checks['source_supported_changes_only_diagnostic_heading']=commands['source_original']['asset']==commands['source_supported_view']['asset'] and np.array_equal(base[:3,3],supported[:3,3]) and np.allclose(supported[:3,:3].T@supported[:3,:3],np.eye(3),atol=1e-6)
ref=np.asarray(commands['reference_realcar']['asset_to_world']);rot=np.asarray(commands['reference_realcar_yaw15']['asset_to_world'])
checks['reference_same_asset_and_position_plus_15deg']=commands['reference_realcar']['asset']==commands['reference_realcar_yaw15']['asset'] and np.array_equal(ref[:3,3],rot[:3,3]) and np.allclose(rot[:3,:3],Rotation.from_euler('z',15,degrees=True).as_matrix()@ref[:3,:3],atol=1e-9)
checks['supported_view_matches_recorded_source_mean']=abs(report['view']['supported_view_degrees']-report['view']['source_view_target_degrees'])<1e-5
other=np.array(Image.open(RUN/'source_original/without_target.png'))
checks['remaining_scene_layer_identical_all_cases']=all(np.array_equal(other,np.array(Image.open(RUN/name/'without_target.png'))) for name,_ in cases)
files={str(f.relative_to(RUN)):hashlib.sha256(f.read_bytes()).hexdigest() for f in RUN.rglob('*') if f.is_file() and OUT not in f.parents}
(OUT/'original_sha256.json').write_text(json.dumps(files,indent=2))
summary=dict(passed=all(checks.values()),checks=checks,sealed_files=len(files),
 visual_review=dict(source_original='severely blurred and incomplete',source_supported_view='much more recognizable side view; front/edges still imperfect',
 reference_realcar='substantially clearer body, windows and rear; reviewed only this pose',reference_realcar_yaw15='substantially clearer after 15deg rotation; reviewed only this pose'),
 source_asset_reconstructed=False,full_novel_view_quality_certified=False,formal_evaluation_admitted=False)
(OUT/'review.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));assert summary['passed']
