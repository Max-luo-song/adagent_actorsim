"""Native evaluation stays post-hoc; missing scene truth never gets fake scores."""

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

import hashlib,importlib.util,json
pass  # Path is provided by the portable boundary above
import numpy as np
from native_contract import CONTRACT,comfort_terms
FIELDS=('nc','dac','ttc','c','pdms','rc','hdscore')
def evaluate(run):
 run=Path(run);report=_hugsim_expand(json.loads((run/'run_report.json').read_text()));plans=[]
 for path in sorted((run/'uniad_worker').glob('step_*/native_adopted_plan.json')):
  data=_hugsim_expand(json.loads(path.read_text()));traj=np.vstack([np.array(data['ego_box'])[[0,1,6]],np.asarray(data['planned_traj']['traj'])]);terms=comfort_terms(traj,data['planned_traj']['timestep']);plans.append(terms['native_flag'])
 complete_plan_coverage=bool(len(plans)==len(report.get('trace',[])) and plans)
 scores={key:None for key in FIELDS}
 if complete_plan_coverage:scores['c']=float(np.mean(plans))
 result=dict(status='partial' if complete_plan_coverage else 'unavailable',scores=scores,full_native_score_available=False,native_source_sha256=CONTRACT['source_sha256'],comfort_exported_frames=len(plans),executed_frames=len(report.get('trace',[])),complete_plan_coverage=complete_plan_coverage,comfort_implementation='Exact native kinematics and thresholds on exported adopted plans',missing=['canonical_native_ground_and_scene_semantics','time_aligned_native_actor_boxes','canonical_native_route_completion'],ground_truth_used_by_optimizer=False)
 reference=Path(__file__).with_name('scene_data_reference.json')
 if reference.exists():
  ref=_hugsim_expand(json.loads(reference.read_text()));data_root=Path(ref['bundle'])
  if all(hashlib.sha256((data_root/name).read_bytes()).hexdigest()==digest for name,digest in ref['artifact_sha256'].items()):
   readiness=_hugsim_expand(json.loads(Path(ref['readiness']).read_text()));result['scene_data_preparation']=readiness
  else:result['scene_data_preparation']={'status':'changed_source_data_reference','native_evaluation_admitted':False}
 bundle=run/'native_eval';manifest=bundle/'native_export_manifest.json';required=[bundle/name for name in ('data.pkl','ground.ply','scene.ply')]
 if not manifest.exists() or not all(f.is_file() for f in required):return result
 meta=_hugsim_expand(json.loads(manifest.read_text()))
 checks=('scene_and_actor_semantics_verified','coordinate_and_clock_verified','vehicle_dimensions_verified','adopted_control_plan_source_verified')
 if meta.get('native_source_sha256')!=CONTRACT['source_sha256'] or not all(meta.get(k) is True for k in checks):
  result.update(status='invalid_native_export_manifest',missing=['Verified canonical native export contract']);return result
 root=next(p for p in run.parents if (p/'code/HUGSIM').is_dir());source=root/CONTRACT['source']
 if hashlib.sha256(source.read_bytes()).hexdigest()!=CONTRACT['source_sha256']:raise ValueError('Native scorer source changed')
 try:
  spec=importlib.util.spec_from_file_location('official_hugsim_native_score',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  if not module.torch.cuda.is_available():
   result.update(status='native_gpu_evaluation_pending',missing=['CUDA for unmodified native scene collision scoring']);return result
  official=module.calculate(module.parse_data(str(bundle)))
  for key in FIELDS:
   value=float(official[key])
   if not np.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid native score '+key)
  result.update(status='complete',scores={key:float(official[key]) for key in FIELDS},full_native_score_available=True,missing=[],official_details=official.get('details'))
 except Exception as exc:result.update(status='native_evaluation_error',error=repr(exc),full_native_score_available=False)
 return result
