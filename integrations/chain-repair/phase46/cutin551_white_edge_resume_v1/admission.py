"""Offline release gate; quality, physical checks and measured aggregate speed."""

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

import json,math
pass  # Path is provided by the portable boundary above
from paths import HERE
STEPS=(0,1,10,19,22,23)
def evaluate(require_geometry=True):
 folder=HERE/'validation/current';rows=[]
 for step in STEPS:
  def load(variant,kind):return _hugsim_expand(json.loads((folder/('%s_%03d_%s.json'%(variant,step,kind))).read_text()))
  a,b=load('baseline','summary'),load('ranked','summary');base,new=load('baseline','result'),load('ranked','result');old,current=base['selected'],new['selected']
  if not a['source_plan_bitwise_equal']:raise ValueError('Original current input was not reconstructed exactly: '+str(step))
  if not old or not current or current['minimum_constraint']< -1e-5 or not current['feasible'] or new.get('forecast_recovery'):raise ValueError('Current-problem feasibility regression: '+str(step))
  if current['objective']>old['objective']+.01*max(abs(old['objective']),1.):raise ValueError('Objective quality regression: '+str(step))
  if current['native_alignment_proxy']['comfort_native_kinematic_flag']<old['native_alignment_proxy']['comfort_native_kinematic_flag']:raise ValueError('Comfort quality regression: '+str(step))
  rows.append(dict(step=step,baseline_seconds=a['wall_seconds'],ranked_seconds=b['wall_seconds'],baseline_cpu_seconds=a['process_cpu_seconds'],ranked_cpu_seconds=b['process_cpu_seconds'],baseline_trials=a['trials'],ranked_trials=b['trials'],baseline_objective=old['objective'],ranked_objective=current['objective'],objective_delta=current['objective']-old['objective'],baseline_first_control=old['controls'][0],ranked_first_control=current['controls'][0],minimum_constraint=current['minimum_constraint'],plan_changed=not b['source_plan_bitwise_equal']))
 total=lambda key:sum(row[key] for row in rows)
 improvement=1-total('ranked_seconds')/total('baseline_seconds')
 if improvement<.05:raise ValueError('No aggregate fixed-input speed benefit')
 if require_geometry:
  physical=_hugsim_expand(json.loads((folder/'ranked_independent_checks.json').read_text()))
  if not physical['passed'] or set(r['step'] for r in physical['rows'])!=set(STEPS):raise ValueError('Missing independent full-geometry/physics validation')
 result=dict(passed=True,source_inputs_reconstructed_bitwise=True,steps=list(STEPS),rows=rows,total_baseline_seconds=total('baseline_seconds'),total_ranked_seconds=total('ranked_seconds'),speedup_factor=total('baseline_seconds')/total('ranked_seconds'),time_reduction_fraction=improvement,maximum_frame_regression_seconds=max(r['ranked_seconds']-r['baseline_seconds'] for r in rows),full_physical_checks_passed=require_geometry,physical_check_scope="Native bicycle bounds, current sensed/predicted full geometry and full-envelope braking witness; no full-scene physical certification",acceptance_and_constraints_unchanged=True,search_order_changed=True,trajectory_equality_claimed=False,gpu_executed=False,closed_loop_speedup_certified=False,global_generalization_certified=False)
 return result
if __name__=='__main__':
 result=evaluate();(HERE/'performance_admission.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
