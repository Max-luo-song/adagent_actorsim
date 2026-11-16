"""Evidence capture only: never change constraints on the basis of a guess."""

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

import json
pass  # Path is provided by the portable boundary above

def diagnose_stop(report, output):
    if not report.get('trace'):return dict(category='no_control_trace',cause_verified=False)
    row=report['trace'][-1]; folder=Path(output)/'uniad_worker'/f"step_{row['step']:03d}"
    def read(name):
        p=folder/name
        return _hugsim_expand(json.loads(p.read_text())) if p.exists() else None
    optimization=read('optimization.json') or {}
    decision=row.get('control_decision') or {}
    reason=optimization.get('reason')
    category=('initial_constraint_conflict' if reason and 'fixed_initial_' in reason else
              'no_feasible_candidate' if optimization.get('selected') is None else
              'controller_or_selected_trajectory_stop')
    return dict(category=category,cause_verified=False,
        interpretation='Constraint reason is evidence, not proof of perception or model fault; inspect the saved predictions and trial constraints.',
        last_step=row['step'],elapsed_s=len(report['trace'])*report['dt'],
        threshold_mps=report['stationary_threshold_mps'],speed_before_mps=row['state_before'][3],
        speed_after_mps=row['state_after'][3],action=row['action'],optimization_reason=reason,
        control_decision=decision,neural_plan=row.get('neural_plan'),
        selected=optimization.get('selected'),trials=optimization.get('trials'),
        initial_conflict=optimization.get('initial_conflict'),guard=optimization.get('guard'),
        forecast_recovery=optimization.get('forecast_recovery'),
        evidence={name:str(folder/name) for name in ('optimization.json','control_decision.json',
            'guard_memory_report.json','predicted_drivable_audit.json','depth_geometry.json','sensor_metadata.json')},
        prediction_file=row.get('perception'),ground_truth_used_by_optimizer=False)
