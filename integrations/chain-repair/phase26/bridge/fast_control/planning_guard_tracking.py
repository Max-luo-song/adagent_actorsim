"""planning_guard substitute: honest labeling for the tracking-passthrough arm.

Re-exports the reviewed waymo23-native-import/planning_guard.py unchanged,
except:
- POLICY is replaced so every control_decision.json records the real path
  instead of the MPC policy string;
- RecoveryController.update is subclassed to relabel reason 'feasible_plan'
  (hardcoded in the reviewed implementation) as 'tracking_passthrough_unverified'
  and to stamp decision['control_path'].

Known artifact, documented in README: the phase11 harness builds its trace
`mode` string as decision['state_after'] + ' / MPC' — that literal suffix is a
sealed-harness string and does NOT mean the MPC ran. Authoritative per-step
record: control_decision.json (policy / control_path / reason).
"""

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

import importlib.util
import sys
pass  # Path is provided by the portable boundary above

HERE = Path(__file__).resolve().parent
JOB = HERE.parents[3]
ROOT = JOB.parents[3]

_spec = importlib.util.spec_from_file_location(
    'planning_guard_reviewed', ROOT / 'waymo23-native-import/planning_guard.py')
_real = importlib.util.module_from_spec(_spec)
# Pickle resolves ObstacleMemory through its defining module at the harness's
# periodic checkpoint. exec_module alone does not make that module importable.
sys.modules[_spec.name] = _real
_spec.loader.exec_module(_real)

# Full public re-export (whitelist caused the third launch failure:
# phase11/drivable_area.py lazily imports rectangle_separation from
# planning_guard at call time). Only POLICY and RecoveryController differ.
for _name in dir(_real):
    if not _name.startswith('_') and _name not in ('POLICY', 'RecoveryController'):
        globals()[_name] = getattr(_real, _name)
del _name

POLICY = ('tracking_passthrough_no_obstacle_optimization_v1 '
          '(NOT the evidence-tiered MPC; no safety equivalence)')


class RecoveryController(_real.RecoveryController):
    def update(self, optimized, speed, timestamp, dt, braking):
        action, decision = super().update(optimized, speed, timestamp, dt, braking)
        if decision.get('reason') == 'feasible_plan':
            decision['reason'] = 'tracking_passthrough_unverified'
        decision['policy'] = POLICY  # the reviewed update() reads its own module global
        decision['control_path'] = POLICY
        return action, decision
