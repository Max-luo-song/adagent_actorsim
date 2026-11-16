"""Exact search-only patch contract over our frozen envelope/memory/road fork."""

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

import hashlib,json
pass  # Path is provided by the portable boundary above
BEFORE="""        from consensus_search import ConsensusSearch
        consensus=ConsensusSearch(POLICY,max(0.,aligned.cruise_preference-vehicle.emergency_braking_mps2*POLICY['replan_period_s']),aligned.stop)"""
AFTER="""        if runtime_policy.get('adaptive_consensus',True):
            from consensus_search import ConsensusSearch
            quality_speed=min(aligned.cruise_preference,speed)
        else:
            from baseline_consensus import ConsensusSearch
            quality_speed=aligned.cruise_preference
        consensus=ConsensusSearch(POLICY,max(0.,quality_speed-vehicle.emergency_braking_mps2*POLICY['replan_period_s']),aligned.stop)"""
PRESERVED=('time_local_motion.py','motion_tube_policy.json','motion_memory.py','motion_observer.py','expired_memory_guard.py','memory_visibility.py','memory_lifecycle_policy.json','native_objective.py','road_priority.py','road_priority_policy.json','depth_ambiguity.py','temporal_context.py','context_factory.py','speed_policy.py','native_contract.py','native_contract.json','mpc_schedule.py','fast_road.py','exact_queries.py','reference_geometry.py','road_soft_cost.py','worker.py','run_driver.py','protocol_harness.py')
def validate(root):
 p=Path(root);reference=p.parent/'motion_envelope_v2';receipt=_hugsim_expand(json.loads((p/'migration_receipt.json').read_text()))
 for name in PRESERVED:
  if (p/name).read_bytes()!=(reference/name).read_bytes():raise ValueError('Own fix or physical component changed: '+name)
 original=(reference/'empty_controller.py').read_text()
 if original.count(BEFORE)!=1 or (p/'empty_controller.py').read_text()!=original.replace(BEFORE,AFTER):raise ValueError('Controller changed beyond the declared search patch')
 if (p/'baseline_consensus.py').read_bytes()!=(reference/'consensus_search.py').read_bytes():raise ValueError('Baseline search differs from our frozen version')
 for name,expected in receipt['migrated_sha256'].items():
  if hashlib.sha256((p/name).read_bytes()).hexdigest()!=expected:raise ValueError('Migrated phase40 search changed: '+name)
 policy=_hugsim_expand(json.loads((reference/'runtime_policy.json').read_text()));policy['adaptive_consensus']=True
 if _hugsim_expand(json.loads((p/'runtime_policy.json').read_text()))!=policy:raise ValueError('Unexpected runtime policy change')
 return True
