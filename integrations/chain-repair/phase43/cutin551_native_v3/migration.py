"""Whole-controller declared performance patch, with upstream repairs locked."""

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

import ast,hashlib,json
pass  # Path is provided by the portable boundary above
from paths import HERE,UPSTREAM,DRIVER,CR
PERFORMANCE=CR/'phase42/mpc_runtime_v3'
def expected_controller(source):
 s=source;helper=(PERFORMANCE/'empty_controller.py').read_text();a=helper.index('def _polygon_backend_scope(');b=helper.index('def optimize(',a);s=s.replace('def optimize(',helper[a:b]+'def optimize(',1)
 s=s.replace('    @memo.wrap\n    def simulate_controls','    from prefix_rollout import PrefixRollout\n    prefix=None\n    @memo.wrap\n    def simulate_controls',1).replace('        return rollout(controls, speed, steer, offset, vehicle.dt, vehicle.wheelbase_m,control_interval_s,interpolation=interpolation)','        return prefix(controls) if prefix is not None else rollout(controls, speed, steer, offset, vehicle.dt, vehicle.wheelbase_m,control_interval_s,interpolation=interpolation)',1)
 s=s.replace('    initial = np.array([-offset[0],-offset[1],0.,speed,steer])',"    prefix=PrefixRollout(speed,steer,offset,vehicle.dt,vehicle.wheelbase_m,control_interval_s,interpolation) if runtime_policy.get('rollout_prefix_reuse') else None\n    initial = np.array([-offset[0],-offset[1],0.,speed,steer])",1)
 s=s.replace("        result['runtime_performance_policy']=runtime_policy","        result['runtime_performance_policy']=runtime_policy\n        result['prefix_rollout_metrics']=dict(prefix.metrics) if prefix is not None else None",1)
 return s.replace('from mpc_reuse_v1.cached_geometry import EpistemicRisk','from polygon_backend import EpistemicRisk').replace('        road_cache=RoadPoseCache(road.clearance)',"        if runtime_policy.get('bulk_pose_cache'):\n            from bulk_road_cache import BulkRoadPoseCache as RoadPoseCache\n        road_cache=RoadPoseCache(road.clearance)",1)
def validate():
 receipt=_hugsim_expand(json.loads((HERE/'upstream_and_migration.json').read_text()))
 for name,digest in receipt['upstream_sha256'].items():
  if hashlib.sha256((UPSTREAM/name).read_bytes()).hexdigest()!=digest:raise ValueError('Upstream changed: '+name)
 if (DRIVER/'empty_controller.py').read_text()!=expected_controller((UPSTREAM/'empty_controller.py').read_text()):raise ValueError('Controller changed beyond declared performance wiring')
 preserved=[]
 exceptions={'empty_controller.py','empty_native_temporal.py','temporal_context.py','context_factory.py','run.sh','runtime_policy.json','readiness.json','validation.json','manifest.json','README.md'}
 for f in UPSTREAM.glob('*.py'):
  if f.name not in exceptions and f.read_bytes()!=(DRIVER/f.name).read_bytes():raise ValueError('Framework behavior module changed: '+f.name)
  if f.name not in exceptions:preserved.append(f.name)
 for name in ('temporal_context.py','context_factory.py'):
  expected=(UPSTREAM/name).read_text().replace('from mpc_reuse_v1.cached_geometry import CachedPolygon','from polygon_backend import CachedPolygon').replace('from mpc_reuse_v1.cached_geometry import EpistemicRisk','from polygon_backend import EpistemicRisk')
  if (DRIVER/name).read_text()!=expected:raise ValueError('Unexpected geometry wiring change: '+name)
 if (DRIVER/'empty_native_temporal.py').read_bytes()!=(PERFORMANCE/'empty_native_temporal.py').read_bytes():raise ValueError('Batched occupancy migration differs')
 policy=_hugsim_expand(json.loads((UPSTREAM/'runtime_policy.json').read_text()));policy.update(compiled_polygon=True,rollout_prefix_reuse=True,batched_occupancy=True,bulk_pose_cache=True)
 if _hugsim_expand(json.loads((DRIVER/'runtime_policy.json').read_text()))!=policy:raise ValueError('Unexpected runtime policy change')
 return dict(passed=True,behavior_modules_byte_preserved=preserved,whole_controller_patch_verified=True,ground_truth_used_by_optimizer=False)
