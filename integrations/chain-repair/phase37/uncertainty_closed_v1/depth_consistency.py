"""Fresh observed obstacle geometry; terrain ambiguity and stale history stay risk.

Only current/past depth and ego poses enter. No scene/ID/ego-erasure exception.
The .5m observation error budget is explicit policy, not a certified bound.
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

import json
pass  # Path is provided by the portable boundary above
import numpy as np
from global_repair_v2.boundary_policy import BoundedDepthMemory
from depth_surface_geometry import ground_plane
from runtime_configuration import configured_reliability
from ground_reference import from_observation as ground_reference
CONFIG=_hugsim_expand(json.loads((Path(__file__).resolve().parent/'config.json').read_text()))

class DepthConsistencyMemory(BoundedDepthMemory):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.height_evidence={}
 def update(self,obs,info):
  now=float(info['timestamp']);scene=info['scene_token']
  if self.time is not None and (scene!=self.scene or now<=self.time):self.height_evidence={}
  original_xy,audit=super().update(obs,info)
  cfg=configured_reliability();lease=float(cfg.max_gap_seconds)
  if not np.isfinite(lease) or lease<=0:raise ValueError('Invalid shared depth observation lease')
  plane=ground_plane(obs,info) if obs.get('depth') and info.get('depth_encoding')=='expected_camera_z_v1' else None
  r=np.asarray(info['explicit_l2g_r_mat']);t=np.asarray(info['explicit_l2g_t'])
  budget=float(CONFIG['height_observation_error_budget_m']);minimum=float(CONFIG['minimum_non_ground_height_m'])
  if not np.isfinite([budget,minimum]).all() or budget<=0 or minimum<0:raise ValueError('Invalid height uncertainty policy')
  reference=ground_reference(obs,info,plane,budget)
  hard=[];risk=[];rows=[];braking_uncertainty=[]
  for key,cell in self.cells.items():
   age=now-cell['seen']
   if age< -1e-8:raise ValueError('Depth observation from the future')
   if plane is not None and abs(cell['seen']-now)<1e-8:
    local=(cell['point']-t)@r
    height=float(local[2]-np.r_[local[:2],1.]@plane)
    lower,upper=reference.bounds(local[None,:2]);supported=np.isfinite([lower[0],upper[0]]).all() and lower[0]<=upper[0]
    self.height_evidence[key]=dict(timestamp=now,height_m=height,lower_height_m=(float(local[2]-budget-upper[0]) if supported else None),upper_height_m=(float(local[2]+budget-lower[0]) if supported else None),ground_lower_m=(float(lower[0]) if supported else None),ground_upper_m=(float(upper[0]) if supported else None),ground_supported=bool(supported),old_single_plane_hard=bool(height-budget>minimum))
   evidence=self.height_evidence.get(key)
   fresh=age<=lease+1e-8
   elevated=evidence is not None and evidence.get('ground_supported',False) and evidence['lower_height_m']>minimum
   center=((np.asarray(key,dtype=float)+.5)*self.resolution-t)@r
   if np.linalg.norm(center[:2])>=self.max_range-.5:continue
   supported=fresh and elevated
   (hard if supported else risk).append(center[:2])
   if not supported and evidence is not None and evidence.get('old_single_plane_hard'):braking_uncertainty.append(center[:2])
   rows.append(dict(world_voxel=list(map(int,key)),last_seen_s=cell['seen'],age_s=float(age),observations=cell['count'],hard_geometry=bool(supported),
    reason='fresh_height_interval_above_ground' if supported else ('observation_lease_expired' if not fresh else 'height_interval_not_separated_from_ground'),height_evidence=evidence))
  self.height_evidence={k:e for k,e in self.height_evidence.items() if k in self.cells}
  def unique(points):
   return np.unique(np.round(np.asarray(points,dtype=float).reshape(-1,2),7),axis=0)
  hard=unique(hard);risk=unique(risk)
  # Preserve the pre-existing bounded unknown visibility constraints. Add the
  # ambiguous/expired depth cells to its existing uncertainty-cost channel.
  unknown=audit['unknown_visibility_geometry']
  existing=np.asarray(unknown.get('epistemic_only_points_xy',[]),dtype=float).reshape(-1,2)
  combined=unique(np.concatenate([existing,risk]))
  unknown.update(epistemic_only_points_xy=combined.tolist(),epistemic_only_points=len(combined),depth_uncertainty_points=len(risk),depth_uncertainty_policy='height_ambiguous_or_expired_history_retained_as_risk',free_space_asserted=False)
  audit['ground_reference']=reference.audit()
  audit['height_ambiguity_braking_points_xy']=unique(braking_uncertainty).tolist()
  audit.update(points=len(hard),unfiltered_points=len(original_xy),all_depth_cells=len(self.cells),
    depth_consistency_policy='local_ground_interval_and_braking_viability_v2',hard_support_max_age_s=lease,height_observation_error_budget_m=budget,observation_error_bound_certified=False,
    height_interval_minimum_m=minimum,ambiguous_or_expired_cells=len(risk),cell_evidence=rows,
    clearing='original repeated free-space clearing retained; expiry changes evidence tier, never declares free space',
    forecast='fresh supported surfaces hard within observation lease; historical location is not a motion forecast')
  return hard,audit
