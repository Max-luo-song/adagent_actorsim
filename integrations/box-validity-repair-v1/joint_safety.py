"""v15 selection unchanged; only independently qualified geometry renews hard memory."""

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

import importlib.util,sys
pass  # Path is provided by the portable boundary above
import numpy as np
ROOT=Path(__file__).resolve().parents[2];PARENT=ROOT/'integrations/light-tracking-curvature-v15-r2'
spec=importlib.util.spec_from_file_location('_frozen_v15_geometry_parent',PARENT/'joint_safety.py');parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
for name in dir(parent):
 if not name.startswith('_'):globals()[name]=getattr(parent,name)
from evidence_packet import validate
Base=parent.JointReference
class JointReference(Base):
 supports_box_observation_contract=True
 require_box_observation_contract=False
 def update_tracks(self,arrays,info):
  if 'box_observation_contract' not in arrays:
   if self.require_box_observation_contract:raise ValueError('Live box observation contract missing; no legacy fallback')
   self._weak_geometry=np.empty((0,9));self.box_observation_audit={'mode':'legacy_fixture_contract'}
   return super().update_tracks(arrays,info)
  geometry,mask=validate(arrays,info)
  self._weak_geometry=geometry[~mask].copy()
  qualified=dict(arrays,boxes=geometry[mask].copy(),scores=np.asarray(arrays['scores'])[mask].copy(),labels=np.asarray(arrays['labels'])[mask].copy())
  self.box_observation_audit=dict(mode='qualified_observation_renews_hard_geometry',qualified=int(mask.sum()),weak_risk_retained=int((~mask).sum()),raw_neural_modified=False,free_space_asserted=False)
  # Existing confirmed/latching history is retained by the unchanged parent.
  # An unobserved decoder echo cannot update its position, timestamp or velocity.
  return super().update_tracks(qualified,info)
 def uncertainty_cost(self,states,boxes,tracks):
  weak=getattr(self,'_weak_geometry',np.empty((0,9)))
  if not len(weak):return super().uncertainty_cost(states,boxes,tracks)
  risk=np.concatenate([boxes,weak],axis=0)
  evidence=list(tracks)+[dict(last=self.clock,height_supported=False) for _ in weak]
  return super().uncertainty_cost(states,risk,evidence)
