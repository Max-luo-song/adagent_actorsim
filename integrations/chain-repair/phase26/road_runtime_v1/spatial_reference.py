"""Current neural geometry plus a soft entry-speed preference; no speed cap."""

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
POLICY=_hugsim_expand(json.loads(Path(__file__).with_name('reference_speed_policy.json').read_text()))
class CruisePreference:
 def __init__(self):self.scene=None;self.time=None;self.value=None
 def update(self,reference,info,report,vehicle):
  now=float(info['timestamp']);scene=info['scene_token'];plan=np.asarray(reference,dtype=float);reset=self.value is None or scene!=self.scene or self.time is not None and (now<=self.time or now-self.time>report.get('config',{}).get('max_gap_seconds',.75))
  if reset:
   segments=np.diff(np.vstack([np.zeros(2),plan]),axis=0)/.5
   self.value=min(vehicle.max_speed_mps,max(float(info['ego_velo']),float(np.median(np.linalg.norm(segments,axis=1)))))
  desired=report.get('requested_cruise_speed_mps',self.value)
  if not np.isfinite(desired) or desired<0:raise ValueError('Invalid soft cruise preference')
  self.scene=scene;self.time=now
  return min(float(desired),vehicle.max_speed_mps)

def spatial_error(points,plan,stop):
 points=np.asarray(points,dtype=float);path=np.vstack([np.zeros(2),np.asarray(plan)]);segments=np.diff(path,axis=0);length2=np.sum(segments**2,axis=1);keep=length2>1e-10
 if not np.any(keep):return float(np.mean(np.sum(points**2,axis=1)))
 starts=path[:-1][keep];segments=segments[keep];length2=length2[keep];delta=points[:,None]-starts[None];u=np.sum(delta*segments[None],axis=2)/length2[None];u=np.clip(u,0.,1.)
 if not stop:u[:,-1]=np.maximum(0.,np.sum(delta[:,-1]*segments[-1],axis=1)/length2[-1])
 closest=starts[None]+u[:,:,None]*segments[None];d2=np.sum((points[:,None]-closest)**2,axis=2)
 # Exact nearest-segment distance. Tail direction is from this frame only.
 return float(np.mean(np.min(d2,axis=1)))
