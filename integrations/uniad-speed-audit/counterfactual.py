"""Offline-only state isolation and CAN-feature toggle; no parameter updates."""

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

import inspect
pass  # Path is provided by the portable boundary above
import json

def read(p):return _hugsim_expand(json.loads(Path(p).read_text()))
def write(p,v):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)

def wrap_forward(core,case_folder):
    from resident_session_state import reset_episode,snapshot
    original=core.forward_test
    switches=[(name,m) for name,m in core.named_modules() if hasattr(m,'can_bus_mlp') and hasattr(m,'use_can_bus')]
    if len(switches)!=1:raise ValueError('Unexpected CAN bus modules: '+str([n for n,m in switches]))
    def forward(*args,**kwargs):
        folder=case_folder();case=read(folder/'case.json')
        before=snapshot(core);reset_episode(core);cleared=snapshot(core)
        bad=[k for k in ['bev_present','tracking_present','metadata_bev_present','tracking_pose_present'] if cleared[k]]
        if bad or any(cleared[k]!=0 for k in ['track_next_id','reliability_tracks','reliability_next_id']):
            raise ValueError('Counterfactual did not start from a cold model state')
        for name,m in switches:m.use_can_bus=case['use_can_bus']
        bound=inspect.signature(original).bind(*args,**kwargs);meta=bound.arguments['img_metas']
        while isinstance(meta,list):meta=meta[0]
        bus=meta['can_bus'].copy()
        if float(bus[13])!=case['declared_speed_mps']:raise ValueError('Counterfactual CAN speed mismatch')
        write(folder/'speed_feature_audit.json',dict(case=case,before_reset=before,cold_state=cleared,
            incoming_can_bus=bus.tolist(),actual_runtime_gate={n:bool(m.use_can_bus) for n,m in switches},
            weights_changed=False,offline_same_image_counterfactual=True,driving_executed=False))
        return original(*args,**kwargs)
    core.forward_test=forward
    return original

