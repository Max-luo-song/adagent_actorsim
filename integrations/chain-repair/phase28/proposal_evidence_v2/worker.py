"""Frozen detector; optional invalid historical-query cache retirement.

Current detections and all external obstacle memory are untouched.
Fresh baseline or repair replay only; no snapshot/restore of policy state.
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

import inspect,json,pickle,sys,types
pass  # Path is provided by the portable boundary above
HERE=Path(__file__).resolve().parent;P28=HERE.parent;ROOT=P28.parents[5]
sys.path[:0]=[str(HERE),str(P28/'heading_fix_v1'),str(ROOT/'waymo23-native-import'),str(ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad')]
if __name__=='__main__':
    from runtime_bundle import activate_configured
    activate_configured()
import numpy as np
import origin_worker
from lifecycle import QueryLifecycle,retire_queries
original_serve=origin_worker.original_serve

def serve(model,namespace):
    core=getattr(model,'module',model);original_update=core.scene_reliability.update
    original_forward=core._forward_single_frame_inference;signature=inspect.signature(original_forward)
    lifecycle=None
    pending=[]
    mode=_hugsim_expand(json.loads((origin_worker.worker.JOB/'lifecycle_mode.json').read_text()))['mode']
    if mode not in ('baseline','repair'):raise ValueError('Unknown lifecycle mode')
    def forward(self,*args,**kwargs):
        bound=signature.bind(*args,**kwargs);bound.apply_defaults()
        before=bound.arguments['track_instances'];counter=int(core.track_base.max_obj_id)
        incoming_ids=set(before.obj_idxes.detach().cpu().tolist())
        used=sorted(set(pending)&incoming_ids) if mode=='repair' else []
        bound.arguments['track_instances']=retire_queries(before,used)
        current=max(origin_worker.worker.JOB.glob('step_[0-9][0-9][0-9]'))
        origin_worker.worker.atomic_json(current/'query_lifecycle_transition.json',dict(mode=mode,retired_previous_cache_ids=used,
            before_rows=len(before),after_rows=len(bound.arguments['track_instances']),tracker_counter_unchanged=(counter==core.track_base.max_obj_id),
            current_detections_filtered=False,bev_reset=False))
        pending.clear()
        return original_forward(**bound.arguments)
    def update(*args,**kwargs):
        nonlocal lifecycle
        report=original_update(*args,**kwargs)
        current=max(origin_worker.worker.JOB.glob('step_[0-9][0-9][0-9]'))
        q=_hugsim_expand(json.loads((current/'neural_query_origin.json').read_text()))
        with (current/'observation.pkl').open('rb') as f:obs,info=pickle.load(f)
        raw=np.asarray(q['decoded_gravity_boxes'],float).copy();raw[:,2]-=raw[:,5]/2
        if lifecycle is None:lifecycle=QueryLifecycle(report['config'])
        labels=args[2]
        if hasattr(labels,'detach'):labels=labels.detach().cpu().numpy()
        audit=lifecycle.observe(raw,q['selected_track_ids'],labels,obs,info)
        pending[:]=audit['retire_next_forward']
        audit['mode']=mode
        origin_worker.worker.atomic_json(current/'query_lifecycle_audit.json',audit)
        return report
    core._forward_single_frame_inference=types.MethodType(forward,core);core.scene_reliability.update=update
    try:return original_serve(model,namespace)
    finally:core._forward_single_frame_inference=original_forward;core.scene_reliability.update=original_update

if __name__=='__main__':
    origin_worker.original_serve=serve
    origin_worker.worker.serve=origin_worker.serve
    origin_worker.worker.main()
