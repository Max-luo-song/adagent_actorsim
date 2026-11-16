"""Frozen resident inference, external evidence admission after decoding."""

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

import json,pickle,sys
pass  # Path is provided by the portable boundary above
HERE=Path(__file__).resolve().parent;P28=HERE.parent;ROOT=P28.parents[5]
sys.path[:0]=[str(HERE),str(P28/'heading_fix_v1'),str(ROOT/'waymo23-native-import'),str(ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad')]
if __name__=='__main__':
    from runtime_bundle import activate_configured
    activate_configured()
import numpy as np
import origin_worker
from evidence import EvidenceHistory
from admission import apply
original_serve=origin_worker.original_serve

def serve(model,namespace):
    core=getattr(model,'module',model);original_update=core.scene_reliability.update
    observer=None
    mode=_hugsim_expand(json.loads((origin_worker.worker.JOB/'admission_mode.json').read_text()))['mode']
    if mode not in ('baseline','admission'):raise ValueError('Unknown admission mode')
    def update(*args,**kwargs):
        nonlocal observer
        report=original_update(*args,**kwargs)
        current=max(origin_worker.worker.JOB.glob('step_[0-9][0-9][0-9]'))
        q=_hugsim_expand(json.loads((current/'neural_query_origin.json').read_text()))
        with (current/'observation.pkl').open('rb') as f:obs,info=pickle.load(f)
        if observer is None:observer=EvidenceHistory(report['config'])
        raw=np.asarray(q['decoded_gravity_boxes'],float).copy();raw[:,2]-=raw[:,5]/2
        labels=args[2].detach().cpu().numpy() if hasattr(args[2],'detach') else args[2]
        evidence=observer.observe(raw,q['selected_track_ids'],labels,obs,info)
        evidence.update(timestamp=float(info['timestamp']),scene_token=info['scene_token'])
        result=apply(report,evidence,q['selected_track_ids'],info) if mode=='admission' else report
        origin_worker.worker.atomic_json(current/'admission_evidence.json',evidence)
        origin_worker.worker.atomic_json(current/'admission_audit.json',result.get('proposal_admission',dict(quarantined=[],neural_state_modified=False,raw_neural_outputs_modified=False)))
        return result
    core.scene_reliability.update=update
    try:return original_serve(model,namespace)
    finally:core.scene_reliability.update=original_update

if __name__=='__main__':
    origin_worker.original_serve=serve
    origin_worker.worker.serve=origin_worker.serve
    origin_worker.worker.main()
