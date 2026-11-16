"""Frozen perception; filter one declared neural lineage before guard memory."""

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
import sys
HERE=Path(__file__).resolve().parent
P28=HERE.parent
ROOT=P28.parents[5]
sys.path[:0]=[str(HERE),str(P28/'heading_fix_v1'),str(ROOT/'waymo23-native-import')]
if __name__=='__main__':
    from runtime_bundle import activate_configured
    activate_configured()
import numpy as np
import origin_worker
from policy import filter_report

original_serve=origin_worker.original_serve

def serve(model,namespace):
    core=getattr(model,'module',model)
    original_update=core.scene_reliability.update
    def update(*args,**kwargs):
        report=original_update(*args,**kwargs)
        current=max(origin_worker.worker.JOB.glob('step_[0-9][0-9][0-9]'))
        query=_hugsim_expand(json.loads((current/'neural_query_origin.json').read_text()))
        index=int(current.name.split('_')[1])
        if index<39:
            old=P28/f'runs/heading_fix_scene552_v1/uniad_worker/step_{index:03d}/perception.npz'
            with np.load(old) as data:
                before=np.asarray(query['decoded_gravity_boxes'])
                expected=data['candidate_boxes_gravity']
                if before.shape!=expected.shape or np.abs(before-expected).max()>1e-4:
                    raise RuntimeError('Pre-intervention perception prefix differs at '+str(index))
        filtered=filter_report(report,query['selected_track_ids'])
        origin_worker.worker.atomic_json(current/'lineage92_filter.json',filtered['lineage92_counterfactual'])
        return filtered
    core.scene_reliability.update=update
    try:return original_serve(model,namespace)
    finally:core.scene_reliability.update=original_update

if __name__=='__main__':
    origin_worker.original_serve=serve
    origin_worker.worker.serve=origin_worker.serve
    origin_worker.worker.main()
