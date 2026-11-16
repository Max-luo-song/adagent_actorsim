"""Frozen NN inference with a single external evidence/state contract."""

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
HERE=Path(__file__).resolve().parent;CR=HERE.parents[1];ROOT=CR.parents[4]
sys.path[:0]=[str(HERE.parent),str(CR/'phase28/heading_fix_v1'),str(ROOT/'waymo23-native-import'),str(ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad')]
if __name__=='__main__':
    from runtime_bundle import activate_configured
    activate_configured()
import numpy as np
import origin_worker
from global_repair_v1.pipeline import EvidencePipeline
from global_repair_v1.snapshot import extend_capture,extend_apply
original_serve=origin_worker.original_serve

def array(value):
    if hasattr(value,'tensor'):value=value.tensor
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    return np.asarray(value)

def serve(model,namespace):
    from dataclasses import asdict
    import branch_state
    from runtime_configuration import configured_reliability
    core=getattr(model,'module',model);original_update=core.scene_reliability.update
    path=origin_worker.worker.JOB/'global_mode.json';mode=_hugsim_expand(json.loads(path.read_text()))['mode'] if path.exists() else 'global'
    if mode not in ('baseline','global'):raise ValueError('Unknown global pipeline mode')
    pipeline=EvidencePipeline(asdict(configured_reliability()),enabled=mode=='global')
    capture=branch_state.capture_worker_state;restore=branch_state.apply_worker_state
    branch_state.capture_worker_state=extend_capture(capture,pipeline)
    branch_state.apply_worker_state=extend_apply(restore,pipeline)
    def update(*args,**kwargs):
        report=original_update(*args,**kwargs)
        current=max(origin_worker.worker.JOB.glob('step_[0-9][0-9][0-9]'))
        query=_hugsim_expand(json.loads((current/'neural_query_origin.json').read_text()))
        if query['decode_input_has_special_ego']:raise ValueError('Special ego leaked to candidate export')
        with (current/'observation.pkl').open('rb') as f:obs,info=pickle.load(f)
        raw=np.asarray(query['decoded_gravity_boxes'],dtype=np.float32).copy();raw[:,2]-=raw[:,5]/2
        result=pipeline.process(report,raw,array(args[0]),array(args[1]),array(args[2]),query['selected_track_ids'],obs,info)
        origin_worker.worker.atomic_json(current/'global_evidence.json',result['global_evidence'])
        origin_worker.worker.atomic_json(current/'global_evidence_receipt.json',result['global_evidence_receipt'])
        return result
    core.scene_reliability.update=update
    try:return original_serve(model,namespace)
    finally:
        core.scene_reliability.update=original_update
        branch_state.capture_worker_state=capture;branch_state.apply_worker_state=restore

if __name__=='__main__':
    origin_worker.original_serve=serve
    origin_worker.worker.serve=origin_worker.serve
    origin_worker.worker.main()
