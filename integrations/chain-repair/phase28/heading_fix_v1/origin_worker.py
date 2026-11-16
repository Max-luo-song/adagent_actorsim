"""Isolated inference diagnostic wrapper; original files and weights unchanged."""

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
import json
pass  # Path is provided by the portable boundary above
import sys
import types

P28=Path(__file__).resolve().parents[1]
ROOT=P28.parents[5]
sys.path.insert(0,str(ROOT/'waymo23-native-import'))
if __name__=='__main__':
    # The wrapped worker is imported as a module, so its original __main__
    # early bootstrap is not executed. Preserve it before numerical imports.
    from runtime_bundle import activate_configured
    activate_configured()
import numpy as np
from origin_state import extend_capture,extend_apply
sys.path.insert(0,str(P28.parent/'phase27'))
import branch_worker as worker

original_serve=worker.serve

def serve(model, namespace):
    core=getattr(model,'module',model)
    original_forward=core._forward_single_frame_inference
    original_decode_method=core._det_instances2results
    signature=inspect.signature(original_forward)
    context={}
    import branch_state
    original_capture=branch_state.capture_worker_state
    original_apply=branch_state.apply_worker_state
    branch_state.capture_worker_state=extend_capture(original_capture)
    branch_state.apply_worker_state=extend_apply(original_apply)

    def forward(self,*args,**kwargs):
        bound=signature.bind(*args,**kwargs);bound.apply_defaults()
        path=worker.JOB/'origin_mode.json'
        mode=_hugsim_expand(json.loads(path.read_text()))['mode'] if path.exists() else 'baseline'
        if mode not in ('baseline','drop_bev','drop_tracks','drop_both'):
            raise ValueError('Invalid diagnostic mode')
        if mode in ('drop_bev','drop_both'):bound.arguments['prev_bev']=None
        if mode in ('drop_tracks','drop_both'):
            bound.arguments['track_instances']=self._generate_empty_tracks()
        incoming=bound.arguments['track_instances']
        ids=incoming.obj_idxes.detach().cpu().numpy()
        # Actual forward concatenates inactive/fresh before active instances.
        context.clear()
        context.update(mode=mode,source_order=np.r_[np.flatnonzero(ids<0),np.flatnonzero(ids>=0)],
            incoming_ids=ids.copy(),prev_bev_present=bound.arguments['prev_bev'] is not None,
            tracker_max_obj_id_before=int(self.track_base.max_obj_id))
        return original_forward(**bound.arguments)

    def decode(self,instances,results,img_metas):
        original=self.bbox_coder.decode
        evidence={}
        def captured(predictions,*args,**kwargs):
            decoded=original(predictions,*args,**kwargs)
            ids=predictions['obj_idxes'].detach().cpu()
            record=decoded[0]
            selected=record['bbox_index'].detach().cpu()
            mask=record['mask'].detach().cpu()
            selected=selected[mask]
            full_ids=instances.obj_idxes.detach().cpu().numpy()
            retained_full_indices=np.flatnonzero(full_ids!=-2)
            if len(retained_full_indices)!=len(ids):raise RuntimeError('Unexpected candidate decoder selection; cannot map query origin')
            selected_full=retained_full_indices[selected.numpy()]
            source=context['source_order'][selected_full]
            old_ids=context['incoming_ids'][source]
            evidence.update(decode_input_has_special_ego=bool((ids==-2).any()),
                selected_track_ids=ids[selected].tolist(),selected_filtered_instance_indices=selected.tolist(),
                selected_full_instance_indices=selected_full.tolist(),
                selected_forward_input_indices=source.tolist(),
                selected_preupdate_track_ids=old_ids.tolist(),
                selected_source_kind=['historical_active_track' if i>=0 else 'fresh_or_inactive_query' for i in old_ids],
                prev_bev_present=context['prev_bev_present'],
                tracker_max_obj_id_before=context['tracker_max_obj_id_before'],
                decoded_gravity_boxes=record['bboxes'].detach().cpu().tolist(),
                note='Track IDs are neural IDs, distinct from scene_reliability evidence_track_id. Indices are per-frame and not persistent query identities.')
            return decoded
        self.bbox_coder.decode=captured
        try:result=original_decode_method(instances,results,img_metas)
        finally:self.bbox_coder.decode=original
        # The serial worker creates only the current output directory before
        # forward. The coordinator moves case outputs away before restoring.
        current=max(worker.JOB.glob('step_[0-9][0-9][0-9]'))
        worker.atomic_json(current/'neural_query_origin.json',evidence)
        return result

    core._forward_single_frame_inference=types.MethodType(forward,core)
    core._det_instances2results=types.MethodType(decode,core)
    try:return original_serve(model,namespace)
    finally:
        core._forward_single_frame_inference=original_forward
        core._det_instances2results=original_decode_method
        branch_state.capture_worker_state=original_capture
        branch_state.apply_worker_state=original_apply

if __name__=='__main__':
    worker.serve=serve
    worker.main()
