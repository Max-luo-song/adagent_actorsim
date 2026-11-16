"""Persist the external evidence contract alongside the full model snapshot."""
import hashlib
import numpy as np
KEY='global_evidence_pipeline_state_v1'
COUNTER='origin_runtime_tracker_v1'

def extend_capture(original,pipeline):
    def capture(core,**kwargs):
        payload,meta=original(core,**kwargs);package=pipeline.capture();payload[KEY]=package
        meta[KEY]=dict(kind='global_evidence_pipeline',sha256=package['sha256'],bytes=len(package['blob']))
        return payload,meta
    return capture

def extend_apply(original,pipeline):
    def apply(core,payload):
        if KEY not in payload or COUNTER not in payload:raise ValueError('Incomplete global snapshot: evidence history or tracker counter missing')
        state=pipeline.validate_state(payload[KEY])
        if state['count']:
            adapter=payload.get('adapter',{})
            if adapter.get('scene_token')!=state['observer']['scene'] or adapter.get('timestamp')!=state['observer']['time']:
                raise ValueError('Model and evidence snapshot episode/clock differ')
        counter=payload[COUNTER]['max_obj_id']
        if isinstance(counter,bool) or not isinstance(counter,int) or counter<0:raise ValueError('Invalid tracker counter')
        if state['observer']['records'] and max(state['observer']['records'])>=counter:raise ValueError('Evidence ID history exceeds tracker counter')
        tracks=payload.get('test_track_instances')
        if tracks is not None:
            ids=np.asarray(tracks['fields']['obj_idxes']['data']);active=ids[ids>=0]
            if len(active) and int(active.max())>=counter:raise ValueError('Tracker counter would reuse an existing neural ID')
        # All newly added state is validated before the original model restore.
        result=original(core,payload);pipeline.restore(payload[KEY]);return result
    return apply
