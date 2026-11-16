"""Extra state for NEW diagnostic snapshots; old branch snapshots unchanged."""
import hashlib
import json

KEY='origin_runtime_tracker_v1'

def capture_runtime_tracker(core):
    tracker=core.track_base
    return dict(max_obj_id=int(tracker.max_obj_id),
        config={k:getattr(tracker,k) for k in ('score_thresh','filter_score_thresh','miss_tolerance')})

def restore_runtime_tracker(core, state):
    config={k:getattr(core.track_base,k) for k in state['config']}
    if config!=state['config']:raise ValueError('Tracker configuration changed')
    value=state['max_obj_id']
    if isinstance(value,bool) or not isinstance(value,int) or value<0:raise ValueError('Invalid tracker ID counter')
    core.track_base.max_obj_id=value

def extend_capture(original):
    def capture(core,**kwargs):
        payload,meta=original(core,**kwargs)
        payload[KEY]=capture_runtime_tracker(core)
        meta[KEY]=dict(kind='origin_runtime_tracker',state=payload[KEY],
            sha256=hashlib.sha256(json.dumps(payload[KEY],sort_keys=True).encode()).hexdigest())
        return payload,meta
    return capture

def extend_apply(original):
    def apply(core,payload):
        if KEY not in payload:raise ValueError('Diagnostic restore requires complete new snapshot; replay from frame0 instead of guessing an old counter')
        restore_runtime_tracker(core,payload[KEY])
        return original(core,payload)
    return apply
