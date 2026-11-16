"""Read-only motion-head export around the existing frozen UniAD worker."""
import json,sys
from common import ROOT,CR,configure,write
configure();sys.path.insert(0,str(CR/'phase46/framework_driver_v1'))
from runtime_bundle import activate_configured
activate_configured()
import worker as frozen_worker

original_serve=frozen_worker.serve

def trace_serve(model,namespace):
    import numpy as np
    core=getattr(model,'module',model);original=core.motion_head.forward_test
    def observed(*args,**kwargs):
        result=original(*args,**kwargs)
        values=result[1];arrays={}
        for key in ['all_traj_preds','all_traj_scores','track_scores']:
            value=values.get(key)
            if value is not None:arrays[key]=value.detach().cpu().numpy().copy()
        track=kwargs.get('outs_track',args[1] if len(args)>1 else {})
        boxes=track.get('track_bbox_results')
        if boxes:
            for key,value in zip(['motion_query_boxes','motion_query_scores','motion_query_labels','motion_query_bbox_index'],boxes[0][:4]):
                value=getattr(value,'tensor',value)
                if hasattr(value,'detach'):arrays[key]=value.detach().cpu().numpy().copy()
        folder=max(frozen_worker.worker.JOB.glob('step_[0-9][0-9][0-9]'))
        np.savez_compressed(folder/'motion_head_raw.npz',**arrays)
        write(folder/'motion_head_export.json',dict(read_only=True,original_forward_result_returned=True,
            raw_decoder_layers_preserved=True,sdc_last_query_in_all_traj_arrays=True,
            query_association='motion_query_* arrays from the same original forward call; SDC is last',
            prediction_coordinate_semantics='unmodified native motion-head output; not scenario truth',keys=list(arrays)))
        return result
    core.motion_head.forward_test=observed
    try:return original_serve(model,namespace)
    finally:core.motion_head.forward_test=original

if __name__=='__main__':
    frozen_worker.serve=trace_serve;frozen_worker.install();frozen_worker.worker.main()
