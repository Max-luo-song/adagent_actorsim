"""Recheck historical frozen-weight edited predictions without training or inference."""
import json
import pickle
from pathlib import Path
import sys
import numpy as np
BASE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(BASE/'waymo23-native-import'),str(BASE/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad')]
from box_geometry import box_footprint
from evaluate_preview_vehicle import reference_polygon,polygon_iou


def dimensions(poly):
    edges=np.roll(poly,-1,axis=0)-poly
    lengths=np.linalg.norm(edges,axis=1)
    i=int(np.argmax(lengths))
    return np.array([lengths.min(),lengths.max()]),edges[i]/lengths[i]


def audit():
    plan=json.loads((BASE/'outputs/edit_vehicle_resolution_20261021/joint_plan_v2.json').read_text())
    manifest=json.loads(Path(plan['jobs'][0]['args']['sequence_report']).read_text())
    inference=BASE/'outputs/gpu_test_service/jobs'/plan['jobs'][0]['id']/'result'
    cases=[]
    for segment in plan['segments']:
        if segment['role']!='edited' or 'warmup' in str(segment['scene']):continue
        asset=json.loads(Path(segment['asset_report']).read_text())
        for index in range(segment['first_index'],segment['first_index']+segment['frames']):
            frame=manifest['frames'][index]
            with Path(frame['packet']).open('rb') as stream:_,info=pickle.load(stream)
            reference=reference_polygon(asset,info);ref_dims,ref_axis=dimensions(reference)
            for variant in plan['variants']:
                folder=inference/variant
                run=json.loads((folder/'run_report.json').read_text())
                arrays=np.load(folder/f'frame_{index:04d}/perception.npz')
                matches={}
                for prefix in ('candidate_',''):
                    values=[]
                    for i,(box,score,label) in enumerate(zip(arrays[prefix+'boxes'],arrays[prefix+'scores'],arrays[prefix+'labels'])):
                        if label!=0:continue
                        poly=box_footprint(box);iou=polygon_iou(reference,poly)
                        if iou<.25:continue
                        dims,axis=dimensions(poly)
                        error=box[:2]-reference.mean(0)
                        values.append(dict(index=i,raw_score=float(score),proxy_iou=float(iou),center_error_m=float(np.linalg.norm(error)),
                                           signed_xy_error_m=error.tolist(),undirected_heading_error_degrees=float(np.degrees(np.arccos(np.clip(abs(ref_axis@axis),0,1)))),
                                           predicted_width_length_m=dims.tolist(),reference_width_length_m=ref_dims.tolist(),dimension_ratio=(dims/ref_dims).tolist()))
                    matches['candidate' if prefix else 'retained']=sorted(values,key=lambda x:-x['raw_score'])
                cases.append(dict(scene=str(segment['scene']),variant=variant,frame=frame['frame'],
                                  historical_run_implementation_unchanged=run['implementation_unchanged_during_run'],
                                  diagnostic_preview_only=run['diagnostic_preview_only'],raw_scores_modified=run['raw_scores_modified'],
                                  historical_thresholds=run['thresholds'],**matches))
    report=dict(cases=cases,matching_iou=.25,proxy_matching_is_not_identity_certification=True,
                new_training=False,learned_adapters_used=False,new_inference=False,n1_certified=False,recognition_fully_solved=False)
    out=BASE/'outputs/edited_vehicle_causal_audit_20261021';(out/'audit.json').write_text(json.dumps(report,indent=2))
    for c in cases:
        if c['scene'] not in ('172_new','552_new'):continue
        m=c['candidate'][0] if c['candidate'] else None
        print(c['scene'],c['variant'],c['frame'],'retained',len(c['retained']),'best',m)

if __name__=='__main__':audit()
