"""Reference-only A/B on immutable historical predictions; does not improve detections."""
import json,pickle
from pathlib import Path
import numpy as np
from asset_calibration import load,rebase_report
from asset_geometry import source_corners
from evaluate_preview_vehicle import reference_polygon,polygon_iou
import sys
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad'))
from box_geometry import box_footprint


def analyze():
    calibration_path=BASE/'waymo23-native-import/configs/asset_calibrations/2024_06_04_13_44_39-support-v1.json'
    calibration=json.loads(calibration_path.read_text());load(calibration_path,calibration['asset'])
    plan=json.loads((BASE/'outputs/edit_vehicle_resolution_20261021/joint_plan_v2.json').read_text())
    manifest=json.loads(Path(plan['jobs'][0]['args']['sequence_report']).read_text())
    inference=BASE/'outputs/gpu_test_service/jobs'/plan['jobs'][0]['id']/'result';rows=[]
    out=BASE/'outputs/asset_calibration_20261021'
    for seg in plan['segments']:
        if seg['role']!='edited' or str(seg['scene']) not in ('172_new','552_new','172_baseline_proxy','552_baseline_proxy'):continue
        old=json.loads(Path(seg['asset_report']).read_text());cal=rebase_report(old,calibration)
        (out/(str(seg['scene'])+'_calibrated_asset_report.json')).write_text(json.dumps(cal,indent=2))
        for index in range(seg['first_index'],seg['first_index']+seg['frames']):
            record=manifest['frames'][index]
            with Path(record['packet']).open('rb') as f:_,info=pickle.load(f)
            for variant in plan['variants']:
                arrays=np.load(inference/variant/f'frame_{index:04d}/perception.npz')
                result=dict(scene=str(seg['scene']),frame=record['frame'],variant=variant)
                for name,asset in [('old',old),('calibrated',cal)]:
                    reference=reference_polygon(asset,info);matches={}
                    for prefix in ('candidate_',''):
                        values=[]
                        for i,(b,score,label) in enumerate(zip(arrays[prefix+'boxes'],arrays[prefix+'scores'],arrays[prefix+'labels'])):
                            if label!=0:continue
                            iou=polygon_iou(reference,box_footprint(b))
                            if iou>=.25:values.append(dict(index=i,score=float(score),iou=float(iou),center_error_m=float(np.linalg.norm(b[:2]-reference.mean(0)))))
                        matches['candidate' if prefix else 'retained']=sorted(values,key=lambda x:-x['score'])
                    result[name]=matches
                rows.append(result)
    report=dict(calibration=str(calibration_path),new_inference=False,model_predictions_changed=False,
                calibration_used_predictions=False,n1_certified=False,recognition_solved=False,records=rows)
    (out/'reference_validation.json').write_text(json.dumps(report,indent=2))
    for scene in ('172_new','552_new','172_baseline_proxy','552_baseline_proxy'):
        for v in plan['variants']:
            group=[r for r in rows if r['scene']==scene and r['variant']==v]
            print(scene,v,'old retained',sum(bool(r['old']['retained']) for r in group),'cal retained',sum(bool(r['calibrated']['retained']) for r in group),flush=True)

if __name__=='__main__':analyze()
