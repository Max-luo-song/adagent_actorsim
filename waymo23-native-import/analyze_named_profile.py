"""Validate the canonical named profile against frozen factor-replay outputs."""
import json
from pathlib import Path
import pickle
import time
import numpy as np
from perception_run_binding import file_fingerprint
from evaluate_preview_vehicle import reference_polygon,box_polygon,polygon_iou

BASE=Path(__file__).resolve().parents[1]


def analyze(output):
    output=Path(output).resolve();plan=json.loads((output/'named_profile_plan.json').read_text());identity=plan['job']['id'];folder=BASE/'outputs/gpu_test_service/jobs'/identity/'result'
    if json.loads((folder.parent/'status.json').read_text())['status']!='passed':raise ValueError('Named-profile replay not complete')
    run=json.loads((folder/'run_report.json').read_text());sequence_path=Path(plan['job']['args']['sequence_report']);sequence=json.loads(sequence_path.read_text())
    if (run['model_profile_name']!='calibration_geometry_v1' or run['effective_model_input']['resize_mode']!='letterbox'
            or run['effective_model_input']['camera_embedding_policy']!='geometry_only'
            or run['completed_frames']!=len(sequence['frames']) or not run['implementation_unchanged_during_run']
            or run['input_binding']['implementation_hash']!=plan['implementation_hash']
            or run['raw_scores_modified'] or run['planning_outputs_exported'] or run['control_actions_executed']
            or run['input_binding']['sequence']!=file_fingerprint(sequence_path)):
        raise ValueError('Named profile binding or isolation failed')
    for frame,bound in zip(sequence['frames'],run['input_binding']['packets']):
        current=file_fingerprint(frame['packet'])
        if any(current[k]!=bound[k] for k in ('path','bytes','sha256')):raise ValueError('Input packet changed')
    reference=BASE/'outputs/gpu_test_service/jobs/edzz-20-joint-validation-v2/result/letterbox_no_camera_embedding'
    previous=json.loads((reference/'run_report.json').read_text())
    if (previous['input_binding']['implementation_hash']!=plan['implementation_hash']
            or previous['input_binding']['checkpoint']!=run['input_binding']['checkpoint']
            or previous['thresholds']!=run['thresholds']):raise ValueError('Model/checkpoint/thresholds changed')
    equivalence=[]
    for index in range(previous['completed_frames']):
        with np.load(folder/f'frame_{index:04d}/perception.npz',allow_pickle=False) as actual,np.load(reference/f'frame_{index:04d}/perception.npz',allow_pickle=False) as expected:
            matches={};errors={}
            for name in actual.files:
                a,b=actual[name],expected[name]
                if a.dtype.kind in 'US':matches[name]=bool(np.array_equal(a,b));continue
                same=a.shape==b.shape;matches[name]=bool(same and np.allclose(a,b,rtol=1e-5,atol=2e-4,equal_nan=True))
                errors[name]=float(np.max(np.abs(a-b))) if same and a.size else (0. if same else None)
        audit=json.loads((folder/f'frame_{index:04d}/input_geometry.json').read_text());prior=json.loads((reference/f'frame_{index:04d}/input_geometry.json').read_text())
        equivalence.append(dict(index=index,outputs_equal=all(matches.values()),array_equal=matches,array_max_error=errors,
                                input_geometry_equal=audit==prior))
    segment=plan['additional_segment'];asset=json.loads(Path(segment['asset_report']).read_text());records=[];pixel_checks=[]
    for index in range(segment['first_index'],segment['first_index']+segment['frames']):
        frame=sequence['frames'][index]
        with Path(frame['packet']).open('rb') as stream:obs,info=pickle.load(stream)
        truth=reference_polygon(asset,info);matches=[]
        with np.load(folder/f'frame_{index:04d}/perception.npz',allow_pickle=False) as data:
            for box,score,label in zip(data['boxes'],data['scores'],data['labels']):
                if label!=0:continue
                iou=polygon_iou(truth,box_polygon(box))
                if iou>=.25:matches.append(dict(score=float(score),bev_iou=iou,center_error_m=float(np.linalg.norm(box[:2]-truth.mean(0)))))
        records.append(dict(frame=frame['frame'],retained_matches=matches))
        if frame['frame'] in (60,65,70):
            cold=BASE/f"outputs/gpu_test_service/jobs/edfix-20-552-render-v1/result/frame_{frame['frame']:03d}/observation.pkl"
            with cold.open('rb') as stream:before,before_info=pickle.load(stream)
            differences=[np.abs(obs['rgb'][name].astype(int)-before['rgb'][name].astype(int)) for name in obs['rgb']]
            pose_equal=all(np.allclose(info[key],before_info[key],rtol=0,atol=1e-6) for key in ('explicit_l2g_r_mat','explicit_l2g_t'))
            pixel_checks.append(dict(frame=frame['frame'],max_rgb_difference=max(int(a.max()) for a in differences),pose_equal=pose_equal))
    result=dict(profile='calibration_geometry_v1',equivalence=equivalence,
                equivalent_outputs_within_numeric_tolerance=all(r['outputs_equal'] for r in equivalence),
                exact_input_geometry_equal=all(r['input_geometry_equal'] for r in equivalence),
                comparison_numeric_tolerance=dict(atol=2e-4,rtol=1e-5),
                prefix55_records=records,prefix55_vs_cold_input_checks=pixel_checks,
                prefix45_excluded_due_to_possible_actor_overlap=True,
                recognition_solved=False,n1_edit_certified=False,unseen_scene_generalization_proven=False)
    (output/'named_profile_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 命名配置入口验证','',f"相同前30帧的数值输出等价：{result['equivalent_outputs_within_numeric_tolerance']}。",f"模型前输入几何逐项完全一致：{result['exact_input_geometry_equal']}。",'', '## 55帧开始的前缀','', '| 帧 | 保留框数 | 最高匹配分数 | 中心误差m |','|---|---:|---:|---:|']
    for row in records:
        top=max(row['retained_matches'],key=lambda r:r['score']) if row['retained_matches'] else None
        score=f"{top['score']:.4f}" if top else '-';error=f"{top['center_error_m']:.4f}" if top else '-'
        lines.append(f"|{row['frame']}|{len(row['retained_matches'])}|{score}|{error}|")
    lines+=['','45/50帧存在可能的原生车体包络交叠，较长前缀不用于证明效果。55帧开始前缀未发现该交叠，但仍无独立遮挡、接触或N1证书。',
            '保留框匹配不等于准确定位或控制正确。输入一致性、完整逐数组差异及原始结果见named_profile_validation.json。','']
    (output/'named_profile_validation.md').write_text('\n'.join(lines));return result


if __name__=='__main__':
    import sys
    output=Path(sys.argv[1]);status=BASE/'outputs/gpu_test_service/jobs/edzz-30-named-profile-v1/status.json';deadline=time.monotonic()+3600
    while time.monotonic()<deadline:
        state=json.loads(status.read_text())['status'] if status.exists() else 'pending'
        if state=='passed':analyze(output);print('NAMED_PROFILE_ANALYSIS_COMPLETE',flush=True);break
        if state in ('failed','error'):raise RuntimeError('Named-profile GPU replay '+state)
        time.sleep(15)
    else:raise TimeoutError('Named-profile replay collection budget reached')
