"""Offline bounded-factor replay analysis. No inference, controls or threshold tuning."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad'))
from box_geometry import box_footprint
from diagnose_box_generalization import native_matches
from evaluate_preview_vehicle import reference_polygon, polygon_iou
from perception_run_binding import file_fingerprint


def write_readable(report, path):
    def number(value):
        return '-' if value is None else f'{value:.4f}'
    lines=['# 联合验证结果（自动汇总）', '',
           '不作为有效性证据的输入流：'+str(report.get('invalid_effectiveness_cases', [])), '',
           '以下为离线匹配诊断，原始分数未修改；匹配代理不证明车辆身份、编辑真实性、控制正确或正式泛化。', '',
           '## 原生车辆', '',
           '| 场景 | 配置 | 匹配数 | 中心误差中位m | 配置共同匹配数 | 共同集合误差中位m |',
           '|---|---|---:|---:|---:|---:|']
    for scene, variants in report['native'].items():
        for variant, row in variants.items():
            lines.append(f"|{scene}|{variant}|{row['match_count']}|{number(row['median_center_error_m'])}|{row['all_variants_common_match_count']}|{number(row['common_match_median_center_error_m'])}|")
    lines+=['', '## 编辑位置及基线反查', '',
            'old/new分别为颜色旋转修复前/后，位置与尺寸按独立渲染汇总核对；baseline_proxy为未插车图像在同一位置的反查。warmup保留原始困难帧，同时加入更早上下文。', '',
            '| 输入流 | 配置 | 候选匹配帧 | 保留框匹配帧 |', '|---|---|---:|---:|']
    for scene, variants in report['edited'].items():
        for variant, row in variants.items():
            frames=row['records'];candidate=sum(bool(r['candidates']) for r in frames);retained=sum(bool(r['retained']) for r in frames)
            lines.append(f'|{scene}|{variant}|{candidate}/{len(frames)}|{retained}/{len(frames)}|')
    lines+=['', '## 逐帧保留框', '', '| 输入流 | 配置 | 帧 | 分数 | IoU | 中心误差m |', '|---|---|---:|---:|---:|---:|']
    for scene, variants in report['edited'].items():
        for variant, row in variants.items():
            for frame in row['records']:
                matches=frame['retained']
                if matches:
                    match=max(matches,key=lambda r:r['score'])
                    lines.append(f"|{scene}|{variant}|{frame['frame']}|{number(match['score'])}|{number(match['iou'])}|{number(match['center_error_m'])}|")
                else:
                    lines.append(f"|{scene}|{variant}|{frame['frame']}|未保留|-|-|")
    lines+=['', '配置仅改变图像几何预处理和相机嵌入开关，检测阈值仍为0.4/0.35。原生GT仅在推理后分析，没有进入感知输入。',
            '023/621为原生照片检查，域不同于渲染；样本仅各三帧，点云/视锥筛选未证明遮挡可见。缺失匹配会造成选择偏差，因此同时报告计数与共同集合误差。',
            'N1、准确定位和未见场景泛化仍需独立证据；脚本不自动修改生产配置或提升任何可信标记。', '']
    Path(path).write_text('\n'.join(lines))


def analyze(plan_path):
    plan_path=Path(plan_path);plan=json.loads(plan_path.read_text());jobs=BASE/'outputs/gpu_test_service/jobs'
    report=dict(native={},edited={},invalid_effectiveness_cases=plan.get('invalid_effectiveness_cases', []),causal_hypotheses_not_proven=True,production_variant_selected=False,
                unseen_scene_generalization_proven=False,n1_edit_certified=False,plan=file_fingerprint(plan_path))
    for job in plan['jobs']:
        status=json.loads((jobs/job['id']/'status.json').read_text())
        if status['status']!='passed':
            raise ValueError('Ablation job is not complete: '+job['id'])
        sequence=Path(job['args']['sequence_report']);manifest=json.loads(sequence.read_text())
        for variant in plan['variants']:
            output=jobs/job['id']/'result'/variant
            run=json.loads((output/'run_report.json').read_text())
            if (run['completed_frames']!=len(manifest['frames']) or not run['diagnostic_preview_only']
                    or run['input_binding']['implementation_hash']!=plan['implementation_hash']
                    or not run['implementation_unchanged_during_run'] or run['diagnostic_variant']!=variant
                    or run['raw_scores_modified'] or run['planning_outputs_exported'] or run['control_actions_executed']
                    or run['input_binding']['sequence']!=file_fingerprint(sequence)):
                raise ValueError('Invalid frozen diagnostic binding')
            for frame, bound in zip(manifest['frames'], run['input_binding']['packets']):
                packet=Path(frame['packet'])
                if not packet.is_absolute():packet=sequence.parent/packet
                current=file_fingerprint(packet)
                if any(current[key]!=bound[key] for key in ('path','bytes','sha256')):
                    raise ValueError('Input packet changed after inference')
            for segment in plan['segments']:
                scene=str(segment['scene'])
                if segment['role']=='baseline':
                    report['native'].setdefault(scene,{})[variant]=native_matches(
                        Path(segment['sequence']),output,box_footprint,index_offset=segment['first_index'])
                    continue
                asset=json.loads(Path(segment['asset_report']).read_text());records=[]
                for index in range(segment['first_index'],segment['first_index']+segment['frames']):
                    frame=manifest['frames'][index]
                    with Path(frame['packet']).open('rb') as stream:_,info=pickle.load(stream)
                    reference=reference_polygon(asset,info)
                    with np.load(output/f'frame_{index:04d}/perception.npz',allow_pickle=False) as arrays:
                        row=dict(frame=frame['frame'])
                        for prefix in ('candidate_',''):
                            boxes,scores,labels=[arrays[prefix+k] for k in ('boxes','scores','labels')]
                            matches=[]
                            for i,(box,score,label) in enumerate(zip(boxes,scores,labels)):
                                if label!=0:continue
                                iou=polygon_iou(reference,box_footprint(box))
                                if iou>=.25:
                                    matches.append(dict(index=i,score=float(score),iou=float(iou),center_error_m=float(np.linalg.norm(box[:2]-reference.mean(0)))))
                            row['candidates' if prefix else 'retained']=matches
                        records.append(row)
                report['edited'].setdefault(scene,{})[variant]=dict(records=records,reference_is_placement_proxy=True)
    # Compare the same associated object/frame set; raw match counts alone can hide selection bias.
    for scene,variants in report['native'].items():
        maps={name:{(r['frame'],m['instance_key']):m for r in value['records'] for m in r['matches']}
              for name,value in variants.items()}
        common=set.intersection(*(set(m) for m in maps.values()))
        for name,value in variants.items():
            subset=[maps[name][key]['center_error_m'] for key in sorted(common)]
            value['all_variants_common_match_count']=len(common)
            value['common_match_median_center_error_m']=float(np.median(subset)) if subset else None
    path=plan_path.parent/'results.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write_readable(report,plan_path.parent/'joint_results.md')
    return report


if __name__=='__main__':
    result=analyze(sys.argv[1]);print(json.dumps({k:result[k] for k in ('production_variant_selected','unseen_scene_generalization_proven')}))
