"""Collect calibrated asset render/inference jobs without changing their inputs."""
import json,time
from pathlib import Path
import numpy as np
from evaluate_preview_vehicle import evaluate
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'outputs/asset_calibration_20261021'


def collect():
    scenes={};finished=True
    for scene,tag in [('172','10'),('552','20')]:
        name=f'calasset-{tag}-{scene}-render-v1';root=BASE/'outputs/gpu_test_service/jobs';render=root/name;infer=root/(name+'-infer')
        def status(folder):
            p=folder/'status.json'
            return json.loads(p.read_text()) if p.exists() else dict(status='queued')
        rs,ins=status(render),status(infer);row=dict(render=rs,inference=ins)
        if rs['status'] in ('failed','error'):
            row['inference']=dict(status='not_scheduled_render_failed')
        elif rs['status']=='passed' and ins['status'] in ('passed','failed','error'):
            if ins['status']=='passed':
                result_path=OUT/f'scene_{scene}_recognition.json'
                if result_path.exists():report=json.loads(result_path.read_text())
                else:report=evaluate(render/'result/sequence_report.json',render/'result/asset_report.json',infer/'result',result_path)
                row['retained_matching_frames']=sum(r['retained']['geometric_match_found'] for r in report['records'])
                row['total_frames']=len(report['records']);row['records']=report['records']
                old=json.loads((BASE/f'outputs/gpu_test_service/jobs/edfix-{tag}-{scene}-render-v1/result/asset_report.json').read_text())
                new=json.loads((render/'result/asset_report.json').read_text())
                row['placement_translation_change_m']=float(np.linalg.norm(np.asarray(old['placements'][0]['asset_to_source'])[:3,3]-np.asarray(new['placements'][0]['asset_to_source'])[:3,3]))
                row['calibration_sha256']=new['asset_calibration']['sha256']
        else:finished=False
        scenes[scene]=row
    report=dict(finished=finished,new_training=False,physical_dimensions_certified=False,n1_certified=False,
                correct_recognition_proven=False,scenes=scenes)
    (OUT/'gpu_results.json').write_text(json.dumps(report,indent=2))
    lines=['# 校准资产 GPU 检查','','仅为隔离编辑预览；代理匹配不等于正确识别或N1通过。','']
    for scene,row in scenes.items():
        lines += [f"## {scene}",'',f"渲染：{row['render']['status']}；推理：{row['inference']['status']}。",'']
        if 'records' in row:
            lines += [f"保留匹配：{row['retained_matching_frames']}/{row['total_frames']}；相对旧放置平移变化：{row['placement_translation_change_m']:.6f}m。",'']
            for r in row['records']:
                m=r['retained']['matches']
                lines.append(f"- 帧{r['frame']}："+(f"score={m[0]['raw_score']:.6f}，中心偏差={m[0]['center_error_m']:.3f}m，长轴偏差={m[0]['heading_error_modulo_180_degrees']:.2f}°。" if m else '无保留匹配。'))
            lines.append('')
    (OUT/'gpu_results.md').write_text('\n'.join(lines))
    if finished:
        note='\n\n## 校准资产GPU任务结束（自动同步）\n\n详见 [结果文件](outputs/asset_calibration_20261021/gpu_results.md)。执行完成不代表识别准确或N1认证通过。\n'
        for name in ('cli-notes.md','ASSET_CALIBRATION_PROGRESS.md'):
            p=BASE/name
            if p.exists() and '## 校准资产GPU任务结束（自动同步）' not in p.read_text():p.write_text(p.read_text()+note)
    return finished


if __name__=='__main__':
    deadline=time.monotonic()+5400
    while True:
        try:
            if collect():print('Calibration jobs completed; reports synchronized.',flush=True);break
        except Exception as exc:
            (OUT/'collector_error.txt').write_text(repr(exc));print(repr(exc),flush=True);raise
        if time.monotonic()>=deadline:print('Collector budget exhausted; pending status saved.',flush=True);break
        time.sleep(15)
