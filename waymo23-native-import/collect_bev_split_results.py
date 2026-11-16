"""Bounded read-only collection of the two predeclared BEV jobs and report sync."""
import json
from pathlib import Path
import time

BASE = Path(__file__).resolve().parents[1]
OUT = BASE/'outputs/zero_training_bev_split_20261021'
IDS = {scene: f'bevsplit-{scene}-20261021-v1' for scene in ('172','552')}


def collect():
    results = {}
    finished = True
    for scene, identity in IDS.items():
        folder = BASE/'outputs/gpu_test_service/jobs'/identity
        status = json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else dict(status='queued')
        report_path = folder/'result/experiment_report.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        finished &= status['status'] in ('passed','failed','error')
        results[scene] = dict(job=identity, status=status, experiment=report)
    result = dict(all_jobs_finished=finished, new_training_allowed=False, edited_vehicle_recognition_validated=False,
                  generalization_proven=False, scenes=results)
    OUT.mkdir(exist_ok=True)
    (OUT/'gpu_results.json').write_text(json.dumps(result,indent=2))
    return result


def sync(result):
    lines = ['# 零训练 BEV 分流 GPU 结果', '', '执行通过只表示实验完成；本轮是无编辑场景，不能证明编辑车辆识别或泛化成功。', '']
    for scene, row in result['scenes'].items():
        lines += [f"## 场景 {scene}", '', f"任务状态：{row['status']['status']}", '']
        for policy, completed in row['experiment'].get('completed',{}).items():
            points = completed['endpoints_xy']
            lines += [f"- {policy}：{completed['steps']} 步，紧急制动 {completed['emergency_braking_steps']} 步；预测终点原始 XY 首／末：{points[0] if points else None} ／ {points[-1] if points else None}。"]
        lines += ['', f"检查点前后相同：{row['experiment'].get('checkpoint_unchanged', '未完成')}；实现前后相同：{row['experiment'].get('implementation_unchanged', '未完成')}。", '']
    (OUT/'gpu_results.md').write_text('\n'.join(lines))
    if result['all_jobs_finished']:
        note='\n\n## BEV 分流 GPU 任务已结束（自动同步）\n\n'+ '\n'.join(f"- {s}：{r['status']['status']}。" for s,r in result['scenes'].items())+'\n\n详见 [完整结果](outputs/zero_training_bev_split_20261021/gpu_results.md)；任务执行状态不等于有效性结论。\n'
        for name in ('ZERO_TRAINING_BEV_SPLIT_VALIDATION.md', 'cli-notes.md'):
            path=BASE/name
            if '## BEV 分流 GPU 任务已结束（自动同步）' not in path.read_text():
                path.write_text(path.read_text()+note)


if __name__ == '__main__':
    deadline = time.monotonic()+5400
    while True:
        result=collect();sync(result)
        if result['all_jobs_finished']:
            print('Both jobs finished; reports synchronized.',flush=True)
            break
        if time.monotonic()>=deadline:
            print('Collector budget ended; GPU status remains in gpu_results.json.',flush=True)
            break
        time.sleep(15)
