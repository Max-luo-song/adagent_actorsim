"""Collect frozen render/inference evidence; this program issues no GPU commands."""
import argparse
import json
from pathlib import Path
import time
import numpy as np

BASE=Path(__file__).resolve().parents[1]


def collect(output):
    from perception_run_binding import file_fingerprint
    from paired_edit_diagnostic import compare
    output=Path(output).resolve();plan=json.loads((output/'gpu_plan.json').read_text())
    for bound in plan['renderer_binding']+plan['source_bindings']:
        if file_fingerprint(bound['path'])!=bound:
            raise ValueError('Frozen renderer/source changed: '+bound['path'])
    result=dict(task_statuses={},pairs={},geometry_unchanged={},recognition_solved=False,
                n1_edit_certified=False,unseen_scene_generalization_proven=False)
    ids=[j['id'] for j in plan['jobs']]
    ids += [j['id']+'-infer' for j in plan['jobs'] if j['args'].get('infer_after_render')]
    for identity in ids:
        p=BASE/'outputs/gpu_test_service/jobs'/identity/'status.json'
        result['task_statuses'][identity]=json.loads(p.read_text())['status'] if p.exists() else 'pending'
    for pair_path in plan['pairs']:
        pair_path=Path(pair_path);pair=json.loads(pair_path.read_text());name=pair_path.stem
        paths=[Path(pair[k])/'run_report.json' for k in ('edited_inference','baseline_inference')]
        if not all(p.is_file() for p in paths):continue
        # Never analyze partial finally-written reports from running/failed tasks.
        parent_ids=[Path(pair[k]).parent.name for k in ('edited_inference','baseline_inference')]
        if any(result['task_statuses'].get(identity)!='passed' for identity in parent_ids):continue
        dest=output/name
        report_path=dest/'paired_report.json'
        report=json.loads(report_path.read_text()) if report_path.exists() else compare(pair_path,dest)
        result['pairs'][name]=dict(candidate_matched_frames=sum(r['candidates']['edited_match'] for r in report['records']),
            retained_matched_frames=sum(r['retained']['edited_match'] for r in report['records']),frames=len(report['records']),report=str(report_path))
    for scene in (172,552):
        old=json.loads((output/f'scene_{scene}_old_pair.json').read_text());new=json.loads((output/f'scene_{scene}_new_pair.json').read_text())
        p=Path(new['asset_report'])
        if not p.exists():continue
        before=json.loads(Path(old['asset_report']).read_text());after=json.loads(p.read_text())
        same=bool(np.allclose(before['placements'][0]['asset_to_source'],after['placements'][0]['asset_to_source'],atol=1e-6,rtol=0)
            and np.allclose(before['asset_wlh'],after['asset_wlh'],atol=1e-6,rtol=0))
        result['geometry_unchanged'][str(scene)]=same
        if not same:raise ValueError('Radiance comparison also changed geometry')
    result['all_tasks_finished']=all(v in ('passed','failed','error') for v in result['task_statuses'].values())
    result['all_tasks_passed']=all(v=='passed' for v in result['task_statuses'].values())
    (output/'gpu_results.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    deadline=time.monotonic()+2400
    previous=None
    while True:
        result=collect(args.output)
        state=json.dumps(result['task_statuses'],sort_keys=True)
        if state!=previous:print(state,flush=True);previous=state
        if result['all_tasks_finished'] or not args.wait:break
        if time.monotonic()>deadline:raise TimeoutError('Collection budget reached; GPU jobs may still be running')
        time.sleep(15)
