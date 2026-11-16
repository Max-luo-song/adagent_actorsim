"""Read-only A/B/C closed-loop arm summariser (diagnostic, uncertified).

Reads only finished-or-in-progress arm result directories; never writes into
them. Handles partial runs, differing lengths and missing metrics. Comparisons
use the common simulated-time range; full-run results are listed separately.
'Run completed' is never equated with 'avoidance succeeded'.

Metric sources:
- gaps/termination/speed: run_report.json (observations, trace, report fields)
- detections: uniad_worker/step_XXX/perception.npz (candidate after filtering:
  boxes/scores/labels), matched to the inserted asset reference box by xy
  centre distance in source world
- box frame chain: model lidar frame --(virtual_to_ego from sensor_metadata.pkl)-->
  ego --(closed_ego_to_world from observations)--> source world
- heading error is modulo pi (box yaw sign convention not independently
  verified; asset signed front unverified) — relative cross-arm comparison only
"""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand
Path = _HugsimPath

import argparse
import json
import pickle
pass  # Path is provided by the portable boundary above

import numpy as np


def load_report(arm_dir):
    p = Path(arm_dir)/'run_report.json'
    if not p.is_file():
        return None
    return _hugsim_expand(json.loads(p.read_text()))


def asset_world_box(arm_dir):
    """Inserted asset reference in source world: centre xyz, wlh, heading."""
    ref = _hugsim_expand(json.loads((Path(arm_dir)/'asset_reference.json').read_text()))
    P = np.asarray(ref['placements'][0]['asset_to_source'], dtype=float)
    w, l, h = ref['asset_wlh']
    bottom_y = ref['local_bottom_y']
    # asset local centre (recentered footprint, mid height)
    c_local = np.array([0., bottom_y - h/2, 0., 1.])
    centre = P @ c_local
    heading = float(np.arctan2(P[1, 0], P[0, 0]))
    return dict(centre=centre[:3], wlh=[w, l, h], heading=heading,
                footprint_xy=np.asarray(ref['placements'][0]['asset_to_source'])[:2, 3].tolist())


def detection_match(perception_npz, ego_to_world, target_xy):
    """Nearest final detection to the inserted-asset xy centre, in source world.

    Boxes are in the ego-local model frame (x right, y forward; planning_guard
    convention 'source (forward,left) -> model (right,forward)'). Yaw is the
    native clockwise model yaw: heading_world = yaw_ego - yaw_box.
    Box z is not transformed (bottom-centre, mount offset unresolved -> unknown).
    """
    try:
        with np.load(perception_npz) as d:
            boxes = d['boxes']; scores = d['scores']; labels = d['labels']
    except FileNotFoundError:
        return None
    if not len(boxes):
        return dict(detected=False, count=0)
    fwd = boxes[:, 1].astype(float)   # model y = forward
    left = -boxes[:, 0].astype(float)  # model x = right -> left = -x
    R = ego_to_world[:3, :3]
    centres_world = np.stack([fwd, left, np.zeros_like(fwd)], axis=1) @ R.T + ego_to_world[:3, 3]
    dist = np.linalg.norm(centres_world[:, :2] - np.asarray(target_xy), axis=1)
    i = int(np.argmin(dist))
    yaw_ego = float(np.arctan2(ego_to_world[1, 0], ego_to_world[0, 0]))
    yaw_world = yaw_ego-float(boxes[i, 6])
    return dict(detected=True, count=int(len(boxes)), nearest_distance_m=float(dist[i]),
                score=float(scores[i]), label=int(labels[i]), yaw_world=yaw_world,
                centre_world=centres_world[i].tolist())


def summarise_arm(arm_dir, label):
    arm_dir = Path(arm_dir)
    out = dict(arm=label, dir=str(arm_dir), status='missing', n1_certified=False,
               formal_evaluation_admitted=False)
    report = load_report(arm_dir)
    if report is None:
        out['note'] = 'run_report.json not found (not started or pre-first-step)'
        return out
    out['status'] = report.get('status')
    out['termination_reason'] = report.get('termination_reason')
    out['controls_attempted'] = report.get('controls_attempted', 0)
    out['avoidance_success_recorded'] = report.get('avoidance_success')
    out['avoidance_semantics'] = 'harness dynamic_completion criteria; not a certification'
    for k in ('minimum_inserted_reference_gap_m', 'minimum_native_vehicle_reference_gap_m',
              'minimum_all_actor_reference_gap_m'):
        v = report.get(k)
        out[k] = None if v in (None, 'Infinity', float('inf')) else v
    obs = report.get('observations', [])
    trace = report.get('trace', [])
    if obs:
        ego = np.asarray([o['closed_ego_to_world'] for o in obs])[:, :3, 3]
        out['simulated_duration_s'] = float(obs[-1]['timestamp'])
        out['distance_travelled_m'] = float(np.linalg.norm(np.diff(ego, axis=0), axis=1).sum())
        out['final_speed_mps'] = float(obs[-1]['closed_speed_mps'])
        out['speed_series'] = [round(float(o['closed_speed_mps']), 3) for o in obs]
    try:
        box = asset_world_box(arm_dir)
        out['inserted_asset_reference'] = dict(centre=[round(float(v), 3) for v in box['centre']],
                                               wlh=box['wlh'], heading=round(box['heading'], 4))
    except FileNotFoundError:
        box = None
        out['inserted_asset_reference'] = None
    det = []
    for o, t in zip(obs, trace):
        entry = dict(step=o['index'], timestamp=o['timestamp'])
        if box is not None:
            m = detection_match(t['perception'],
                                np.asarray(o['closed_ego_to_world']), box['centre'][:2])
            if m is not None and m.get('detected'):
                err = abs((m['yaw_world']-box['heading']+np.pi/2) % np.pi-np.pi/2)
                entry.update(detected=True, score=round(m['score'], 3),
                             centre_error_m=round(m['nearest_distance_m'], 3),
                             heading_error_rad_mod_pi=round(float(err), 4),
                             detection_count=m['count'])
            else:
                entry.update(detected=False)
        entry.update(gap_to_inserted_m=o.get('inserted_reference_gap_m'),
                     action=t.get('action'), mode=t.get('mode'),
                     optimization_reason=t.get('optimization_reason'))
        det.append(entry)
    out['per_step'] = det
    if det:
        detected = [d for d in det if d.get('detected')]
        out['detection_coverage'] = round(len(detected)/len(det), 3)
        if detected:
            out['inserted_vehicle_score_mean'] = round(float(np.mean([d['score'] for d in detected])), 4)
            out['inserted_vehicle_centre_error_mean_m'] = round(float(np.mean([d['centre_error_m'] for d in detected])), 4)
            out['inserted_vehicle_heading_error_mean_rad_mod_pi'] = round(float(np.mean([d['heading_error_rad_mod_pi'] for d in detected])), 4)
    out['video'] = report.get('video')
    return out


def common_range_table(arms):
    present = [a for a in arms if a.get('per_step')]
    if len(present) < 2:
        return dict(note='fewer than two arms with per-step data; comparison skipped')
    t_max = min(a['per_step'][-1]['timestamp'] for a in present)
    rows = []
    for a in present:
        sel = [d for d in a['per_step'] if d['timestamp'] <= t_max+1e-9]
        speed = [s for s, d in zip(a['speed_series'], a['per_step']) if d['timestamp'] <= t_max+1e-9]
        detected = [d for d in sel if d.get('detected')]
        gaps = [d['gap_to_inserted_m'] for d in sel if d.get('gap_to_inserted_m') is not None]
        rows.append(dict(arm=a['arm'], steps=len(sel),
                         mean_speed_mps=round(float(np.mean(speed)), 3) if speed else None,
                         detection_coverage=round(len(detected)/len(sel), 3) if sel else None,
                         min_gap_to_inserted_m=min(gaps) if gaps else None,
                         score_mean=round(float(np.mean([d['score'] for d in detected])), 4) if detected else None))
    return dict(common_simulated_time_s=float(t_max), rows=rows,
                note='single-run diagnostics; no success-rate or generalisation inference')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arms', nargs='+', required=True, help='label=dir pairs')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out_path = Path(args.output).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    arms = []
    for item in args.arms:
        label, _, d = item.partition('=')
        if not d:
            raise ValueError('arms must be label=dir')
        arms.append(summarise_arm(d, label))
    table = common_range_table(arms)
    result = dict(schema='phase26_arm_comparison_v1', diagnostic_uncertified=True,
                  n1_certified=False, formal_evaluation_admitted=False,
                  arms=arms, common_range_comparison=table,
                  full_run_results='per-arm sections above; partial runs included as-is',
                  heading_error_note='mod pi; box yaw sign convention not independently verified')
    (out_path/'arm_comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = ['# A/B/C 闭环对照（诊断，未认证）', '',
             '| 臂 | 状态 | 步数 | 终止原因 | 记录绕行成功* | 最小插入车间隙(m) | 最小原生车间隙(m) | 检测覆盖率 | 平均置信度 | 中心误差(m) | 朝向误差(rad,mod π) |', 
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for a in arms:
        lines.append('| {arm} | {status} | {controls_attempted} | {termination_reason} | {avoidance_success_recorded} | {g1} | {g2} | {cov} | {score} | {ce} | {he} |'.format(
            arm=a['arm'], status=a.get('status'), controls_attempted=a.get('controls_attempted'),
            termination_reason=a.get('termination_reason'), avoidance_success_recorded=a.get('avoidance_success_recorded'),
            g1=a.get('minimum_inserted_reference_gap_m'), g2=a.get('minimum_all_actor_reference_gap_m'),
            score=a.get('inserted_vehicle_score_mean'), cov=a.get('detection_coverage'),
            ce=a.get('inserted_vehicle_centre_error_mean_m'), he=a.get('inserted_vehicle_heading_error_mean_rad_mod_pi')))
    lines.append('')
    lines.append('*harness 记录值，非认证；"运行完成"不等于"绕行成功"。')
    lines.append('')
    if 'rows' in table:
        lines.append(f"共同时间范围 0–{table['common_simulated_time_s']:.1f}s：")
        lines.append('')
        lines.append('| 臂 | 步数 | 平均速度 | 检测覆盖率 | 最小插入间隙 | 平均置信度 |')
        lines.append('|---|---|---|---|---|---|')
        for r in table['rows']:
            lines.append(f"| {r['arm']} | {r['steps']} | {r['mean_speed_mps']} | {r['detection_coverage']} | {r['min_gap_to_inserted_m']} | {r['score_mean']} |")
    (out_path/'arm_comparison.md').write_text('\n'.join(lines))
    print('written', out_path/'arm_comparison.json')
    print('written', out_path/'arm_comparison.md')


if __name__ == '__main__':
    main()
