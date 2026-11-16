"""Authorized static-map 551 episode, all remaining valid source frames."""

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

import argparse,copy,datetime,importlib.util,json,sys
pass  # Path is provided by the portable boundary above
import numpy as np
LOCAL=Path(__file__).resolve().parent;OLD=LOCAL.parent/'light-cutin-diagnostic'
sys.path.insert(1,str(OLD))
from common import *
spec=importlib.util.spec_from_file_location('map_original_pipeline',OLD/'pipeline.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base);original_check=base.check
OUTPUT=P26/'light_map551_v4';PREVIOUS=P26/'light_road551_v3'

def pins():return {str(p):sha(p) for p in LOCAL.glob('*.py')}|{str(LOCAL/'run.sh'):sha(LOCAL/'run.sh')}

def check(output):
    root,plan=original_check(output)
    if plan['map_v4_code']!=pins():raise ValueError('Frozen static-map implementation changed')
    for key in ['static_map','boot_manifest','boot_archive','initial_context_asset']:verify(plan[key])
    mapping=read(plan['static_map']['path'])
    for key in ['support','fusion_support','source_receipts']:verify(mapping[key])
    if not mapping['online_static_map_allowed'] or mapping['online_actor_truth_allowed']:raise ValueError('Map authorization mismatch')
    if plan['controls']!=plan['source_end_frame']-plan['initial_context']['source_frame']:raise ValueError('Episode does not cover all valid source frames')
    return root,plan
base.check=check

def prepare(output):
    root=Path(output).resolve()
    if (root/'plan.json').exists():raise FileExistsError('Prepared version exists; reuse it')
    root.mkdir(parents=True,exist_ok=True)
    if not (root/'map/map.json').exists():raise ValueError('Compile/review static map before freezing the run')
    if not (root/'boot/manifest.json').exists():raise ValueError('Build byte-identical legacy boot support first')
    _,old=original_check(PREVIOUS);configure()
    from runtime import TERMINATION
    from map_reference import StaticGeometry
    from runtime_configuration import load_vehicle_config
    mapping=read(root/'map/map.json');v=load_vehicle_config();geometry=StaticGeometry(mapping,v)
    for name in ['baseline_freeze.json','baseline_verification.json','scene_manifest.json','initial_context.json','baseline_input_audit.json']:
        (root/name).write_bytes((PREVIOUS/name).read_bytes())
    context=read(root/'initial_context.json');gap=geometry.pose_clearance(context['ego_to_world'])
    if gap<.05:raise ValueError('Initial body does not fit authorized map with safety margins')
    for asset_entry in read(root/'map/fusion_sources.json')['sources']:verify(asset_entry)
    write(root/'scenario/scenario.json',dict(config=dict(scene=551,initial_ego_speed_mps=context['initial_speed_mps'],intervention_start_s=1.2),
        original_native_actors=True,intervention_admitted=False,static_map_authorized=True))
    boot=read(root/'boot/manifest.json');plan=copy.deepcopy(old)
    for key in ['road_v3_code','road_admission','A_off','A_off_summary','road_policy','external_road_policy_enabled']:plan.pop(key,None)
    plan.update(schema='authorized_static_map_full_source551_v4',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        code=code_pins(),runtime_code=base.runtime_code_pins(),map_v4_code=pins(),
        controls=188,source_end_frame=198,termination=TERMINATION,initial_context=context,
        initial_context_asset=asset(root/'initial_context.json'),static_map=asset(root/'map/map.json'),
        boot_manifest=asset(root/'boot/manifest.json'),boot_archive=boot['archive'],
        online_static_map_authorized=True,online_actor_truth_authorized=False,online_future_recorded_ego_authorized=False,
        online_truth_allowed='static road map only, explicitly authorized by user',
        allowed_online_inputs=['static road polygon','current frozen UniAD plan','actual ego kinematics'],
        forbidden_online_inputs=['future actor trajectories','recorded future ego trajectory'],
        patch_initially=None,B_admitted=False,gpu_executed=False,
        full_episode_definition='Remaining valid scene source frames10..198, 188 executed controls/18.8s; route completion reported separately',
        preregistered_acceptance=dict(no_map_departure=True,all_stopping_certificates=True,no_environment_collision=True,
            no_abnormal_stall=True,full_valid_time_or_native_route_goal=True,max_tracking_error_m=1.),
        old_A_v3=asset(PREVIOUS/'A/run_report.json'),previous_plan=asset(PREVIOUS/'plan.json'))
    plan['assets'].update(scene=asset(root/'scene_manifest.json'),scenario=asset(root/'scenario/scenario.json'),baseline=asset(root/'baseline_freeze.json'))
    write(root/'plan.json',plan);write(root/'status.json',dict(stage='static_map_full_episode_prepared',controls=188,simulation_s=18.8,gpu_executed=False))
    return dict(prepared=True,controls=188,source_end_frame=198,initial_map_clearance_m=gap,per_frame_line_detection=False)

def result(root):
    root=Path(root);a=read(root/'A/run_report.json');s=read(root/'A/summary.json');plan=read(root/'plan.json')
    gaps=[o['environment'].get('authorized_static_map_clearance_m') for o in a['observations']]
    full=a['status']=='completed' and len(a['trace'])==plan['controls']
    reached=a.get('termination_reason')=='native_route_complete'
    safe=bool(a['trace']) and all(t['road_gate']['certificate']['passed'] for t in a['trace'])
    checks=dict(full_valid_time_or_native_goal=full or reached,all_action_certificates=safe,
        all_map_clearances_known=bool(gaps) and all(g is not None and g>=0 for g in gaps),
        original_no_anomaly=not s['any_anomaly'],model_unchanged=a.get('model_weights_unchanged') is True)
    out=dict(passed=all(checks.values()),checks=checks,controls=len(a['trace']),simulation_s=s['elapsed_simulation_s'],
        full_source_time_window_completed=full,native_route_completed=reached,route_completion=s['native_scores'].get('actual_progress'),
        min_authorized_map_clearance_m=min(g for g in gaps if g is not None) if gaps else None,
        max_lateral_drift_m=max(abs(o['motion_state'][0]) for o in a['observations']) if gaps else None,
        per_frame_line_detector_calls=0,source_clock_end=10+len(a['trace']),
        driving_wall_seconds=a.get('driving_wall_seconds'),total_driving_and_setup_wall_seconds=a['wall_seconds'],
        map=asset(root/'map/map.json'),summary=asset(root/'A/summary.json'),
        ground_truth_scope='User-authorized static sensor-derived map; no future actor or recorded future ego execution',
        independent_HD_map_certificate=False,route_completion_is_not_implied_by_temporal_completion=True,B_admitted=False)
    write(root/'full_episode_admission.json',out);write(root/'status.json',dict(stage='full_episode_finished',result=out));return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','check','full','inspect','render']);p.add_argument('--output',default=str(OUTPUT));args=p.parse_args()
    if args.stage=='prepare':out=prepare(args.output)
    elif args.stage=='check':
        root,plan=check(args.output);configure()
        import torch
        from native_car_edit import make_types
        make_types()
        from map_reference import StaticGeometry
        from runtime_configuration import load_vehicle_config
        geometry=StaticGeometry(read(plan['static_map']['path']),load_vehicle_config())
        out=dict(verified=True,cuda_available=torch.cuda.is_available(),controls=plan['controls'],initial_map_clearance_m=geometry.pose_clearance(plan['initial_context']['ego_to_world']),per_frame_line_detection=False)
    elif args.stage=='full':
        root,plan=check(args.output)
        base.run_and_score(root,'A',plan['controls']);out=result(root)
        if not out['passed']:raise RuntimeError('Full scene ended without admission; inspect full_episode_admission.json')
    elif args.stage=='inspect':root,_=check(args.output);out=result(root)
    else:
        root,_=check(args.output);configure()
        from diagnostics import artifacts
        out=artifacts(root,'A','A')
    print(json.dumps(out,indent=2,allow_nan=False))

if __name__=='__main__':main()
