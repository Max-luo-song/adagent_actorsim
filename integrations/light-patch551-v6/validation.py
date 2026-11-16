"""Evidence gates and preregistered acceptance; truth restricted to post-hoc review."""

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

pass  # Path is provided by the portable boundary above
import numpy as np
from common import P26,asset,read,write
OFF_A=P26/'light_map551_v4/A';OFF_B=P26/'light_map551_cutin_v5/B'

def actor_sync(root,name,tracks,session):
    root=Path(root);off=OFF_B if name.startswith('B') else OFF_A
    with np.load(off/'actor_tracks.npz') as z:
        same=set(z.files)==set(tracks) and all(np.array_equal(z[k],tracks[k]) for k in z.files)
    model=session.backend.trainer.models['RigidNodes'];env=session.backend.env.models['RigidNodes']
    shared=all(env[k] is getattr(model,k) for k in ['instances_trans','instances_quats','instances_size','instances_fv'])
    result=dict(passed=same and shared,actor_values_identical_to_patch_off=same,renderer_and_collision_share_state=shared,
        reference=asset(off/'actor_tracks.npz'),current=asset(root/name/'actor_tracks.npz'),actual_arm=name)
    write(root/name/'actor_sync.json',result)
    if not result['passed']:raise ValueError('Patch arm actor state differs from its off reference')
    return result

def repeat_gate(root):
    root=Path(root);old=read(OFF_B/'run_report.json');new=read(root/'B_repeat/run_report.json')
    collision=new.get('terminal_collision',{});previous=old.get('terminal_collision',{})
    target=lambda r:any(b['group']=='RigidNodes' and b['index']==3 for b in r.get('overlaps',[]))
    association=read(root/'B_repeat/target_detection_association_reference_only.json')['rows'];evidence=[]
    for a in association:
        i=a['index'];d=a.get('retained')
        if i>=len(new['trace']) or d is None or d['xy_distance_m']>=2. or d['score']<.4:continue
        row=new['trace'][i];relative=row['environment'].get('target_relative_forward_left_up_m')
        if relative and 0<relative[0]<25 and abs(relative[1])<2 and row['motion_state'][3]>2 and row['control']['target_speed_mps']>3:
            evidence.append(dict(index=i,relative=relative,score=d['score'],center_error_m=d['xy_distance_m'],
                actual_speed_mps=row['motion_state'][3],requested_speed_mps=row['control']['target_speed_mps']))
    checks=dict(target_collision_repeated=target(collision) and target(previous),
        collision_time_within_preregistered_point2s=bool(collision) and abs(collision['timestamp_s']-previous['timestamp_s'])<=.2,
        no_runtime_failure=new['status']!='failed',common_history_verified=all((root/f'B_repeat/steps/{i:03d}/prefix_gate.json').exists() for i in range(6)),
        actor_values_verified=read(root/'B_repeat/actor_sync.json')['passed'],
        no_braking_despite_near_target_detections_repeated=len(evidence)>=10,
        original_A_no_anomaly=not read(OFF_A/'summary.json')['any_anomaly'])
    result=dict(passed=all(checks.values()),checks=checks,evidence=evidence,
        repeated_run=asset(root/'B_repeat/run_report.json'),original_run=asset(OFF_B/'run_report.json'),
        conclusion='repeatable B-only operational failure: front vehicle detections do not lead to adequate slowing; later detections are lost' if all(checks.values()) else 'repeatability/mechanism insufficient; candidate not admitted',
        unresolved=['neural internal reason for weak planning response','effect of lateral map alignment','close-view rendering/detection degradation'],
        ground_truth_association_reference_only=True)
    write(root/'failure_repeat_verification.json',result);return result

def summarize(root):
    root=Path(root);a=read(root/'A_patch_on/summary.json');b=read(root/'B_patch_on/summary.json');off=read(OFF_A/'summary.json')
    replay_b=read(root/'evaluations/B/pair_scores.json');replay_a=read(root/'evaluations/A/pair_scores.json')
    exec_a=read(root/'evaluations/A_execution/pair_scores.json');planning_a=read(root/'evaluations/A_planning/pair_scores.json')
    exec_b=read(root/'evaluations/B_execution/pair_scores.json');planning_b=read(root/'evaluations/B_planning/pair_scores.json')
    if any(v.get('scores') is None for v in [replay_a,replay_b,exec_a,planning_a,exec_b,planning_b]):raise ValueError('Native GPU scores missing; no success decision')
    def nondegraded(value):
        x=value['scores']['raw_uniad'];y=value['scores']['adopted_plan']
        return all(y[k]['lower']>=x[k]['upper']-1e-6 for k in ['nc','dac','ttc','c','pdms','rc','hdscore'])
    def benefit(value):return value['scores']['adopted_plan']['pdms']['lower']>value['scores']['raw_uniad']['pdms']['upper']+1e-6
    br=read(root/'B_patch_on/run_report.json');ar=read(root/'A_patch_on/run_report.json')
    checks=dict(B_fixed_observation_planning_gain=benefit(replay_b),B_closed_loop_planning_gain=benefit(planning_b),
        B_complete=br['status']=='completed' and len(br['trace'])==188,B_no_anomaly=not b['any_anomaly'],
        B_map_safe=all(not o['environment'].get('authorized_static_map_departure',False) for o in br['observations']),
        B_patch_modified_reference=any(t.get('patch_applied') for t in br['trace']),
        A_fixed_observation_no_score_drop=nondegraded(replay_a),A_closed_loop_plan_no_score_drop=nondegraded(planning_a),
        A_executed_score_no_drop=nondegraded(exec_a),A_complete=ar['status']=='completed' and len(ar['trace'])==188,
        A_no_anomaly=not a['any_anomaly'],A_progress_no_drop=a['progress_along_initial_forward_m']>=off['progress_along_initial_forward_m']-.001,
        A_no_extra_stopping=a['stopped_seconds']<=off['stopped_seconds'],A_patch_inactive=not any(t.get('patch_applied') for t in ar['trace']))
    passed=all(checks.values());unresolved=b['stop_recovery_unresolved']
    result=dict(passed=passed,checks=checks,conclusion=('effective_on_this_low_speed_development_case_recovery_unresolved' if unresolved else 'effective_on_this_low_speed_development_case') if passed else 'ineffective_or_regressed_rollback',
        candidate=asset(root/'candidate.json'),default_version_changed=False,adopted_as_default=False,
        rollback=not passed,rollback_action='keep original frozen baseline and disable candidate' if not passed else None,
        recovery_observation_unresolved=unresolved,unseen_cut_in_timing_gap_and_normal_speed_validation_required=True,
        score_scope='fixed-observation hypothetical plans are separated from closed-loop plans and actual executed future poses; no planning score replaces actual collision checks',
        B_native_execution_scores=exec_b['scores'],B_native_planning_scores=planning_b['scores'])
    write(root/'patch_validation.json',result);return result
