"""Failure evidence -> repeated mechanism -> frozen candidate -> paired validation."""

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

import argparse,ast,copy,datetime,importlib.util,json,sys
pass  # Path is provided by the portable boundary above
LOCAL=Path(__file__).resolve().parent;V5=LOCAL.parent/'light-map551-cutin-v5';V4=LOCAL.parent/'light-map551-v4';OLD=LOCAL.parent/'light-cutin-diagnostic'
sys.path[:0]=[str(LOCAL),str(V5),str(V4),str(OLD)]
from common import *

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

base=module('patch6_original_pipeline',OLD/'pipeline.py');original_check=base.check
v5=module('patch6_frozen_B_pipeline',V5/'pipeline.py')
sys.path[:0]=[str(LOCAL),str(V5),str(V4),str(OLD)]
OUTPUT=P26/'light_patch551_v6';OFF_A=P26/'light_map551_v4/A';OFF_B=P26/'light_map551_cutin_v5/B'

def pins():return {str(p):sha(p) for p in [*LOCAL.glob('*.py'),LOCAL/'run.sh']}

def bindings():
    import runtime,scoring,local_boot,tracker
    expect={runtime:LOCAL/'runtime.py',scoring:V4/'scoring.py',local_boot:V4/'local_boot.py',tracker:OLD/'tracker.py'}
    configure()
    import map_reference,intervention
    expect.update({map_reference:V4/'map_reference.py',intervention:V5/'intervention.py'})
    for m,path in expect.items():
        if Path(m.__file__).resolve()!=path:raise ValueError('Wrong frozen implementation: '+m.__name__)
    return {m.__name__:str(p) for m,p in expect.items()}

def check(output):
    root,plan=original_check(output)
    if plan['patch_v6_code']!=pins():raise ValueError('Prepared v6 implementation changed; retain existing results')
    v5.check(OFF_B.parent)
    for entry in plan['off_sources'].values():verify(entry)
    for key in ['scene_manifest.json','baseline_freeze.json','map/map.json','boot/manifest.json','initial_context.json','scenario/target_track.npz','scenario/scenario.json']:
        if sha(root/key)!=sha(OFF_B.parent/key):raise ValueError('Fixed experiment input changed: '+key)
    return root,plan
base.check=check

def prepare(output):
    root=Path(output).resolve()
    if (root/'plan.json').exists():check(root);return dict(reused=True,root=str(root))
    _,previous=v5.check(OFF_B.parent)
    if not v5.base.completed_arm(OFF_B.parent,'B'):raise ValueError('No compatible actual B failure')
    root.mkdir(parents=True,exist_ok=True)
    for name in ['scene_manifest.json','baseline_freeze.json','baseline_verification.json','initial_context.json','baseline_input_audit.json','map/map.json','boot/manifest.json','scenario/scenario.json','scenario/target_track.npz']:
        dst=root/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((OFF_B.parent/name).read_bytes())
    plan=copy.deepcopy(previous);plan.update(schema='low_speed551_failure_driven_external_patch_v6',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),patch_v6_code=pins(),
        off_sources={str(p.relative_to(P26)):asset(p) for p in [OFF_A/'run_report.json',OFF_A/'summary.json',OFF_A/'actor_tracks.npz',OFF_B/'run_report.json',OFF_B/'summary.json',OFF_B/'actor_tracks.npz',OFF_B.parent/'comparison_A_B.json']},
        validation_order=['repeat actual B off','verify repeated mechanism','freeze candidate','fixed-observation native plan scores','B patch on actual closed loop','A patch on actual closed loop','matched native execution/planning scores','decision'],
        patch_initially=None,default_version_modified=False,normal_speed_claim=False,
        candidate_allowed_fields=['boxes','scores','labels'],
        patch_online_inputs=['current/past retained neural boxes/scores/labels','current actual ego speed/steer/pose','vehicle limits/geometry','declared lidar mount','dt'],
        preregistered_patch_acceptance=dict(B_plan_PDMS_strict_gain=1e-6,A_score_drop_tolerance=1e-6,
            A_progress_drop_tolerance_m=.001,A_extra_stopped_seconds=0.,A_patch_must_remain_inactive=True,
            B_actual_collision_must_resolve=True,B_full188_required=True,B_recovery_if_still_stopped='unresolved, no claim of bypass or recovery',
            repeat_collision_time_tolerance_s=.2,repeat_near_detection_without_slowing_min_observations=10),
        candidate_proposed_only_after_repeat=True)
    plan['assets'].update(scene=asset(root/'scene_manifest.json'),scenario=asset(root/'scenario/scenario.json'),
        baseline=asset(root/'baseline_freeze.json'),target_track=asset(root/'scenario/target_track.npz'))
    write(root/'plan.json',plan)
    r=read(OFF_B/'run_report.json');observed=r['trace'][120]
    evidence=dict(schema='operational_failure_and_patch_decision_v1',status='single_B_failure_repeat_required',
        supported_failure='Observed front vehicle does not produce adequate slowing in the neural plan; retained detections later disappear; ego collides with the stopped native vehicle.',
        collision_time_s=r['terminal_collision']['timestamp_s'],collision_object='native:RigidNodes:3',
        observed_detection_window_s=[11.5,13.5],example=dict(time_s=12.,actual_speed_mps=observed['motion_state'][3],
            reference_speed_mps=observed['control']['target_speed_mps'],target_forward_center_m=observed['environment']['target_relative_forward_left_up_m'][0]),
        external_forward_nodes_preserved=all([x[1] for x in t['raw_uniad']]==[x[1] for x in t['adopted']] for t in r['trace']),
        map_brake_overrides=sum(t['road_gate']['override'] for t in r['trace']),
        supporting_sources=[asset(OFF_B/'run_report.json'),asset(OFF_B/'target_detection_association_reference_only.json'),asset(OFF_B.parent/'comparison_A_B.json')],
        unexcluded_explanations=['neural-domain/feature failure','lateral map alignment influence','close-view rendering and final detection degradation'],
        decision='Implement one draft external front-reference braking/retiming candidate; do not admit it until B failure repeats. Preserve the fixed tracker, map and model.',
        alternative_not_selected='No new obstacle MPC or lateral bypass search: no evidence yet selects a safe alternative route, and that would confound tracker comparison.',
        candidate_parameter_basis=dict(confidence='native retained detection threshold0.4',confirmation='3 observations to reject one-frame spurious boxes',
            memory='0.6s bounded association; expired geometry removed',brake='same declared vehicle min_acc=-2m/s2',
            standstill_gap='7m conservative near-camera visibility reserve; first-case development parameter, not calibrated from future actor script',
            scope='development-case validation only; conservative stationary assumption may brake early for moving traffic'),
        formal_patch_selected=False)
    write(root/'failure_analysis.json',evidence)
    write(root/'status.json',dict(stage='failure_evidence_recorded_repeat_B_required',GPU_executed=False,patch_admitted=False))
    return dict(prepared=True,root=str(root),A_B_off_reused=True,repeat_required=True)

def safety_audit_module(path):
    tree=ast.parse(Path(path).read_text())
    banned={'open','eval','exec','compile','__import__','globals','locals','getattr','setattr'}
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names=[n.name for n in node.names] if isinstance(node,ast.Import) else [node.module or '']
            if any(n.split('.')[0] not in {'numpy','math'} for n in names):raise ValueError('Patch ambient dependency forbidden')
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in banned:raise ValueError('Patch I/O or introspection forbidden')
        if isinstance(node,ast.Attribute) and (node.attr.startswith('__') or node.attr in {'load','save','loadtxt','savetxt','fromfile','tofile','memmap','load_library'}):raise ValueError('Patch ambient file access forbidden')
        if isinstance(node,ast.Subscript) and isinstance(node.slice,ast.Constant) and node.slice.value in {'scene','scene_id','step','source_frame','absolute_time','target_script','ground_truth'}:raise ValueError('Privileged patch trigger forbidden')

def repeat(output):
    root,plan=check(output);bindings()
    base.run_and_score(root,'B_repeat',plan['controls'],OFF_A)
    from validation import repeat_gate
    result=repeat_gate(root)
    if not result['passed']:raise ValueError('B mechanism did not meet repetition criteria; no candidate admission')
    return result

def admit(root):
    root=Path(root);failure=read(root/'failure_repeat_verification.json')
    if not failure['passed']:raise ValueError('Repeated B mechanism required')
    from front_brake import PARAMETERS
    path=LOCAL/'front_brake.py';safety_audit_module(path)
    candidate=dict(status='frozen_for_validation',module=asset(path),mechanism=asset(root/'failure_repeat_verification.json'),
        supporting_evidence=[asset(root/'failure_analysis.json'),asset(root/'replay/summary.json')],parameters=PARAMETERS,
        allowed_perception_fields=['boxes','scores','labels'],no_privileged_inputs=True,
        trigger='At least3 confirmed current/past retained vehicle detections overlap the forward body corridor, and current reference exceeds conservative stopping space.',
        modification_location='After unchanged map reference, before unchanged tracker; retime XY together and revalidate the same full-body map certificate.',
        expected_benefit='Brake for a confirmed in-path vehicle before perception degrades; improve matched native planning and avoid the observed collision.',
        possible_regressions=['false positive vehicles','early braking behind moving traffic','short detection-memory expiry followed by resumed motion','insufficient stopping distance'],
        rollback_conditions='Any A regression, B collision, no robust B planning-score benefit, runtime/sync mismatch or missing comparison evidence.',
        default_enabled=False)
    if (root/'candidate.json').exists():
        if read(root/'candidate.json')!=candidate:raise ValueError('Frozen candidate differs; no retuning inside validation')
    else:write(root/'candidate.json',candidate)
    return candidate

def validate(output):
    root,plan=check(output);bindings()
    if not (root/'failure_repeat_verification.json').exists():repeat(root)
    from replay import run as replay
    replay(root);candidate=admit(root)
    from reference_eval import score_pair
    for name,off in [('A',OFF_A),('B',OFF_B)]:score_pair(root,name,off,replay=True)
    try:
        base.run_and_score(root,'B_patch_on',plan['controls'],OFF_A,candidate)
        base.run_and_score(root,'A_patch_on',plan['controls'],OFF_A,candidate)
        for name,off in [('A',OFF_A),('B',OFF_B)]:
            on=root/(name+'_patch_on')
            score_pair(root,name+'_planning',off,on)
            score_pair(root,name+'_execution',off,on,executed=True)
        from validation import summarize
        result=summarize(root)
    except BaseException as exc:
        result=dict(passed=False,conclusion='unresolved_run_or_evaluation_failed',error=repr(exc),rollback=True,adopted_as_default=False,default_version_changed=False)
        write(root/'patch_validation.json',result);raise
    write(root/'status.json',dict(stage='patch_development_validation_finished',result=asset(root/'patch_validation.json'),conclusion=result['conclusion'],default_version_changed=False))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','check','replay','repeat-B','validate','inspect']);p.add_argument('--output',default=str(OUTPUT));args=p.parse_args()
    if args.stage=='prepare':out=prepare(args.output)
    elif args.stage=='check':
        root,plan=check(args.output);resolved=bindings();safety_audit_module(LOCAL/'front_brake.py')
        import torch
        out=dict(verified=True,cuda_available=torch.cuda.is_available(),bindings=resolved,controls=plan['controls'],candidate_status='requires actual B repetition')
        write(root/'checks/preflight.json',out)
    elif args.stage=='replay':
        root,_=check(args.output);bindings()
        from replay import run
        out=run(root)
    elif args.stage=='repeat-B':out=repeat(args.output)
    elif args.stage=='validate':out=validate(args.output)
    else:root,_=check(args.output);out=read(root/'patch_validation.json')
    print(json.dumps(out,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
