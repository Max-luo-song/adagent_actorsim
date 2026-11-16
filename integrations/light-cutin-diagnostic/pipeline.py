"""Ordered baseline freeze -> native intervention -> diagnosis -> gated patch pairs."""

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

import argparse,ast,datetime,json,os,pickle,sys
pass  # Path is provided by the portable boundary above
import numpy as np
from common import *


def baseline_pins():
    paths=[HERE/'tracker.py',HERE/'common.py',ROOT/'waymo23-native-import/live_env.py',ROOT/'waymo23-native-import/configs/vehicle.json',P26/'bridge/agent_json/ego_execution.py']
    paths+=list((ROOT/'code/HUGSIM/sim/ilqr').glob('*.py'))
    return {str(p):sha(p) for p in paths}


def runtime_code_pins():
    """Bind full neural/trainer sources and worker helpers, not only edited files in old releases."""
    recursive=[ROOT/'code/UniAD_SIM/projects',ROOT/'code/UniAD_SIM/tools',
        ROOT/'waymo23-native-import/waymo172_closed_loop_v1',SOURCE/'framework_snapshot',
        P26/'framework_repo/models',P26/'framework_repo/datasets',P26/'framework_repo/utils']
    paths=[p for directory in recursive for p in directory.rglob('*.py') if '__pycache__' not in p.parts]
    for directory in [ROOT/'waymo23-native-import',ROOT/'integrations/sr-hugsim',CR/'phase46/framework_driver_v1',CR/'phase29/global_repair_v1',CR/'phase27']:
        paths.extend(directory.glob('*.py'))
    paths.extend((SOURCE/'native_runtime').rglob('*.so*'))
    fix=P26/'bridge/env_fix/nvdiffrast_build_src/build'
    paths.extend(fix.glob('lib.linux-*/*.so'))
    return {str(p.resolve()):sha(p) for p in dict.fromkeys(paths) if p.is_file()}


def prepare(output):
    root=Path(output).resolve()
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True);configure()
    from tracker import synthetic_audit,PARAMETERS
    from scenario551 import generate,CONFIG
    from runtime import TERMINATION
    print('Auditing existing light tracker and native Motion on straight/left/right/stop...',flush=True)
    audit=synthetic_audit();write(root/'baseline_verification.json',audit)
    if not audit['passed']:raise ValueError('Light baseline failed; no intervention launch admitted')
    write(root/'baseline_freeze.json',dict(schema='fixed_light_tracker_v1',parameters=PARAMETERS,
        code=baseline_pins(),verification=asset(root/'baseline_verification.json'),controller_change_after_this_point=False,
        gpu_baseline_validation_required=True,old_heavy_versions_preserved=True))
    scenario=generate(root/'scenario')
    manifest=read(CR/'phase31/stopped_vehicle551_v1/scene_manifest.json')
    source=read(manifest['source_report'])
    if source['scene']!=551 or str(source['checkpoint'])!=read(BUNDLE/'source_manifest.json')['source_checkpoint']:raise ValueError('Actual scene is not source551')
    if manifest['render_scale']!=1.:raise ValueError('Unexpected original camera scale')
    write(root/'scene_manifest.json',manifest)
    assets=dict(scene=asset(root/'scene_manifest.json'),scene_checkpoint=asset(source['checkpoint']),source_report=asset(manifest['source_report']),
        uniad_weights=asset(ROOT/'code/UniAD_SIM/ckpts/uniad_base_e2e.pth'),scenario=asset(root/'scenario/scenario.json'),
        target_track=asset(root/'scenario/target_track.npz'),baseline=asset(root/'baseline_freeze.json'))
    write(root/'plan.json',dict(schema='light_native_cutin551_diagnostic_v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        code=code_pins(),runtime_code=runtime_code_pins(),assets=assets,seed=20261109,common_scene_token='light551_'+root.name,
        controls=int(round(CONFIG['observation_end_s']*CONFIG['fps'])),termination=TERMINATION,
        coordinate_contract='raw virtual lidar right/forward -> declared mount -> local forward/right -> native acc m/s2 + steering rate rad/s',
        prefix_policy='fresh rendered prefix inference; replay verified historical common actions; model history never reset at fork',
        old_snapshots_reused=False,reason='old heavy-MPC/partial-tensor snapshots are incompatible with the frozen light baseline',
        default_version_modified=False,online_truth_allowed=False,patch_initially=None,
        permitted_patch_inputs=['current/past UniAD outputs','declared current sensor-derived geometry_candidate_boxes','ego speed','ego steer','dt'],
        forbidden_patch_inputs=['scene id','absolute step trigger','target script','future actor truth','reference route future'],
        native_score_streams=['raw_uniad planning','adopted reference planning','actual executed future path where recorded'],
        gpu_executed=False))
    write(root/'status.json',dict(stage='baseline_cpu_verified_and_frozen',gpu_required=['original_baseline','A_B_intervention_pair'],patch=None,
        physical_hd_road_boundary_available=False))
    return dict(prepared=True,output=str(root),baseline_passed=True,scenario_passed=scenario['passed'],gpu_executed=False)


def check(output):
    root=Path(output).resolve();plan=read(root/'plan.json')
    for p,digest in plan['code'].items():
        if sha(p)!=digest:raise ValueError('Frozen experiment implementation changed: '+p)
    if plan.get('runtime_code')!=runtime_code_pins():raise ValueError('Frozen neural/renderer/worker runtime sources changed')
    baseline=read(root/'baseline_freeze.json')
    if baseline['code']!=baseline_pins():raise ValueError('Fixed tracker changed after freeze')
    for a in plan['assets'].values():verify(a)
    verify(baseline['verification'])
    return root,plan


def completed_arm(root,name):
    path=root/name/'run_report.json'
    if not path.exists():return False
    report=read(path)
    if report['status'] not in ('completed','terminated'):raise ValueError('Existing failed/incomplete arm retained; inspect '+str(path))
    check(root)
    if report.get('experiment_plan')!=asset(root/'plan.json') or report.get('tracker')!=read(root/'baseline_freeze.json'):raise ValueError('Existing arm is not compatible with this frozen light baseline')
    verify(report['experiment_plan']);verify(report['actor_tracks_asset'])
    for row in report['trace']:
        for key in ['perception_asset','motion_prediction_asset','reference_asset']:verify(row[key])
        for camera,digest in row['image_sha256'].items():
            if sha(root/name/f'steps/{row["index"]:03d}'/(camera+'.png'))!=digest:raise ValueError('Cached arm observation changed')
    return True


def run_and_score(root,name,steps,common=None,candidate=None):
    from runtime import run_arm
    from scoring import score_arm
    from diagnostics import arm_summary
    if completed_arm(root,name):
        print('Reusing existing arm:',name,flush=True)
    else:
        report=run_arm(root,name,steps,common,candidate)
        # Bind the finished arm to the exact frozen experiment and model weights.
        check(root);report['model_weights_unchanged']=True;report['experiment_plan']=asset(root/'plan.json');write(root/name/'run_report.json',report)
    report=read(root/name/'run_report.json')
    if report['status']=='failed':
        raise RuntimeError('Real closed-loop arm failed: '+name+'; '+str(report.get('error'))+'; inspect '+str(root/name/'error.log')+' and '+str(root/name/'uniad_worker.log'))
    path=root/name/'native_eval/native_scores.json'
    if not path.exists():score_arm(root,name)
    summary=arm_summary(root/name)
    if summary['status']=='failed':raise RuntimeError('Real closed-loop arm failed: '+name)
    return summary


def original_baseline(output):
    root,plan=check(output)
    summary=run_and_score(root,'A',plan['controls'])
    gate=not summary['any_anomaly'] and summary['controls']==plan['controls']
    write(root/'baseline_gpu_admission.json',dict(passed=gate,summary=asset(root/'A/summary.json'),
        first_3s_observed=True if summary['elapsed_simulation_s']>=3 else False,tracker_frozen_unchanged=True,
        physical_road_boundary_verified=False,does_not_certify_avoidance=True))
    if not gate:raise RuntimeError('Original A has baseline anomaly/short run; diagnose it before intervention')
    return dict(baseline_verified=True,summary=summary)


def pair(output):
    root,plan=check(output)
    admission=root/'baseline_gpu_admission.json'
    if not admission.exists() or not completed_arm(root,'A'):original_baseline(root)
    gate=read(admission);verify(gate['summary'])
    if not gate['passed']:raise ValueError('Original baseline not admitted; do not launch B')
    run_and_score(root,'B',plan['controls'],root/'A')
    from diagnostics import paired,artifacts
    result=paired(root);artifacts(root)
    write(root/'status.json',dict(stage='scene_pair_diagnosed',conclusion=result['conclusion'],patch_generated=False,
        next_stage='repeat B and identify a mechanism only if B alone has an anomaly'))
    return result


def repeat_b(output):
    root,plan=check(output);diagnosis=read(root/'comparison_A_B.json')
    if diagnosis['summaries']['A']['any_anomaly'] or not diagnosis['summaries']['B']['any_anomaly']:
        raise ValueError('Repeat is only needed for a B-specific anomaly; no forced patch')
    run_and_score(root,'B_repeat1',plan['controls'],root/'A')
    from diagnostics import paired,artifacts
    result=paired(root,'A','B_repeat1');artifacts(root,'A','B_repeat1');return result


def candidate_check(output):
    root,_=check(output);path=root/'candidate.json'
    if not path.exists():raise ValueError('No evidence-supported candidate exists; analyze A/B first, do not manufacture a patch')
    candidate=read(path);required={'status','module','mechanism','allowed_perception_fields','supporting_evidence','unexcluded_explanations','trigger','modification_location','parameter_basis','expected_benefit','possible_regressions','rollback_conditions','no_privileged_inputs'}
    if set(candidate)!=required or candidate['status']!='frozen_for_validation' or candidate['no_privileged_inputs'] is not True:raise ValueError('Incomplete external patch specification')
    verify(candidate['module']);verify(candidate['mechanism'])
    mechanism=read(candidate['mechanism']['path'])
    if mechanism.get('status')!='repeatable_B_only_mechanism_supported':raise ValueError('Mechanism not supported by repeats')
    for name in ['comparison_A_B.json','comparison_A_B_repeat1.json']:
        d=read(root/name)
        if d['summaries']['A']['any_anomaly'] or not d['summaries'][d['right']]['any_anomaly'] or not d['common_prefix_complete'] or not d['intervention_verified_in_camera_observation'] or not d.get('target_entered_actual_ego_front_corridor'):
            raise ValueError('Candidate lacks a verified, repeatable B-only failure')
    for a in candidate['supporting_evidence']:verify(a)
    allowed={'trajectory','boxes','scores','labels','occupancy','occupancy_contract_json','box_convention','decoded_box_origin',
        'drivable_mask','drivable_bounds_xy','drivable_schema','drivable_axes','geometry_candidate_boxes','candidate_boxes','candidate_scores','candidate_labels',
        'motion_all_traj_preds','motion_all_traj_scores','motion_track_scores','motion_motion_query_boxes','motion_motion_query_scores','motion_motion_query_labels','motion_motion_query_bbox_index','motion_sdc_last_query'}
    if not set(candidate['allowed_perception_fields'])<=allowed:raise ValueError('Patch requests undeclared privileged inputs')
    tree=ast.parse(Path(candidate['module']['path']).read_text())
    banned={'open','eval','exec','compile','__import__','globals','locals','getattr','setattr'}
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names=[n.name for n in node.names] if isinstance(node,ast.Import) else [node.module or '']
            if any(n.split('.')[0] not in {'numpy','math'} for n in names):raise ValueError('Patch import exceeds pure numerical interface')
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in banned:raise ValueError('Patch requests ambient I/O/introspection')
        if isinstance(node,ast.Attribute) and (node.attr.startswith('__') or node.attr in {'load','save','loadtxt','savetxt','fromfile','tofile','memmap','load_library'}):raise ValueError('Patch requests external file access')
        if isinstance(node,ast.Subscript):
            key=node.slice.value if isinstance(node.slice,ast.Constant) else None
            if key in {'scene','scene_id','step','source_frame','absolute_time','target_script','ground_truth'}:raise ValueError('Patch requests scene/time/script privileged triggers')
    return candidate


def patch_pairs(output):
    root,plan=check(output);candidate=candidate_check(root)
    try:
        run_and_score(root,'B_patch_on',plan['controls'],root/'A',candidate)
        run_and_score(root,'A_patch_on',plan['controls'],root/'A',candidate)
    except BaseException as exc:
        write(root/'patch_validation.json',dict(conclusion='unresolved_validation_run_failed',error=repr(exc),
            candidate=asset(root/'candidate.json'),adopted_as_default=False,rollback=True,
            rollback_action='retain original fixed light baseline; candidate remains disabled',default_version_changed=False))
        raise
    from diagnostics import paired,artifacts
    b=paired(root,'B','B_patch_on');a=paired(root,'A','A_patch_on');artifacts(root,'B','B_patch_on');artifacts(root,'A','A_patch_on')
    from runtime import TERMINATION
    off=a['summaries']['A'];on=a['summaries']['A_patch_on']
    progress_ok=on['progress_along_initial_forward_m']>=TERMINATION['normal_A_min_progress_ratio']*off['progress_along_initial_forward_m']
    stops_ok=on['stopped_seconds']<=off['stopped_seconds']+TERMINATION['normal_A_max_extra_stopped_s']
    def executed_hd(summary):
        scores=(summary.get('native_scores') or {}).get('scores') or {}
        return scores.get('executed_path',{}).get('hdscore')
    hd_off=executed_hd(off);hd_on=executed_hd(on)
    hd_ok=bool(hd_off and hd_on and hd_on['lower']>=hd_off['lower']-TERMINATION['normal_A_max_native_execution_hd_drop'])
    normal_regressed=on['any_anomaly'] or not(progress_ok and stops_ok and hd_ok)
    observed_modification=b['summaries']['B_patch_on']['external_reference_changes']>0
    improved=b['summaries']['B']['any_anomaly'] and not b['summaries']['B_patch_on']['any_anomaly'] and observed_modification
    recovery_unresolved=b['summaries']['B_patch_on']['stop_recovery_unresolved']
    conclusion=('benefit_provisional_recovery_observation_unresolved' if recovery_unresolved else 'effective_on_development_pair_unseen_validation_required') if improved and not normal_regressed else 'ineffective_or_regressed_rollback'
    result=dict(conclusion=conclusion,normal_A_regressed=normal_regressed,B_anomaly_resolved=improved,
        candidate=asset(root/'candidate.json'),adopted_as_default=False,rollback=not(improved and not normal_regressed),
        normal_A_checks=dict(progress=progress_ok,stopping=stops_ok,native_executed_hd=hd_ok),
        candidate_reference_modification_observed=observed_modification,
        recovery_observation_unresolved=recovery_unresolved,
        unseen_timing_gap_or_scene_test_required=True)
    write(root/'patch_validation.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','check','baseline','pair','repeat-B','diagnose','render','score','candidate-check','patch-pairs'])
    p.add_argument('--output',default=str(DEFAULT));p.add_argument('--arm',default='A');args=p.parse_args()
    if args.stage=='prepare':result=prepare(args.output)
    elif args.stage=='check':
        root,plan=check(args.output);configure()
        from native_car_edit import make_types
        make_types()
        import importlib.util,torch
        spec=importlib.util.spec_from_file_location('native_scorer_preflight',SCORER);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        if not callable(module.hugsim_evaluate):raise ValueError('Original native scorer unavailable')
        result=dict(verified=True,controls=plan['controls'],gpu_not_required=True,all_delayed_session_types_imported=True,
            native_scorer_imported=True,cuda_available=torch.cuda.is_available())
    elif args.stage in ['baseline','pair','repeat-B','candidate-check','patch-pairs']:
        result={'baseline':original_baseline,'pair':pair,'repeat-B':repeat_b,'candidate-check':candidate_check,'patch-pairs':patch_pairs}[args.stage](args.output)
    else:
        root,_=check(args.output)
        if args.stage=='diagnose':
            from diagnostics import paired
            result=paired(root)
        elif args.stage=='render':
            from diagnostics import artifacts
            result=artifacts(root)
        else:
            from scoring import score_arm
            result=score_arm(root,args.arm)
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
