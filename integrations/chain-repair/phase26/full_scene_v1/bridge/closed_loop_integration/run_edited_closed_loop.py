"""Isolated task owner: JSON edits -> SAME framework scene -> frozen versioned driver loop.

This adapter replaces orchestration/sensors, never the frozen control algorithm.
It deliberately calls the loop API: the source record wrapper reloads a native
scene and rejects its own snapshot=null template. No branch snapshot is claimed.
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
import datetime
import hashlib
import json
import os
pass  # Path is provided by the portable boundary above
import sys
import time
import uuid

HERE=Path(__file__).resolve().parent
BUNDLE=HERE.parents[1]
SNAPSHOT_BRIDGE=HERE.parent
PHASE=BUNDLE.parent
CR=PHASE.parent
ROOT=next(p for p in HERE.parents if (p/'integrations/sr-hugsim/scene_api.py').is_file())
SOURCE=CR/'phase34/global_control_v1'
PROFILES={
    'phase34_global_control551_v1':(CR/'phase34/global_control_v1',False),
    'phase34_global_control551_geometry_v1':(CR/'phase34/global_control_v1',True),
    'phase35_event_mpc551_geometry_v1':(CR/'phase35/event_mpc_v1',True),
    'phase37_uncertainty551_v1':(CR/'phase37/uncertainty_closed_v1',False),
    'phase37_uncertainty551_geometry_v1':(CR/'phase37/uncertainty_closed_v1',True),
    'phase38_runtime_road551_geometry_v1':(PHASE/'road_runtime_v1',True),
    'framework_memory551_geometry_v1':(PHASE/'memory_lifecycle_v1',True),
    'framework_memory_envelope551_geometry_v2':(PHASE/'motion_envelope_v2',True),
    'framework_memory_search551_geometry_v3':(PHASE/'motion_search_v3',True),
}

def validation_scope(task):
    if task.get('validation')=='full_scene_19p8s_v1':return dict(name='full_scene_19p8s_v1',steps=198,max_wall_seconds=43200)
    if task.get('driver','').startswith('phase37_'):return dict(name='uncertainty_3s_v1',steps=30,max_wall_seconds=7200)
    if task.get('validation')=='diagnostic_10s_v1':return dict(name='diagnostic_10s_v1',steps=100,max_wall_seconds=14400)
    if task.get('validation')=='avoidance_18s_v1':return dict(name='avoidance_18s_v1',steps=180,max_wall_seconds=21600)
    if task.get('driver','').startswith('phase38_'):return dict(name='road_runtime_3s_v1',steps=30,max_wall_seconds=7200)
    if task.get('driver')=='framework_memory_search551_geometry_v3':return dict(name='motion_search_3s_v3',steps=30,max_wall_seconds=7200)
    if task.get('driver')=='framework_memory_envelope551_geometry_v2':return dict(name='motion_envelope_3s_v2',steps=30,max_wall_seconds=7200)
    if task.get('driver','').startswith('framework_memory'):return dict(name='memory_lifecycle_3s_v1',steps=30,max_wall_seconds=7200)
    return dict(name='linkage_3s_v1',steps=30,max_wall_seconds=3600)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read_task(path):
    task=_hugsim_expand(json.loads(Path(path).read_text()))
    if not {'schema','driver','edits'}<=set(task) or set(task)-{'schema','driver','edits','validation'} or task['schema']!='edited_hugsim_closed_loop_v1' or not isinstance(task['driver'],str) or task['driver'] not in PROFILES:
        raise ValueError('Expected fixed edited_hugsim_closed_loop_v1 task')
    if 'validation' in task:
        allowed=task['validation'] in ('avoidance_18s_v1','diagnostic_10s_v1') and task['driver']=='phase35_event_mpc551_geometry_v1'
        allowed=allowed or (task['validation']=='diagnostic_10s_v1' and task['driver'] in ('phase38_runtime_road551_geometry_v1','framework_memory551_geometry_v1','framework_memory_envelope551_geometry_v2','framework_memory_search551_geometry_v3'))
        allowed=allowed or (task['validation']=='full_scene_19p8s_v1' and task['driver']=='framework_memory_search551_geometry_v3')
        if not allowed:raise ValueError('Unsupported validation scope/profile')
    if not isinstance(task['edits'],list):raise ValueError('edits must be an array')
    # Fixed source insertion establishes the source diagnostic reference. Additional
    # native/signal/actor edits use the existing editor and its validation rules.
    if not task['edits'] or task['edits'][0]!={'source_insert':True}:raise ValueError('First edit must select the pinned source insertion')
    for value in task['edits'][1:]:
        if not isinstance(value,dict) or value.get('operation') not in ('translate','rotate','set_signal','set_signal_program','restore_signal','set_actor_trajectory','restore_actor_trajectory'):
            raise ValueError('Additional edits must use native editor operations')
    return task

def load_driver(source=None):
    global SOURCE
    if source is not None:SOURCE=Path(source)
    # A dedicated process keeps unqualified source imports away from editor hosts.
    if 'run_driver' in sys.modules:raise RuntimeError('Driver already configured')
    sys.path.insert(0,str(SOURCE))
    import launch
    launch.verify()
    if (SOURCE/'diagnostic_admission.py').exists():
        from diagnostic_admission import validate_launch
        validate_launch(SOURCE)
    import run_driver
    run_driver.configure()
    from proposal_geometry_audit import install as install_proposal_audit
    install_proposal_audit(run_driver.optimized_controller.PlanningGuard)
    return run_driver,launch

def provenance():
    files=[p for folder in (HERE,SNAPSHOT_BRIDGE/'framework_backend',SNAPSHOT_BRIDGE/'unified') for p in folder.glob('*.py')]
    files.extend((BUNDLE/'framework_repo').rglob('*.py'))
    return {str(p.relative_to(ROOT)):digest(p) for p in files}

def preflight(task_path):
    from full_scene_contract import verify
    verify(BUNDLE)
    task=read_task(task_path)
    source,geometry=PROFILES[task['driver']]
    driver,launch=load_driver(source)
    if geometry:
        from geometry_acceleration import accelerated
        accelerated.install()
    valid=_hugsim_expand(json.loads((SOURCE/'validation.json').read_text()))
    if valid.get('passed') is not True or not valid.get('required_checks') or not all(v is True for v in valid['required_checks'].values()):raise RuntimeError('Source CPU acceptance missing')
    arm=_hugsim_expand(json.loads((SOURCE/'arm.json').read_text()))
    if arm['steps']!=30 or arm['start_frame']!=0 or arm['initial_speed_mps']!=8 or arm['arm']!='edited':raise RuntimeError('Source scope changed')
    scope=validation_scope(task)
    spec=driver.h.SceneSpec.read(arm['scene_manifest'])
    if arm['start_frame']+scope['steps']>=spec.frames:raise ValueError('Validation exceeds source time window')
    if scope['name']=='full_scene_19p8s_v1':
        from full_scene_contract import validate_scope
        validate_scope(task,scope,spec,arm)
    # Pin the exact implementation that defines ED as sum(w*z)/sum(w).
    import importlib.util
    gs=importlib.util.find_spec('gsplat')
    if gs is None or not gs.submodule_search_locations:raise RuntimeError('gsplat missing')
    renderer=Path(next(iter(gs.submodule_search_locations)))/'rendering.py'
    code=renderer.read_text()
    kernels=[renderer.parent/'cuda/csrc'/name for name in ('fully_fused_projection_fwd.cu','fully_fused_projection_packed_fwd.cu')]
    for kernel in kernels:
        if ' = mean_c.z;' not in kernel.read_text():raise RuntimeError('Unreviewed camera-Z projection')
    if 'render_colors[..., -1:] / render_alphas.clamp(min=1e-10)' not in code:raise RuntimeError('Unreviewed expected depth implementation')
    scheduler=getattr(driver.h,'_mpc_scheduler',None)
    if task['driver'].startswith(('phase35_','phase37_','phase38_','framework_memory')) and scheduler is None:raise RuntimeError('Event scheduler is not configured')
    schedule_config=dict(enabled=scheduler is not None,policy=dict(scheduler.policy) if scheduler is not None else None)
    uncertainty=dict(enabled=False)
    if task['driver'].startswith(('phase37_','phase38_','framework_memory')):
        import importlib
        module_names=('ground_reference','depth_consistency','depth_ambiguity','motion_memory','motion_observer','time_local_motion','spatial_reference','motion_policy','native_objective','context_factory')
        bindings={}
        for name in module_names:
            module=importlib.import_module(name);path=Path(module.__file__).resolve()
            if path.parent!=SOURCE:raise RuntimeError('Uncertainty module shadowed: '+name)
            bindings[name]=dict(path=str(path),sha256=digest(path))
        import depth_sensor_map
        if driver.h.ObstacleMemory.__module__!='motion_memory' or depth_sensor_map.DepthObstacleMemory.__module__!='depth_consistency' or driver.h.RecoveryController.__module__!='motion_policy':raise RuntimeError('Incomplete uncertainty driver wiring')
        uncertainty=dict(enabled=True,components=bindings,whole_package=True,policy_changed_vs_phase35=True,performance_only_change=False,source_diagnostic_admission_checked=True,admitted_controls=30,admitted_wall_seconds=7200,global_repair_verified=False,road_boundary_repair_included=False)
    runtime=dict(enabled=False)
    if task['driver'].startswith(('phase38_','framework_memory')):
        runtime=dict(enabled=True,policy=_hugsim_expand(json.loads((SOURCE/'runtime_policy.json').read_text())),road_priority=_hugsim_expand(json.loads((SOURCE/'road_priority_policy.json').read_text())),search_changed=True,global_optimum_claimed=False,source_phase38_frozen=False,isolated_validated_fork=True)
        if runtime['policy'].get('scaled_slsqp') or not runtime['policy'].get('consensus_search') or not runtime['road_priority'].get('enabled'):raise RuntimeError('Unexpected runtime/road policy')
        admitted=_hugsim_expand(json.loads(((BUNDLE/'experiment_scope.json') if scope['name']=='full_scene_19p8s_v1' else (SOURCE/'framework_scope.json')).read_text()))
        if scope['steps'] not in admitted['allowed_controls'] or scope['max_wall_seconds']!=admitted['wall_caps'][str(scope['steps'])]:raise RuntimeError('Unvalidated Framework scope')
        uncertainty.update(admitted_controls=scope['steps'],admitted_wall_seconds=scope['max_wall_seconds'],road_penalty_strengthened=True)
        if task['driver'].startswith('framework_memory'):
            policy=_hugsim_expand(json.loads((SOURCE/'memory_lifecycle_policy.json').read_text()))
            if policy['schema']!='observed_memory_lifecycle_v1':raise RuntimeError('Unknown memory lifecycle')
            for name in ('memory_visibility','expired_memory_guard'):
                module=importlib.import_module(name);path=Path(module.__file__).resolve()
                if path.parent!=SOURCE:raise RuntimeError('Memory repair module shadowed')
            runtime['memory_lifecycle']=dict(enabled=True,policy=policy,current_action_full_backup_required=True,missing_observation_is_free_space=False)
            if task['driver'] in ('framework_memory_envelope551_geometry_v2','framework_memory_search551_geometry_v3'):
                motion_policy=_hugsim_expand(json.loads((SOURCE/'motion_tube_policy.json').read_text()))
                if motion_policy.get('velocity_interval_policy')!='source_hull_plus_metric_floor_v2' or motion_policy.get('actor_acceleration_budget_mps2')!=10. or motion_policy.get('complete_current_action_braking_envelope_retained') is not True:raise RuntimeError('Motion envelope policy changed')
                runtime['motion_envelope']=dict(policy=motion_policy,empirical_speed_interval_changed=True,physical_safety_certificate=False)
                if task['driver']=='framework_memory_search551_geometry_v3':
                    from search_acceptance import validate as validate_search
                    validate_search(SOURCE)
                    runtime['search_migration']=dict(source='phase40/mpc_runtime_v2',adaptive_consensus=True,both_lateral_starts_required=True,original_iteration_budget_preserved=True,own_paired_cpu_acceptance=True,new_gpu_speedup_verified=False)
    audit=dict(runtime_road_bundle=runtime,bbox_diagnostic=dict(enabled=True,proposal_geometry_receipt_enabled=True,reference_scope='inserted static vehicle',ground_truth_used_by_controller=False),uncertainty_bundle=uncertainty,validation_scope=scope,requested_controls=scope['steps'],planning_schedule=schedule_config,driver_profile=task['driver'],geometry_acceleration=accelerated.audit() if geometry else dict(enabled=False),source_manifest_sha256=digest(SOURCE/'manifest.json'),
        source_validation_sha256=digest(SOURCE/'validation.json'),gsplat_rendering=str(renderer),gsplat_sha256=digest(renderer),
        sensor_implementation={str(path):digest(path) for path in [renderer,*kernels]},adapter_implementation=provenance(),source_controls=30,source_initial_speed_mps=8,
        source_freeze_verified=True,gpu_executed=False,formal_evaluation_admitted=False,adapter_snapshot=str(BUNDLE),full_scene_controls=198,full_scene_observations=199)
    from full_scene_contract import bindings
    audit['frozen_bindings']=bindings(BUNDLE)
    return task,driver,launch,audit

def start_worker(h,worker,out):
    """Same frozen worker/arguments; retain handle if readiness fails."""
    import subprocess
    env=dict(os.environ,BASE=str(ROOT),PYTHONUNBUFFERED='1',
        BEV_EXPERIMENT_SEED=str(h.SEED),PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python',
        UNIAD_INIT_TRACE='1',UNIAD_STARTUP_REPORT=str(worker/'startup.json'))
    for key in ('LD_LIBRARY_PATH','PYTHONPATH','PYTHONHOME'):env.pop(key,None)
    log=(out/'uniad_worker.log').open('x');process=None;began=time.monotonic()
    try:
        process=subprocess.Popen([str(ROOT/'tools/pixi-uniad-legacy/.pixi/envs/default/bin/python'),
            str(h.WORKER),'--job',str(worker),'--planning-bev','current',
            '--model-profile','calibration_geometry_v1','--diagnostic-closed-loop'],
            env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        ready=h.wait_json(worker/'ready.json',process,timeout=1200.)
        if ready['planning_bev']!='current':raise RuntimeError('Wrong planning profile')
        return process,log,ready,time.monotonic()-began
    except BaseException:
        h._stop_worker(worker,process,log)
        raise

def run(task,driver,launch,audit,out):
    import numpy as np
    from scene_edit_session import UnifiedSceneSession
    from sensor_backend import ClosedLoopFrameworkBackend
    h=driver.h
    out.mkdir(parents=True,exist_ok=False)
    (out/'comparison_frames').mkdir()
    for name in h.THREE_CAMERAS:(out/'closed_loop'/name).mkdir(parents=True)
    h.atomic_json(out/'task.json',task);h.atomic_json(out/'integration_provenance.json',audit)
    request=_hugsim_expand(json.loads((SOURCE/'request_template.json').read_text()));request['output']=str(out)
    arm=h._load_arm_request(request['arm_request']);scope=validation_scope(task);arm=dict(arm,steps=scope['steps'])
    request['termination']=dict(request.get('termination',{}),max_wall_seconds=scope['max_wall_seconds'])
    h.atomic_json(out/'arm_request.json',arm)
    setup=h._setup(arm,out);spec=setup['spec'];implementation=h._implementation(setup['manifest'],spec)
    report=h._base_report(arm,spec,setup,'edited_framework_adapter',task['driver'])
    report.update(implementation_before=implementation(),checkpoint_before=h.fingerprint(setup['checkpoint']),
        editing_closed_loop_adapter=True,execution_backend='framework_renderer',source_control_bundle=str(SOURCE),driver_profile=task['driver'],runtime_road_bundle=audit.get('runtime_road_bundle',dict(enabled=False)),uncertainty_bundle=audit.get('uncertainty_bundle',dict(enabled=False)),planning_schedule_config=audit.get('planning_schedule',dict(enabled=False)),geometry_acceleration=audit.get('geometry_acceleration',dict(enabled=False)),
        sensor_contract='gsplat_expected_camera_z_alpha_v1',depth_variance_available=False,
        observation_attitude_policy='source_original_rig',online_training=False,formal_evaluation_admitted=False,
        integration_acceptance_passed=False,closed_loop_driving=False,validation_scope=scope,
        bbox_diagnostic_required=True,proposal_geometry_receipt_enabled=True)
    driver.base._TIMING_OUTPUT=out/'mpc_timing.json'
    worker=out/'uniad_worker';process=log=None;began=time.monotonic()
    session=UnifiedSceneSession(backend_factory=ClosedLoopFrameworkBackend)
    try:
        session.load_scene(out/'scene_manifest.json')
        insertion=dict(setup['insert'],schema='scene_edit_v1',object_id='managed:'+setup['insert']['object_id'],expected_revision=0)
        edits=[insertion]+task['edits'][1:]
        for edit in edits:
            command=dict(edit)
            command.setdefault('expected_revision',session.revision);command.setdefault('expected_generation',session.generation)
            session.edit(command)
        revision=session.revision;generation=session.generation;identity=id(session.backend)
        h.atomic_json(out/'edit_receipt.json',dict(generation=generation,edit_revision=revision,journal=session.edit_journal()))
        report.update(scene_generation=generation,edit_revision=revision,scene_loads=1,applied_edits=edits)
        state=h._fresh_driver_state(setup,spec)
        # Token is unique to this edited generation, no pre-edit temporal state.
        state['token']=generation+':'+str(revision)
        model_packet=h._model_packet_factory(setup,spec,dict(state))
        sensor_rows=[]
        def checked_packet(packet):
            if id(session.backend)!=identity or session.generation!=generation or session.revision!=revision:
                raise RuntimeError('Edited scene ownership/revision changed during driving')
            if packet['history_token']!=generation or packet['edit_revision']!=revision:raise RuntimeError('Stale observation')
            obs,info=model_packet(packet)
            if info['depth_encoding']!='expected_camera_z_v1':raise RuntimeError('Unknown control depth')
            sensor_rows.append(dict(frame=packet['source_frame'],timestamp_s=packet['timestamp_s'],generation=generation,revision=revision,
                cameras={name:dict(shape=list(v['rgb'].shape),alpha_valid_fraction=float(np.mean(v['depth_alpha']>=.9)),
                    positive_depth_fraction=float(np.mean(v['depth']>0))) for name,v in packet['cameras'].items()}))
            h.atomic_json(out/'sensor_audit.json',sensor_rows)
            return obs,info
        ctx=dict(spec=spec,setup=setup,arm_request=arm,implementation=implementation,label=task['driver'],
            model_packet=checked_packet,snapshot_after_index=None,snapshot_dir=out/'unused_snapshot',
            applied_edits=edits,state=state,plan_patch=None,reliability_patch=h._load_reliability_patch(request),
            termination=h._termination_cfg(request),_loop_wall0=began)
        if h.reference_check(setup['initial_native'],setup['reference'],setup['vehicle'])['overlap']:raise RuntimeError('Initial overlap')
        (worker/'step_000').mkdir(parents=True)
        first=session.render(setup['relative'][setup['start']],setup['start']/spec.fps)
        h.atomic_pickle(worker/'step_000/observation.pkl',checked_packet(first));ctx['first_render']=first
        process,log,ready,boot=start_worker(h,worker,out)
        report['worker_ready']=ready;report['timing']['worker_boot_seconds']=boot
        h._control_loop(ctx,out,session,report,worker,process,0,lambda:True)
        controls=len(report['trace']);observations=len(report['observations'])
        checks=dict(full_control_window=controls==scope['steps'],full_observation_window=observations==scope['steps']+1,
            physics_feedback=controls>0 and all(row.get('physics_integration_verified') is True for row in report['trace']),
            temporal_history=controls>0 and all(row.get('temporal',{}).get('history_bev_present')==(i>0) and
                row.get('temporal',{}).get('metadata_cache_is_tracking_cache') is True for i,row in enumerate(report['trace'])))
        report['integration_checks']=checks
        report.update(status='completed',closed_loop_driving=controls>0,
            integration_acceptance_passed=all(checks.values()),
            integration_scope=f"{scope['steps']} controls; {scope['name']}; avoidance judged separately after driving")
        report.setdefault('termination_reason','window_completed')
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),integration_acceptance_passed=False)
        # Persist before CUDA/worker cleanup and potentially slow integrity I/O.
        h.atomic_json(out/'run_report.json',report)
        h.atomic_json(out/'integration_acceptance.json',dict(passed=False,reason=repr(exc),cleanup_pending=True,formal_evaluation_admitted=False))
        raise
    finally:
        try:
            if worker.exists():h._stop_worker(worker,process,log)
        except BaseException as exc:
            report.update(worker_cleanup_error=repr(exc),integration_acceptance_passed=False)
        try:session.close()
        except BaseException as exc:
            report.update(scene_cleanup_error=repr(exc),integration_acceptance_passed=False)
        report['implementation_after']=implementation();report['implementation_unchanged']=report['implementation_before']==report['implementation_after']
        report['checkpoint_after']=h.fingerprint(setup['checkpoint']);report['checkpoint_unchanged']=report['checkpoint_before']==report['checkpoint_after']
        report['adapter_unchanged']=audit['adapter_implementation']==provenance()
        geometry=audit.get('geometry_acceleration',{})
        if geometry.get('enabled'):
            from geometry_acceleration import accelerated
            report['geometry_acceleration']=accelerated.audit()
            report['geometry_implementation_unchanged']=all(digest(path)==value for path,value in geometry['implementation'].items())
        else:report['geometry_implementation_unchanged']=True
        if not all(report.get(k) for k in ('implementation_unchanged','checkpoint_unchanged','adapter_unchanged')):report['integration_acceptance_passed']=False
        try:
            launch.verify();report['source_freeze_unchanged']=True
        except Exception as exc:
            report.update(source_freeze_unchanged=False,source_freeze_error=repr(exc))
        report['sensor_implementation_unchanged']=all(digest(path)==expected for path,expected in audit.get('sensor_implementation',{}).items())
        checks=report.setdefault('integration_checks',dict(full_control_window=False,full_observation_window=False))
        checks.update({key:report.get(key) is True for key in ('implementation_unchanged','checkpoint_unchanged','adapter_unchanged','source_freeze_unchanged','sensor_implementation_unchanged','geometry_implementation_unchanged')})
        checks['completed_without_cleanup_error']=report.get('status')=='completed' and not any(key in report for key in ('worker_cleanup_error','scene_cleanup_error'))
        report['integration_acceptance_passed']=all(checks.values())
        h.atomic_json(out/'integration_acceptance.json',dict(checks=checks,passed=report['integration_acceptance_passed'],formal_evaluation_admitted=False))
        audit.update(gpu_executed=bool(report.get('observations')),integration_acceptance_passed=report.get('integration_acceptance_passed',False))
        h.atomic_json(out/'integration_provenance.json',audit)
        report['wall_seconds']=time.monotonic()-began
        h._score_run(report,setup)
        report['termination_policy']=f"{scope['steps']} source-time controls or original safety/error stops; wall cap {scope['max_wall_seconds']}s"
        h.atomic_json(out/'run_report.json',report)
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',type=Path,default=HERE/'examples/source_insert551_phase35_geometry.json');parser.add_argument('--check',action='store_true');parser.add_argument('--output',type=Path);parser.add_argument('--drive-only',action='store_true');parser.add_argument('--print-wall-limit',action='store_true')
    args=parser.parse_args()
    if args.print_wall_limit:
        print(validation_scope(read_task(args.task))['max_wall_seconds']);return
    task,driver,launch,audit=preflight(args.task)
    if args.check:
        if args.output:
            args.output.mkdir(parents=True,exist_ok=False)
            (args.output/'preflight.json').write_text(json.dumps(audit,indent=2)+'\n')
        print(json.dumps({k:v for k,v in audit.items() if k!='adapter_implementation'},indent=2))
        print('Adapter/framework files audited:',len(audit['adapter_implementation']));return
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; execute on GPU terminal')
    for path in (SNAPSHOT_BRIDGE/'unified',SNAPSHOT_BRIDGE/'framework_backend'):sys.path.insert(0,str(path))
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=(args.output or HERE/'runs'/stamp).resolve();print('OUTPUT:',out,flush=True)
    # Whole-episode deadline includes scene/worker boot. Child processes are
    # cleaned by the shell supervisor if initialization itself stalls.
    if args.drive_only:
        result=run(task,driver,launch,audit,out)
        if not result['integration_acceptance_passed']:raise RuntimeError('Incomplete driving acceptance')
        return
    simulation_exit=0
    try:
        result=run(task,driver,launch,audit,out)
        if not result['integration_acceptance_passed']:simulation_exit=1
    except BaseException:
        simulation_exit=1
        raise
    finally:
        from stage_results import finish
        result=finish(out,simulation_exit=simulation_exit)
        print(json.dumps(result,indent=2),flush=True)
    if result['exit_code']:raise SystemExit(result['exit_code'])
if __name__=='__main__':main()
