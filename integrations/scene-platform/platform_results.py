"""Normalize measured execution, effective settings, driving failure and cleanup."""

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
from platform_io import read,write,asset

def collect(task,profile,work,*,reused=False):
    work=Path(work);arm='B' if task['edits'] else 'A';folder=work/arm;r=read(folder/'run_report.json')
    trace=r['trace'];observations=r['observations'];requested=task['simulation']['control_steps']
    summary=read(folder/'summary.json') if (folder/'summary.json').exists() else {}
    scores=read(folder/'native_eval/native_scores.json') if (folder/'native_eval/native_scores.json').exists() else None
    audits=[]
    for p in sorted((folder/'uniad_worker').glob('step_*/native_collision_audit.json')):audits.append(read(p))
    native_enabled=bool(trace) and {t['index'] for t in trace}<={a.get('index') for a in audits} and all(a.get('enabled') is True and a.get('native_code_modified') is False for a in audits)
    cleanup_path=work/'platform_lifecycle/cleanup.json'
    cleanup=read(cleanup_path) if cleanup_path.exists() else dict(reset_verified=False,closed=None,recovery_required=True,reason='No platform-owned reset evidence')
    edit_path=work/'platform_edit_receipt.json'
    edit=read(edit_path) if edit_path.exists() else None
    actual_edited=bool(r.get('actor_intervention',{}).get('applied',r.get('actor_intervention',{}).get('object_id')))
    sync=r.get('actor_sync',{}).get('passed',False)
    checks=dict(real_closed_loop=r.get('real_closed_loop') is True,requested_budget_complete=len(trace)==requested and r['status']=='completed',
        source_clock_within_scene=all(o['index']+profile['source_start_frame']<=profile['source_end_frame'] for o in observations),
        actual_NN_and_history_evidence=bool(trace) and all(t.get('perception_asset') and t.get('temporal',{}).get('history_bev_present')==(t['index']>0) for t in trace),
        actor_state_synchronized=sync,instruction_edit_matches_run=bool(task['edits'])==actual_edited,
        actual_native_collision_enabled=native_enabled,model_weights_unchanged=r.get('model_weights_unchanged') is True,
        native_score_available=bool(scores and scores.get('scores')),reset_and_close_verified=cleanup.get('reset_verified') is True and cleanup.get('closed') is True)
    from optional_sensors import summarize
    sensors=summarize(task,work,r)
    if sensors['enabled']:checks['optional_sensor_recording_verified']=sensors['checks_passed']
    if not reused and task['edits']:checks['actual_edit_dispatch_receipt']=edit is not None
    semantic_failure=summary.get('any_anomaly',False)
    status='completed' if all(checks.values()) and not r.get('scene_cleanup_error') else 'terminated' if r['status']=='terminated' and checks['reset_and_close_verified'] else 'failed'
    raw_end=r.get('termination_reason')
    stop='requested_control_budget_complete' if checks['requested_budget_complete'] else raw_end
    artifacts=[dict(kind='actual_run_report',**asset(folder/'run_report.json'))]
    artifacts.extend(sensors['artifacts'])
    for name,kind in [('actor_tracks.npz','effective_actor_state'),('summary.json','driving_summary'),('native_eval/native_scores.json','native_plan_and_execution_scores')]:
        if (folder/name).exists():artifacts.append(dict(kind=kind,**asset(folder/name)))
    if edit_path.exists():artifacts.append(dict(kind='edit_receipt',**asset(edit_path)))
    if cleanup_path.exists():artifacts.append(dict(kind='reset_cleanup_receipt',**asset(cleanup_path)))
    presentation=work/'platform_presentation/artifacts.json'
    if presentation.exists():artifacts.extend(read(presentation)['artifacts'])
    failures=[]
    if r.get('error'):failures.append(dict(domain='driver_runtime',reason=r['error'],stage=r.get('current_stage')))
    if summary.get('any_anomaly'):failures.append(dict(domain='driving',reason='Recorded driving anomaly',details=summary.get('anomalies')))
    if summary.get('stop_recovery_unresolved'):failures.append(dict(domain='observation_limit',reason='Stop recovery unresolved at observation end; no successful recovery claimed'))
    if not checks['requested_budget_complete']:failures.append(dict(domain='termination',reason=raw_end,controls=len(trace),requested=requested))
    if cleanup.get('recovery_required'):failures.append(dict(domain='cleanup',reason=cleanup.get('error',cleanup.get('reason','reset verification failed'))))
    if not checks['actual_native_collision_enabled']:failures.append(dict(domain='configuration',reason='Native collision enabled evidence missing or disagrees with registered profile'))
    result=dict(schema='scene_task_result_v2',task_id=task['task_id'],status=status,
        context=dict(scene_id=task['scene_id'],driver_profile=task['simulation']['profile'],phase='closed' if cleanup.get('closed') else 'unknown',
            generation=cleanup.get('generation'),edit_revision=cleanup.get('revision'),weather='clear'),
        execution=dict(requested_controls=requested,executed_controls=len(trace),observations=len(observations),
            newly_executed_controls=0 if reused else len(trace),reused_actual_run=reused,
            control_executed=bool(trace) and not reused,source_actual_controls=len(trace),
            perception_connected=bool(trace),uniad_inference_executed=bool(trace) and not reused,
            source_uniad_inference_evidence=bool(trace),native_collision_audits=len(audits),native_collision_enabled=native_enabled,
            heavy_mpc_executed=r.get('heavy_mpc_executed'),sensor_policy=profile['sensor_policy'],
            wall_seconds=r.get('wall_seconds'),driving_wall_seconds=r.get('driving_wall_seconds'),
            termination_reason=stop,source_driver_termination_reason=raw_end,
            full_source_time_window_completed=len(trace)==profile['max_control_steps'] and r['status']=='completed',
            online_training=False,simulation_poses_from_actual_control_feedback=True),
        effective_configuration=dict(native_collision_enabled=native_enabled,
            native_collision_solver_calls=sum(a.get('native_method_calls',0) for a in audits),
            native_collision_skip_reasons=sorted(set(a.get('skip_reason') for a in audits if a.get('skip_reason'))),
            heavy_mpc_executed=r.get('heavy_mpc_executed'),raw_planning_stream=r.get('raw_uniad_stream_semantics')),
        optional_sensors=sensors,
        metrics=dict(native_scores=scores,max_one_step_tracking_error_m=summary.get('max_one_step_tracking_error_m'),
            progress_m=summary.get('progress_along_initial_forward_m'),stopped_seconds=summary.get('stopped_seconds'),
            recovery_unresolved=summary.get('stop_recovery_unresolved'),collision=summary.get('anomalies',{}).get('environment_collision')),
        acceptance=dict(engineering_checks=checks,engineering_passed=all(checks.values()),
            driving_success=False if semantic_failure else None,
            driving_assessment='anomaly observed' if semantic_failure else 'no anomaly recorded within this window; safe-driving certification unavailable',
            formal_evaluation_admitted=False,profile_driving_certification=False),
        failure_reasons=failures,artifacts=artifacts,cleanup=cleanup,recovery_required=cleanup.get('recovery_required',True),
        score_scope='raw stream is after native collision processing; adopted is final reference; executed scores use actual recorded future poses. These are distinct.',
        LLM_invoked=False)
    return result

def present(work,arm):
    import numpy as np
    from PIL import Image,ImageDraw
    import imageio_ffmpeg,subprocess
    work=Path(work);folder=work/arm;r=read(folder/'run_report.json');out=work/'platform_presentation';frames=out/'frames';frames.mkdir(parents=True,exist_ok=True)
    if (out/'artifacts.json').exists():return read(out/'artifacts.json')
    for i,o in enumerate(r['observations']):
        canvas=Image.new('RGB',(1440,360),'#171717');draw=ImageDraw.Draw(canvas)
        draw.text((8,6),f'Actual closed loop | t={o["timestamp"]:.1f}s | source frame={o["index"]+10} | speed={o["motion_state"][3]:.2f} m/s',fill='white')
        for col,name in enumerate(['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT']):
            with Image.open(folder/f'steps/{o["index"]:03d}'/(name+'.png')) as im:im.thumbnail((480,320));canvas.paste(im,(col*480,32))
        canvas.save(frames/f'{i:04d}.png')
    video=out/'actual_closed_loop_3CAM.mp4';binary=imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([binary,'-v','error','-n','-framerate','10','-i',str(frames/'%04d.png'),'-c:v','libx264','-threads','2','-pix_fmt','yuv420p',str(video)],check=True,timeout=180)
    subprocess.run([binary,'-v','error','-threads','1','-i',str(video),'-f','null','-'],check=True,timeout=60)
    # Same source-world coordinates used by the driver environment checker.
    ego=np.asarray([o['pose'] for o in r['observations']])[:,:2,3]
    target=np.asarray([o['environment']['target_world']['box'][:2] for o in r['observations'] if o['environment'].get('target_world')])
    points=np.concatenate([ego,target]) if len(target) else ego
    lo=points.min(axis=0)-3;hi=points.max(axis=0)+3;scale=min(800/(hi[0]-lo[0]),520/(hi[1]-lo[1]))
    graph=Image.new('RGB',(900,650),'white');pen=ImageDraw.Draw(graph)
    def line(values,color):
        coords=[(50+float(p[0]-lo[0])*scale,600-float(p[1]-lo[1])*scale) for p in values]
        if len(coords)>1:pen.line(coords,fill=color,width=4)
        for point in coords[::10]:pen.ellipse((point[0]-3,point[1]-3,point[0]+3,point[1]+3),fill=color)
    line(ego,'blue')
    if len(target):line(target,'red')
    pen.text((20,12),'Actual recorded paths | blue: ego; red: native vehicle 3 | source-world XY metres',fill='black')
    pen.text((20,32),f'Equal axis scale; x range {lo[0]:.1f}..{hi[0]:.1f}; y range {lo[1]:.1f}..{hi[1]:.1f}',fill='black')
    overlay=out/'actual_ego_target_paths.png';graph.save(overlay)
    value=dict(full_decode_passed=True,frames=len(r['observations']),artifacts=[dict(kind='actual_execution_video',**asset(video)),dict(kind='actual_execution_target_path_overlay',**asset(overlay))])
    write(out/'artifacts.json',value);return value
