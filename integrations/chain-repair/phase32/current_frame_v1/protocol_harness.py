"""Branch-simulation harness with an external plan-patch hook (phase28).

DIAGNOSTIC ONLY: N1 suspended, formal_evaluation_admitted=false. UniAD code and
weights are frozen; MPC/guard rules unchanged. The only added freedom is an
optional external patch applied to the neural reference trajectory (plan_raw ->
plan_patched) before the navigation anchor and MPC. The patch receives only
current/past model outputs and ego state; it is pinned by sha256 in the
request. Distinguish three trajectories in every report: UniAD raw plan,
patched reference, executed trajectory (only the last is scored).

Provenance: copy of phase27/branch_run.py with marked additions (search
"phase28").

Modes (request JSON, schema 'branch_experiment_v1'):
  record        — independent run 0..steps; after control step
                  `snapshot_after_index` it saves driver snapshot
                  (branch_driver_state_v2) and asks the worker for a tensor
                  snapshot (branch_worker_state_v1), then continues to the end.
  branch        — restore worker tensors + driver state from a snapshot,
                  pass a recorded-frame fidelity gate, optionally apply one
                  intervention (e.g. insert a vehicle), continue to the end.
  replay-branch — same as branch but rebuilds model history through the
                  existing prefix-replay path (closed_loop_resume); the
                  declared REPLAY BASELINE, not a true state restore.

Isolation: every mode runs in its own process; the snapshot is hashed before
and after each branch and must be byte-identical.
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

import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
pass  # Path is provided by the portable boundary above
from uuid import uuid4

_H = Path(__file__).resolve().parents[2] / 'phase28'  # frozen shared assets
_P11 = _H.parent / 'phase11'
_P27 = _H.parent / 'phase27'
_R = _H.parents[5]                            # repo root
sys.path[:0] = [str(_H), str(_P27), str(_P11), str(_R / 'integrations/sr-hugsim'),
                str(_R / 'waymo23-native-import'),
                str(_R / 'waymo23-native-import/waymo172_closed_loop_v1')]

import numpy as np

# Preload the reviewed phase11 modules (as run_arm.py does) so later transitive
# imports cannot bind the same module names to the waymo23-native-import copies.
import drivable_area as _p11_drivable_area
import road_route_seed as _p11_road_route_seed
assert 'phase11' in str(_p11_drivable_area.__file__) \
    and hasattr(_p11_drivable_area, 'filter_road_speckles'), \
    'phase11 drivable_area preload failed: ' + str(_p11_drivable_area.__file__)
assert 'phase11' in str(_p11_road_route_seed.__file__), \
    'phase11 road_route_seed preload failed: ' + str(_p11_road_route_seed.__file__)

from scene_api import BASE, SceneSpec, SceneSession, validate_action
from waymo_adapter import SR_TO_HUGSIM as T
from uniad_bridge import to_uniad, placement_reference

ADAPTER = BASE / 'waymo23-native-import'
from closed172_common import (atomic_json, atomic_pickle, wait_json, native_delta,
    emergency_stop_action, reference_check, integrate_state)
from heading_fix_v1.closed172_common import optimize
from build_source_sequence import training_lidar_mount, recorded_kinematics
from frozen_bev_experiment import fingerprint
import branch_state

SEED = 20261021
CAMERA = 'CAM_FRONT_RIGHT'
PERCEPTION_TOLERANCE = 1e-4   # same bar as the existing prefix-replay verifier
POSE_TOLERANCE = 1e-9

from comparison_video import compose, encode, compose_three_closed, THREE_CAMERAS
from planning_guard import ObstacleMemory, RecoveryController, POLICY

WORKER = _P27 / 'branch_worker.py'  # phase28: worker protocol unchanged


def _implementation(manifest, spec):
    audit_files = [ADAPTER / 'configs/planning.json', Path(__file__), WORKER,
        _P27 / 'branch_state.py',  # phase28: shared snapshot library
        (BASE / 'integrations/sr-hugsim').joinpath('comparison_video.py'),
        (BASE / 'integrations/sr-hugsim').joinpath('trajectory_baseline_score.py'),
        (BASE / 'integrations/sr-hugsim').joinpath('native_geometry_evaluation.py'),
        (BASE / 'integrations/sr-hugsim').joinpath('native_actor_evaluation.py'),
        (BASE / 'integrations/sr-hugsim').joinpath('closed_loop_resume.py'),
        (BASE / 'integrations/sr-hugsim').joinpath('dynamic_completion.py'),
        ADAPTER / 'planning_guard.py', ADAPTER / 'depth_sensor_map.py',
        ADAPTER / 'visibility_curtain.py', ADAPTER / 'depth_contract.py',
        ADAPTER / 'drivable_area.py', ADAPTER / 'road_route_seed.py',
        BASE / 'code/HUGSIM/gaussian_renderer/__init__.py',
        BASE / 'code/HUGSIM/gaussian_renderer/surface_sensor.py',
        ADAPTER / 'live_env.py',
        (BASE / 'integrations/sr-hugsim').joinpath('native_scene_backend.py'),
        (BASE / 'integrations/sr-hugsim').joinpath('scene_api.py'),
        (BASE / 'integrations/sr-hugsim').joinpath('uniad_bridge.py'),
        ADAPTER / 'depth_surface_geometry.py', ADAPTER / 'pseudo_lidar_adapter.py',
        ADAPTER / 'waymo172_closed_loop_v1/closed172_worker.py',
        BASE / 'code/UniAD_SIM/projects/mmdet3d_plugin/uniad/detectors/uniad_e2e.py',
        ADAPTER / 'waymo172_closed_loop_v1/closed172_common.py',
        ADAPTER / 'runtime_configuration.py', Path(manifest['vehicle_config'])]
    audit_files.extend([_P11 / name for name in
        ('drivable_area.py', 'road_route_seed.py', 'road_prediction_memory.py',
         'closed172_common.py', 'stop_diagnosis.py', 'bounded_visibility_geometry.py',
         'navigation_anchor.py', 'committed_route.py', 'route_progress.py')])
    if spec.source.get('planning_ground_skin'):
        audit_files.extend([Path(spec.source_report),
            Path(spec.source['planning_ground_skin']['data']),
            Path(spec.source['planning_ground_skin']['metadata'])])
    audit_files.extend([_H / "heading_fix_v1/closed172_common.py",
                        _H / "heading_fix_v1/heading.py"])
    audit_files.extend(_PATCH_MODULE_PATHS)
    return lambda: {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in audit_files}


def _termination_cfg(request):
    """Pre-registered termination rules (v3). Defaults match the frozen protocol."""
    cfg = dict(request.get('termination') or {})
    return dict(stall_steps=int(cfg.get('stall_steps', 10)),
                stall_speed_mps=float(cfg.get('stall_speed_mps', 0.05)),
                pass_hold_s=float(cfg.get('pass_hold_s', 0.5)),
                max_wall_seconds=float(cfg.get('max_wall_seconds', 9000.)),
                pass_check=bool(cfg.get('pass_check', False)),
                stall_check=bool(cfg.get('stall_check', True)))


def _load_arm_request(path):
    request = _hugsim_expand(json.loads(Path(path).read_text()))
    if request.get('schema') != 'unified_three_closed_dynamic_v1':
        raise ValueError('Branch experiments reuse the dynamic three-camera arm request')
    if request.get('time_policy') != 'source_time_only':
        raise ValueError('Branch experiments require source_time_only')
    if request.get('resume_checkpoint'):
        raise ValueError('Branch record runs start fresh; resume stays forbidden')
    return request


def _setup(arm_request, output):
    """Recompute every deterministic run input (identical in record and branch)."""
    output = Path(output).resolve()
    manifest = _hugsim_expand(json.loads(Path(arm_request['scene_manifest']).read_text()))
    manifest['render_scale'] = 1.
    atomic_json(output / 'scene_manifest.json', manifest)
    spec = SceneSpec.read(output / 'scene_manifest.json')
    start, steps = arm_request['start_frame'], arm_request['steps']
    if start >= spec.frames or start + steps >= spec.frames or not np.isclose(1 / spec.fps, spec.vehicle.dt):
        raise ValueError('Scene/clock/window mismatch')
    insert = _hugsim_expand(json.loads(Path(arm_request['insert_request']).read_text()))
    atomic_json(output / 'insert_request.json', insert)
    calibration = _hugsim_expand(json.loads(Path(insert['asset_calibration']).read_text()))
    reference = placement_reference(insert, calibration)
    atomic_json(output / 'asset_reference.json', reference)
    inputs = Path(spec.source['inputs'])
    raw_start = spec.source['start_timestep']
    poses = np.stack([np.loadtxt(inputs / 'ego_pose' / f'{raw_start + i:03d}.txt')
                      for i in range(spec.frames)])
    relative = np.linalg.inv(poses[0])[None] @ poses
    mount = training_lidar_mount(BASE / 'code/HUGSIM/configs/sim/nuscenes_camera.yaml')
    vehicle = spec.vehicle
    replan_period = float(_hugsim_expand(json.loads((ADAPTER / 'configs/planning.json').read_text())).get(
        'global_replan_interval_s', vehicle.dt))
    replan_ticks = int(round(replan_period / vehicle.dt))
    if replan_ticks < 1 or not np.isclose(replan_ticks * vehicle.dt, replan_period):
        raise ValueError('Global replanning period must match control clock')
    source_end_time = (spec.frames - 1) / spec.fps
    initial_native = T @ relative[start] @ T.T
    speed, steer, _ = recorded_kinematics(relative, start, spec.fps, vehicle.wheelbase_m)
    checkpoint = BASE / 'code/UniAD_SIM/ckpts/uniad_base_e2e.pth'  # frozen model, as in the arm
    identity = dict(manifest=manifest, source=spec.source, insert=insert, start=start,
        steps=steps, time_policy=arm_request.get('time_policy'),
        source_checkpoint_stat=[Path(spec.source['checkpoint']).stat().st_size,
                                Path(spec.source['checkpoint']).stat().st_mtime_ns],
        asset_stat=[(Path(insert['asset']) / 'gs.pth').stat().st_size,
                    (Path(insert['asset']) / 'gs.pth').stat().st_mtime_ns])
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return dict(manifest=manifest, spec=spec, start=start, steps=steps, insert=insert,
                reference=reference, relative=relative, mount=mount, vehicle=vehicle,
                replan_ticks=replan_ticks, source_end_time=source_end_time,
                initial_native=initial_native, speed=speed, steer=steer,
                checkpoint=checkpoint, signature=signature)


def _base_report(arm_request, spec, setup, mode, label):
    vehicle = setup['vehicle']
    return dict(schema='branch_run_report_v1', diagnostic_mode=True,
        formal_evaluation_admitted=False, n1_certified=False,
        experiment='phase27_branch_state_mechanism', mode=mode, label=label,
        training=False, n1_validation_suspended=True,
        closed_loop_run=True, control_actions_executed=False,
        closed_loop='frozen UniAD current planning BEV, history perception BEV; evidence-tiered rectangular MPC; native Motion.advance',
        guard_policy=POLICY,
        ground_truth_used_by_optimizer=False, recorded_future_ego_used_for_closed_motion=False,
        route_commands_from_recorded_route=True,
        arm=arm_request.get('arm', 'unedited'),
        stationary_threshold_mps=0.001, trace=[], observations=[],
        experiment_signature=setup['signature'],
        scene=spec.source['scene'], start_frame=setup['start'],
        steps_requested=arm_request['steps'], dt=setup['vehicle'].dt,
        time_policy='source_time_only',
        source_end_timestamp_s=setup['source_end_time'], three_view=True,
        camera_order=list(THREE_CAMERAS),
        requested_duration_seconds=arm_request['steps'] * vehicle.dt,
        avoidance_success=False,
        termination_policy='mechanism experiment: no success criterion; window end or safety stop',
        timing=dict(steps=[]))


def _spawn_worker(worker_folder, output, env_extra=None, restore=None):
    (worker_folder / 'step_000').mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, BASE=str(BASE), PYTHONUNBUFFERED='1',
        BEV_EXPERIMENT_SEED=str(SEED), PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python',
        UNIAD_INIT_TRACE='1', UNIAD_STARTUP_REPORT=str(worker_folder / 'startup.json'))
    for key in ('LD_LIBRARY_PATH', 'PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    if env_extra:
        env.update(env_extra)
    if restore is not None:
        atomic_json(worker_folder / 'restore_state.json', dict(snapshot=str(restore)))
    log = (output / 'uniad_worker.log').open('w')
    began = time.monotonic()
    process = subprocess.Popen([str(BASE / 'tools/pixi-uniad-legacy/.pixi/envs/default/bin/python'),
        str(WORKER), '--job', str(worker_folder),
        '--planning-bev', 'current', '--model-profile', 'calibration_geometry_v1',
        '--diagnostic-closed-loop'],
        env=env, cwd=BASE, stdout=log, stderr=subprocess.STDOUT)
    ready = wait_json(worker_folder / 'ready.json', process, timeout=1200.)
    if ready['planning_bev'] != 'current':
        raise ValueError('Wrong planning profile')
    return process, log, ready, time.monotonic() - began


def _stop_worker(worker_folder, process, log):
    atomic_json(worker_folder / 'stop.json', {})
    if process is not None:
        try:
            process.wait(timeout=20.)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10.)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    if log is not None:
        log.close()


def _packet(model_packet_fn, packet_path):
    atomic_pickle(packet_path, model_packet_fn)


def _compare_perception(reference_npz, response_perception):
    diffs = {}
    with np.load(reference_npz) as old, np.load(response_perception) as new:
        for key in ['candidate_boxes', 'candidate_scores', 'trajectory', 'occupancy',
                    'geometry_candidate_boxes']:
            if key not in old.files or key not in new.files:
                raise ValueError('Missing comparison field ' + key)
            if old[key].shape != new[key].shape:
                raise ValueError('Perception shape mismatch: ' + key)
            diffs[key] = float(np.max(np.abs(old[key].astype(float) - new[key].astype(float)))) \
                if old[key].size else 0.
    return diffs


def _control_loop(ctx, output, session, report, worker_folder, process,
                  start_index, edited_now):
    """The closed control loop, shared by record and branch modes.

    Mirrors chain_repair/phase11/paired_closed_loop_arm.py step semantics; adds
    per-step wall timing, branch reproduction gates, and the snapshot hook.
    """
    from PIL import Image
    from depth_sensor_map import DepthObstacleMemory
    from road_prediction_memory import RoadPredictionMemory
    from navigation_anchor import NavigationAnchor
    from committed_route import CommittedRoute
    from native_geometry_evaluation import NativeVehicleEvaluation
    from native_actor_evaluation import NativeActorEvaluation
    from live_env import Motion
    from drivable_area import from_saved, DrivableArea
    import drivable_area as _da
    assert 'phase11' in str(_da.__file__), 'drivable_area bound to ' + str(_da.__file__)

    spec, setup = ctx['spec'], ctx['setup']
    start, steps = setup['start'], setup['steps']
    vehicle, relative = setup['vehicle'], setup['relative']
    initial_native = setup['initial_native']
    source_end_time = setup['source_end_time']
    reference = setup['reference']
    replan_ticks = setup['replan_ticks']

    native_evaluation = NativeVehicleEvaluation(session.backend.env.models['RigidNodes'])
    actor_evaluation = NativeActorEvaluation({name: session.backend.env.models[name]
        for name in ('RigidNodes', 'DeformableNodes', 'SMPLNodes') if name in session.backend.env.models})
    report['actor_reference_groups'] = list(actor_evaluation.groups)

    st = ctx['state']  # restored or fresh driver state
    motion, warm = st['motion'], st['warm']
    controller, obstacle_memory = st['controller'], st['obstacle_memory']
    depth_memory, road_memory = st['depth_memory'], st['road_memory']
    navigation_anchor, committed_route = st['navigation_anchor'], st['committed_route']
    mode, min_gap, token = st['mode'], st['minimum_gap'], st['token']

    snapshot_after = ctx.get('snapshot_after_index')
    snapshot_dir = ctx.get('snapshot_dir')
    reproduce = ctx.get('reproduce')  # record output dir to check against, or None
    reproduction_rows = []

    for index in range(start_index, steps + 1):
        step_began = time.monotonic()
        frame = start + index
        timestamp = frame / spec.fps
        source_frame = frame
        atomic_json(output / 'progress.json', dict(stage='closed_loop_feedback',
            step=index, total_steps=steps, source_frame=source_frame, simulation_frame=frame))
        native = initial_native @ native_delta(motion.x, motion.z, motion.yaw)
        closed_pose = T.T @ native @ T
        if index == 0 and ctx.get('first_render') is not None:
            closed_packet = ctx['first_render']
        else:
            closed_packet = session.render(closed_pose, source_frame / spec.fps)
        closed_packet = dict(closed_packet, timestamp_s=timestamp,
            background_timestamp_s=source_frame / spec.fps,
            background_frozen=timestamp > source_end_time + 1e-8)

        obs, info = ctx['model_packet'](closed_packet)
        for name in THREE_CAMERAS:
            Image.fromarray(closed_packet['cameras'][name]['rgb']).save(
                output / f'closed_loop/{name}/{index:03d}.png')
        compose_three_closed(closed_packet, index * vehicle.dt, motion.speed, mode,
            ctx['label'].upper() + ' | live source time').save(
            output / f'comparison_frames/{index:03d}.png')

        pixel_gate = None
        if reproduce is not None:
            recorded = [np.asarray(Image.open(reproduce['dir'] / f'closed_loop/{name}/{index:03d}.png'))
                        for name in THREE_CAMERAS]
            live = [closed_packet['cameras'][name]['rgb'] for name in THREE_CAMERAS]
            pixel_gate = {name: bool(np.array_equal(a, b))
                          for name, a, b in zip(THREE_CAMERAS, recorded, live)}

        ospeed, _, _ = recorded_kinematics(relative, source_frame, spec.fps, vehicle.wheelbase_m)
        native_check = native_evaluation.check(closed_pose, source_frame, vehicle)
        actor_check = actor_evaluation.check(closed_pose, source_frame, vehicle)
        if actor_check['overlap']:
            report['terminal_actor_reference'] = dict(actor_check,
                source_frame=float(source_frame), ego_to_world=np.asarray(closed_pose).tolist())
            raise RuntimeError('Post-hoc actor reference overlap; diagnostic stopped')
        if native_check['overlap']:
            raise RuntimeError('Post-hoc native-vehicle reference overlap; diagnostic stopped')
        gap = reference_check(native, reference, vehicle) if edited_now() else dict(overlap=False, gap_m=None)
        if edited_now():
            min_gap = min(min_gap, gap['gap_m'])
        if gap['overlap']:
            raise RuntimeError('Post-hoc inserted-vehicle footprint overlap; diagnostic stopped')
        report['observations'].append(dict(index=index, timestamp=timestamp,
            source_frame=source_frame, simulation_frame=frame,
            background_timestamp_s=source_frame / spec.fps,
            background_frozen=closed_packet['background_frozen'],
            open_ego_to_world=relative[source_frame].tolist(),
            closed_ego_to_world=np.asarray(closed_pose).tolist(),
            open_speed_mps=ospeed, closed_speed_mps=motion.speed,
            control_mode_from_previous_step=mode,
            inserted_reference_gap_m=gap['gap_m'],
            native_vehicle_reference=native_check, all_actor_reference=actor_check,
            navigation_rejoin=navigation_anchor.errors(info)))
        atomic_json(output / 'run_report.json', report)
        # v3 pre-registered terminations (request-fixed; never tuned per outcome)
        term = ctx.get('termination')
        if term is not None:
            if term['pass_check'] and edited_now() and not ctx.get('_pass_done'):
                from closed172_common import body_vertices, asset_reference_vertices
                from waymo_adapter import SR_TO_HUGSIM as _T
                poses0 = _T @ np.asarray(report['observations'][0]['closed_ego_to_world']) @ _T.T
                forward = poses0[[0, 2], 2]; forward = forward / np.linalg.norm(forward)
                far = float(np.max(asset_reference_vertices(reference)[:, [0, 2]] @ forward))
                rear = float(np.min(body_vertices(_T @ np.asarray(closed_pose) @ _T.T, vehicle)[:, [0, 2]] @ forward))
                if rear > far:
                    ctx.setdefault('_pass_since', timestamp)
                else:
                    ctx.pop('_pass_since', None)
                if ctx.get('_pass_since') is not None and timestamp - ctx['_pass_since'] >= term['pass_hold_s']:
                    report.update(termination_reason='passed_asset', interaction_outcome='pass',
                                  pass_timestamp=ctx['_pass_since'])
                    ctx['_pass_done'] = True
                    atomic_json(output / 'run_report.json', report)
                    break
            if motion.speed <= term['stall_speed_mps']:
                ctx['_stall_count'] = ctx.get('_stall_count', 0) + 1
            else:
                ctx['_stall_count'] = 0
            if term['stall_check'] and ctx['_stall_count'] >= term['stall_steps']:
                report.update(termination_reason='stalled_10x', interaction_outcome='stall')
                atomic_json(output / 'run_report.json', report)
                break
            if time.monotonic() - ctx['_loop_wall0'] >= term['max_wall_seconds']:
                report.update(termination_reason='wall_cap', interaction_outcome='time_cap')
                atomic_json(output / 'run_report.json', report)
                break
        if index == steps:
            break
        folder = worker_folder / f'step_{index:03d}'
        folder.mkdir(exist_ok=True)
        packet_path = folder / 'observation.pkl'
        atomic_pickle(packet_path, (obs, info))  # kept for branch verification gates
        req = dict(index=index, packet=str(packet_path))
        if index == 0:
            req['diagnostic_seed'] = SEED
        atomic_json(worker_folder / f'request_{index:03d}.json', req)
        response = wait_json(worker_folder / f'done_{index:03d}.json', process, timeout=300.)
        if not np.isclose(response['timestamp'], timestamp):
            raise ValueError('Inference clock mismatch')
        with np.load(response['perception']) as result:
            road_contract = from_saved(result)
            from occupancy_protocol import unpack
            plan = result['trajectory'].copy()
            occupancy, occupancy_contract = unpack(result)
            info['occupancy_contract'] = occupancy_contract
            candidates = [result['candidate_' + key].copy() for key in ('boxes', 'scores', 'labels')]
            if 'geometry_candidate_boxes' in result:
                candidates[0] = result['geometry_candidate_boxes'].copy()
        plan_raw = plan.copy()  # phase28: UniAD raw predicted trajectory
        if ctx.get('plan_patch') is not None:
            plan = ctx['plan_patch'](plan_raw, info, occupancy=occupancy.copy() if occupancy is not None else None,
                                     step_index=index, speed=motion.speed)
            plan = np.asarray(plan, dtype=float).reshape(6, 2)
            if not np.isfinite(plan).all():
                raise ValueError('Plan patch returned non-finite values')
        road_contract = road_memory.update(road_contract, info)
        reliability = _hugsim_expand(json.loads(Path(response['reliability']).read_text()))
        reliability = obstacle_memory.update(reliability, *candidates, info,
            current_occupancy=np.asarray(occupancy).reshape(5, 200, 200)[0] if occupancy is not None else None)
        depth_points, depth_audit = depth_memory.update(obs, info)
        reliability['unknown_visibility_obstacles_xy'] = depth_audit.pop('unknown_visibility_points_xy', [])
        reliability['unknown_visibility_geometry'] = depth_audit.get('unknown_visibility_geometry', {})
        reliability['observed_depth_obstacles_xy'] = depth_points.tolist()
        reliability['observed_depth_geometry'] = depth_audit
        if ctx.get('reliability_patch') is not None:  # phase28: guard-input patch hook
            reliability = ctx['reliability_patch'](reliability, info)
        atomic_json(folder / 'depth_geometry.json', depth_audit)
        reliability['require_predicted_drivable_area'] = True
        reliability['predicted_road_constraint_mode'] = 'soft'
        # This anchor remains an evaluation metric only; it is never passed
        # to the MPC or committed as a navigation target.
        navigation_anchor.update(plan, info)
        from current_reference import prepare_report
        reliability=prepare_report(reliability)
        reliability['predicted_drivable_area'] = road_contract
        atomic_json(folder / 'predicted_drivable_audit.json',
            dict(DrivableArea(road_contract, vehicle).audit(), constraint_mode='soft',
                 physical_road_boundary_verified=False))
        recovery_cap = controller.speed_cap(reliability)
        if recovery_cap is not None:
            previous_cap = reliability.get('speed_limit_mps')
            reliability = dict(reliability, speed_limit_mps=min(recovery_cap, previous_cap)
                if previous_cap is not None else recovery_cap)
        atomic_json(folder / 'guard_memory_report.json', reliability)
        try:
            optimized = optimize(plan, occupancy, info, warm=warm, reliability=reliability,
                vehicle=vehicle, _reuse_verified_warm=False)
        except ValueError as exc:
            if not str(exc).startswith('Empty occupancy:'):
                raise
            optimized = dict(selected=None, reason=str(exc), ground_truth_used_by_optimizer=False)
        atomic_json(folder / 'optimization.json', optimized)
        action, decision = controller.update(optimized, motion.speed, timestamp,
            vehicle.dt, vehicle.emergency_braking_mps2)
        atomic_json(folder / 'control_decision.json', decision)
        selected = optimized['selected']
        following = decision['state_after'] in ('normal', 'caution', 'recovering')
        warm = np.asarray(selected['controls']) if selected is not None and following else None
        mode = (decision['state_after'] + ' / MPC') if following else (decision['state_after'] + ' / reassessing')
        validate_action(dict(acc_mps2=action['acc'], steer_rate_radps=action['steer_rate']),
            vehicle, motion.speed, motion.steer)
        before = np.array([motion.x, motion.z, motion.yaw, motion.speed, motion.steer])
        report['controls_attempted'] = index + 1
        offset = np.asarray(info['model_lidar_frame']['virtual_to_ego'], dtype=float)[:2, 3]
        local_initial = np.r_[-offset, 0., motion.speed, motion.steer]
        local_after = integrate_state(local_initial, [action['acc'], action['steer_rate']],
            vehicle.dt, vehicle.wheelbase_m)
        road_clearance_after = float(DrivableArea(road_contract, vehicle).clearance(local_after[None])[0])
        road_swept_after = float(DrivableArea(road_contract, vehicle).swept_clearance(
            np.stack([local_initial, local_after])).min())
        motion.advance(action)
        after = np.array([motion.x, motion.z, motion.yaw, motion.speed, motion.steer])
        expected = integrate_state(before, [action['acc'], action['steer_rate']], vehicle.dt, vehicle.wheelbase_m)
        np.testing.assert_allclose(after, expected, rtol=0, atol=1e-10)
        for u in np.linspace(0, 1, 11):
            p = before * (1 - u) + after * u
            check = reference_check(initial_native @ native_delta(*p[:3]), reference, vehicle) \
                if edited_now() else dict(overlap=False, gap_m=None)
            if edited_now():
                min_gap = min(min_gap, check['gap_m'])
            if check['overlap']:
                raise RuntimeError('Interpolated reference overlap after control')
            checked_native = initial_native @ native_delta(*p[:3])
            checked_source = T.T @ checked_native @ T
            native_result = native_evaluation.check(checked_source, source_frame, vehicle)
            actor_result = actor_evaluation.check(checked_source,
                min(source_frame + float(u), spec.frames - 1), vehicle)
            if actor_result['overlap']:
                raise RuntimeError('Interpolated actor reference overlap after control')
            if native_result['overlap']:
                raise RuntimeError('Interpolated native-vehicle reference overlap after control')
        temporal = _hugsim_expand(json.loads((folder / 'temporal_state.json').read_text()))
        if temporal['history_bev_present'] != (index > 0 or start_index > 0) \
                or not temporal['metadata_cache_is_tracking_cache']:
            raise RuntimeError('Unexpected temporal BEV state')
        navref = (optimized.get('guard') or {}).get('navigation_reference') or {}
        report['trace'].append(dict(step=index, timestamp=timestamp, action=action, mode=mode,
            optimization_reason=optimized.get('reason'), control_decision=decision,
            state_before=before.tolist(), state_after=after.tolist(),
            neural_plan=plan.tolist(), neural_plan_raw=plan_raw.tolist(),
            plan_patch_active=ctx.get('plan_patch') is not None,
            adopted_reference_source=navref.get('source'),
            adopted_reference_xy=(navref.get('tracking_reference')
                                  or navref.get('path_center_xy')),
            perception=response['perception'],
            reliability=response['reliability'], temporal=temporal,
            physics_integration_verified=True, predicted_road_constraint_mode='soft',
            predicted_road_footprint_clearance_after_m=road_clearance_after,
            predicted_road_swept_clearance_m=road_swept_after))
        report['control_actions_executed'] = True
        atomic_json(output / 'run_report.json', report)
        if obs.get('depth_alpha'):
            np.savez_compressed(folder / 'observed_depth_alpha.npz', **obs['depth_alpha'])
        if obs.get('depth_variance'):
            np.savez_compressed(folder / 'observed_depth_variance.npz', **obs['depth_variance'])
        for field in ['surface_depth', 'surface_valid']:
            if obs.get(field):
                np.savez_compressed(folder / (field + '.npz'), **obs[field])
        atomic_pickle(folder / 'sensor_metadata.pkl', info)
        atomic_json(folder / 'sensor_metadata.json',
            _hugsim_expand(json.loads(json.dumps(info, default=lambda value: str(value) if isinstance(value, Path)
                else np.asarray(value).tolist()))))
        np.savez_compressed(folder / 'observed_depth.npz', **obs['depth'])

        step_wall = time.monotonic() - step_began
        report['timing']['steps'].append(dict(index=index, wall_seconds=step_wall,
            inference_seconds=response.get('inference_seconds'),
            worker_max_memory_bytes=response.get('max_memory_allocated_bytes')))

        if reproduce is not None:
            row = reproduce['trace'][index]
            diffs = _compare_perception(
                reproduce['dir'] / f'uniad_worker/step_{index:03d}/perception.npz',
                response['perception'])
            pose_diff = float(np.max(np.abs(np.asarray(row['state_after']) - after)))
            action_diff = max(abs(float(row['action'][k]) - float(action[k])) for k in ('acc', 'steer_rate'))
            reproduction_rows.append(dict(index=index, perception=diffs,
                pixel_identical=pixel_gate, pose_max_diff=pose_diff,
                action_max_diff=action_diff,
                passed=pixel_gate is not None and all(pixel_gate.values())
                    and all(v <= PERCEPTION_TOLERANCE for v in diffs.values())
                    and pose_diff <= POSE_TOLERANCE and action_diff == 0.))
            atomic_json(output / 'reproduction.json',
                dict(tolerance=PERCEPTION_TOLERANCE, rows=reproduction_rows))

        if snapshot_after is not None and index == snapshot_after:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            began = time.monotonic()
            size = branch_state.save_driver_snapshot(snapshot_dir / 'driver_snapshot.pkl',
                next_index=index + 1, token=token, motion=dict(vars(motion)), warm=warm,
                controller=controller, obstacle_memory=obstacle_memory,
                depth_memory=depth_memory, road_memory=road_memory,
                navigation_anchor=navigation_anchor, committed_route=committed_route,
                mode=mode, report=report, minimum_gap=min_gap,
                rng=branch_state.capture_rng(),
                session=dict(manifest=str(output / 'scene_manifest.json'),
                             generation=session.generation, revision=session.revision,
                             objects=session.objects),
                edits=ctx.get('applied_edits', []),
                request=ctx['arm_request'],
                experiment_signature=setup['signature'],
                implementation=ctx['implementation']())
            atomic_json(worker_folder / 'command_snapshot.json',
                dict(path=str(snapshot_dir)))
            done = wait_json(worker_folder / 'snapshot_done.json', process, timeout=300.)
            report['branch_snapshot'] = dict(
                after_index=index, next_index=index + 1, directory=str(snapshot_dir),
                driver_bytes=size, worker=dict(done),
                wall_seconds=time.monotonic() - began)
            atomic_json(output / 'run_report.json', report)
        print(f'Completed feedback {index + 1}/{steps}: {mode}; speed={motion.speed:.3f}',
              flush=True)

    st.update(motion=motion, warm=warm, controller=controller,
              obstacle_memory=obstacle_memory, depth_memory=depth_memory,
              road_memory=road_memory, navigation_anchor=navigation_anchor,
              committed_route=committed_route, mode=mode, minimum_gap=min_gap)
    return report


def _fresh_driver_state(setup, spec):
    from depth_sensor_map import DepthObstacleMemory
    from road_prediction_memory import RoadPredictionMemory
    from navigation_anchor import NavigationAnchor
    from committed_route import CommittedRoute
    from live_env import Motion
    motion = Motion(dt=spec.vehicle.dt, wheelbase=spec.vehicle.wheelbase_m,
                    start_speed=setup['speed'])
    motion.steer = setup['steer']
    motion.timestamp = setup['start'] / spec.fps
    return dict(motion=motion, warm=None, controller=RecoveryController(),
        obstacle_memory=ObstacleMemory.from_source(spec.source),
        depth_memory=DepthObstacleMemory(spec.vehicle),
        road_memory=RoadPredictionMemory(spec.vehicle),
        navigation_anchor=NavigationAnchor(spec.vehicle.half_length_m),
        committed_route=CommittedRoute(), mode='initial state',
        minimum_gap=float('inf'), token=str(uuid4()))


def _restored_driver_state(payload, setup, spec):
    from live_env import Motion
    motion = Motion(dt=spec.vehicle.dt, wheelbase=spec.vehicle.wheelbase_m,
                    start_speed=setup['speed'])
    motion.__dict__.update(payload['motion'])
    branch_state.restore_rng(payload['rng'])
    return dict(motion=motion, warm=payload['warm'], controller=payload['controller'],
        obstacle_memory=payload['obstacle_memory'], depth_memory=payload['depth_memory'],
        road_memory=payload['road_memory'], navigation_anchor=payload['navigation_anchor'],
        committed_route=payload['committed_route'], mode=payload['mode'],
        minimum_gap=payload['minimum_gap'], token=payload['token'])


def _model_packet_factory(setup, spec, token_ref):
    relative, mount, vehicle = setup['relative'], setup['mount'], setup['vehicle']
    def model_packet(packet):
        motion = token_ref['motion']
        nearest = int(np.argmin(np.linalg.norm(
            relative[:, :2, 3] - np.asarray(packet['ego_to_world'])[:2, 3], axis=1)))
        _, _, command = recorded_kinematics(relative, nearest, spec.fps, vehicle.wheelbase_m)
        obs, info = to_uniad(packet, mount, speed_mps=motion.speed,
            steering_rad=motion.steer, command=command, scene_token=token_ref['token'],
            dt=vehicle.dt, acceleration_mps2=motion.last_accel,
            steering_rate_radps=motion.last_rate)
        if packet.get('background_frozen'):
            info['declared_static_epoch'] = token_ref['token'] + ':' + str(packet['edit_revision']) \
                + ':' + str(packet['background_timestamp_s'])
        info.update(closed_loop_run=True, control_actions_executed=True,
            ego_kinematics_source='native_motion_control_feedback')
        return obs, info
    return model_packet


_PATCH_MODULE_PATHS = []


def _load_patch(request):
    spec_ = request.get('plan_patch')
    if spec_ is None:
        return None
    return _load_pinned(spec_, 'plan')


def _load_pinned(spec_, kind):
    import hashlib as _h
    import importlib.util as _il
    path = Path(spec_['module']).resolve()
    digest = _h.sha256(path.read_bytes()).hexdigest()
    if digest != spec_['sha256']:
        raise ValueError(kind + ' patch module hash mismatch')
    _PATCH_MODULE_PATHS.append(path)
    spec = _il.spec_from_file_location(kind + '_patch_module', path)
    module = _il.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, 'apply', None)):
        raise ValueError(kind + ' patch module must expose apply(...)')
    return module.apply


def _load_reliability_patch(request):
    spec_ = request.get('reliability_patch')
    if spec_ is None:
        return None
    return _load_pinned(spec_, 'reliability')


def _patch_exclusions():
    """Declared patch modules are excluded from the fingerprint comparison
    (their hashes are pinned by the request and recorded in the report)."""
    exclude = set()
    for module_path in _PATCH_MODULE_PATHS:
        exclude.add(str(module_path.relative_to(BASE))
                    if module_path.is_relative_to(BASE) else str(module_path))
    return exclude


def _load_driver_snapshot_declared_patch(path, *, implementation, experiment_signature,
                                         declared_driver_diff=None):
    """phase28: like branch_state.load_driver_snapshot, but declared patch modules
    and an explicitly declared driver-file difference are excluded from the
    fingerprint comparison. Everything else must match exactly."""
    import pickle as _pickle
    with Path(path).open('rb') as handle:
        payload = _pickle.load(handle)
    if payload.get('schema') != branch_state.DRIVER_SCHEMA:
        raise ValueError('Unexpected driver snapshot schema')
    exclude = _patch_exclusions()
    if declared_driver_diff:
        exclude.add(declared_driver_diff['file'])
    stored = {k: v for k, v in payload.get('implementation', {}).items() if k not in exclude}
    expected = {k: v for k, v in implementation.items() if k not in exclude}
    if stored != expected:
        raise ValueError('Branch implementation fingerprint differs beyond the declared patch')
    if payload.get('experiment_signature') != experiment_signature:
        raise ValueError('Branch experiment signature differs; refusing to mix inputs')
    return payload


def run_record(request):
    output = Path(request['output']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'comparison_frames').mkdir()
    for name in THREE_CAMERAS:
        (output / 'closed_loop' / name).mkdir(parents=True)
    worker_folder = output / 'uniad_worker'
    arm_request = _load_arm_request(request['arm_request'])
    atomic_json(output / 'arm_request.json', arm_request)
    record_patch = _load_patch(request)  # before fingerprint: patch is audited code
    setup = _setup(arm_request, output)
    spec = setup['spec']
    implementation = _implementation(setup['manifest'], spec)
    report = _base_report(arm_request, spec, setup, 'record', request.get('label', 'record'))
    report.update(experiment='heading_fix_scene_v1', controller_version='heading_fix_v1',
                  termination_config=_termination_cfg(request),
                  termination_policy='fresh 80-step observation: pass hold 0.5s / proxy overlap / 9000s wall cap; low speed recorded without early stop')
    report['implementation_before'] = implementation()
    report['checkpoint_before'] = fingerprint(setup['checkpoint'])
    snapshot_after = request['snapshot_after_index']
    if isinstance(snapshot_after, bool) or not isinstance(snapshot_after, int) \
            or not 0 <= snapshot_after < arm_request['steps']:
        raise ValueError('snapshot_after_index must be an integer within the run window')
    snapshot_dir = Path(request.get('snapshot_dir') or output / 'branch_snapshot').resolve()
    began = time.monotonic()
    process = log = None
    try:
        with SceneSession() as session:
            scene_began = time.monotonic()
            session.load_scene(output / 'scene_manifest.json')
            edited = arm_request.get('arm') == 'edited'
            applied = []
            if edited:
                edit = _hugsim_expand(json.loads(Path(arm_request['insert_request']).read_text()))
                session.edit(edit)
                applied.append(edit)
            report['timing']['scene_load_seconds'] = time.monotonic() - scene_began
            state = _fresh_driver_state(setup, spec)
            token_ref = dict(state)  # mutable view for model_packet closure
            ctx = dict(spec=spec, setup=setup, arm_request=arm_request,
                       implementation=implementation, label=request.get('label', 'record'),
                       model_packet=_model_packet_factory(setup, spec, token_ref),
                       snapshot_after_index=snapshot_after, snapshot_dir=snapshot_dir,
                       applied_edits=applied, state=state,
                       plan_patch=record_patch,
                       reliability_patch=_load_reliability_patch(request),
                       termination=_termination_cfg(request), _loop_wall0=time.monotonic())
            if edited and reference_check(setup['initial_native'], setup['reference'],
                    setup['vehicle'])['overlap']:
                raise RuntimeError('Initial ego overlaps inserted reference')
            # five_uniad.infer() requires the step_000 packet to exist at startup.
            (worker_folder / 'step_000').mkdir(parents=True, exist_ok=True)
            first = session.render(setup['relative'][setup['start']], setup['start'] / spec.fps)
            atomic_pickle(worker_folder / 'step_000' / 'observation.pkl',
                          ctx['model_packet'](first))
            ctx['first_render'] = first
            process, log, ready, boot = _spawn_worker(worker_folder, output)
            report['worker_ready'] = ready
            report['timing']['worker_boot_seconds'] = boot
            report = _control_loop(ctx, output, session, report, worker_folder, process,
                                   0, lambda: edited)
        report['status'] = 'completed'
        report.setdefault('termination_reason', 'window_completed')
        report.setdefault('interaction_outcome', 'window_exhausted')
    except BaseException as exc:
        report.update(status='failed', error=repr(exc),
                      termination_reason='exception:' + type(exc).__name__)
        raise
    finally:
        if worker_folder.exists():
            _stop_worker(worker_folder, process, log)
        report['implementation_after'] = implementation()
        report['implementation_unchanged'] = report['implementation_after'] == report['implementation_before']
        report['checkpoint_after'] = fingerprint(setup['checkpoint'])
        report['checkpoint_unchanged'] = report['checkpoint_before'] == report['checkpoint_after']
        report['wall_seconds'] = time.monotonic() - began
        _score_run(report, setup)
        atomic_json(output / 'run_report.json', report)
    return report


def _score_run(report, setup):
    """phase28: post-hoc diagnostic score on the EXECUTED trajectory."""
    try:
        if not report.get('control_actions_executed'):
            return
        if report.get('arm') != 'edited' and not (report.get('branch') or {}).get('intervention'):
            return
        from trajectory_baseline_score import score_report
        report['trajectory_score'] = score_report(report, setup['reference'], setup['vehicle'])
    except Exception as exc:
        report['trajectory_score_error'] = repr(exc)


def _hash_tree(directory):
    directory = Path(directory)
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.iterdir()) if p.is_file()}


def run_branch(request, replay):
    output = Path(request['output']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'comparison_frames').mkdir()
    for name in THREE_CAMERAS:
        (output / 'closed_loop' / name).mkdir(parents=True)
    worker_folder = output / 'uniad_worker'
    snapshot_dir = Path(request['snapshot']).resolve()
    hashes_before = _hash_tree(snapshot_dir)
    record_dir = Path(request['record_output']).resolve()
    record_report = _hugsim_expand(json.loads((record_dir / 'run_report.json').read_text()))
    if record_report.get('status') != 'completed':
        raise ValueError('Record run did not complete; refusing to branch from it')
    arm_request = _load_arm_request(record_dir / 'arm_request.json')
    ctx_patch = _load_patch(request)  # before fingerprint: patch is audited code
    setup = _setup(arm_request, output)
    spec = setup['spec']
    implementation = _implementation(setup['manifest'], spec)
    declared_diff = request.get('declare_driver_difference')
    payload = _load_driver_snapshot_declared_patch(snapshot_dir / 'driver_snapshot.pkl',
        implementation=implementation(), experiment_signature=setup['signature'],
        declared_driver_diff=declared_diff)
    if declared_diff:
        print('declared driver difference:', declared_diff)
    next_index = int(payload['next_index'])
    steps = arm_request['steps']
    if not 0 < next_index <= steps:
        raise ValueError('Snapshot next index outside the run window')
    label = request.get('label', 'branch')
    branch_meta = dict(snapshot=str(snapshot_dir), record_output=str(record_dir),
        next_index=next_index, snapshot_hashes_before=hashes_before,
        intervention=request.get('intervention'),
        declared_driver_difference=declared_diff)
    report = _base_report(arm_request, spec, setup,
        'replay-branch' if replay else 'branch', label)
    # fingerprints on the base report too: the finally block must survive
    # failures that happen before the report continuity swap.
    report['implementation_before'] = implementation()
    report['checkpoint_before'] = fingerprint(setup['checkpoint'])
    began = time.monotonic()
    process = log = None
    try:
        with SceneSession() as session:
            scene_began = time.monotonic()
            session.load_scene(output / 'scene_manifest.json')
            applied = []
            for edit in payload.get('edits', []):
                session.edit(edit)
                applied.append(edit)
            report['timing']['scene_load_seconds'] = time.monotonic() - scene_began
            state = _restored_driver_state(payload, setup, spec)
            # Continue the record's report so trace/observations stay continuous.
            report = payload['report']
            report.update(mode='replay-branch' if replay else 'branch', label=label,
                status='running', timing=dict(steps=[]), branch=branch_meta)
            report.pop('error', None)
            report.pop('termination_reason', None)
            report['implementation_before'] = implementation()
            report['checkpoint_before'] = fingerprint(setup['checkpoint'])
            token_ref = dict(state)
            ctx = dict(spec=spec, setup=setup, arm_request=arm_request,
                       implementation=implementation, label=label,
                       model_packet=_model_packet_factory(setup, spec, token_ref),
                       state=state, applied_edits=applied, plan_patch=ctx_patch,
                       reliability_patch=_load_reliability_patch(request),
                       termination=_termination_cfg(request), _loop_wall0=time.monotonic(),
                       reproduce=(None if request.get('intervention') or ctx_patch is not None
                                  else dict(dir=record_dir, trace=record_report['trace'])))
            # five_uniad.infer() requires step_000/observation.pkl to exist at
            # startup; branch runs resume at next_index, so link the record's
            # step-0 packet purely to satisfy the startup contract.
            from closed_loop_resume import link_file
            link_file(record_dir / 'uniad_worker/step_000/observation.pkl',
                      worker_folder / 'step_000/observation.pkl')
            if replay:
                process, log, ready, boot = _spawn_worker(worker_folder, output)
                report['worker_ready'] = ready
                report['timing']['worker_boot_seconds'] = boot
                replay_began = time.monotonic()
                from closed_loop_resume import replay_model_history
                replay_payload = dict(source_result=str(record_dir), next_index=next_index)
                replay_model_history(replay_payload, worker_folder, process,
                    wait_json, atomic_json, atomic_pickle)
                report['timing']['replay_seconds'] = time.monotonic() - replay_began
                report['replay_verification'] = _hugsim_expand(json.loads(
                    (worker_folder / 'resume_verification.json').read_text()))
            else:
                process, log, ready, boot = _spawn_worker(
                    worker_folder, output, restore=snapshot_dir)
                report['worker_ready'] = ready
                report['timing']['worker_boot_seconds'] = boot
                if not (ready.get('branch_restored') or {}).get('next_index') == next_index:
                    raise RuntimeError('Worker boot restore did not resume at the snapshot index')
                # Fidelity gate G4: recorded frame T+1 through restored history.
                gate_began = time.monotonic()
                gate = _verification_gate(worker_folder, process, record_dir, next_index)
                report['restore_fidelity_gate'] = gate
                report['timing']['verify_seconds'] = time.monotonic() - gate_began
                if not gate['passed']:
                    report.update(status='failed',
                        termination_reason='restore_fidelity_gate_failed')
                    atomic_json(output / 'run_report.json', report)
                    raise RuntimeError('Restore fidelity gate failed: ' + json.dumps(gate))
                # Reset worker back to the snapshot so live steps re-run cleanly
                # (the gate consumed one inference; restore also rewinds RNG).
                # Retire the gate request first: after restore the worker resumes
                # at next_index and must not re-serve the recorded packet before
                # the live one is written.
                (worker_folder / f"done_{next_index:03d}.json").unlink()
                (worker_folder / f"request_{next_index:03d}.json").unlink()
                atomic_json(worker_folder / 'command_restore.json', dict(path=str(snapshot_dir)))
                restored = wait_json(worker_folder / 'restore_done.json', process, timeout=300.)
                report['timing']['command_restore_seconds'] = restored['wall_seconds']
            intervention = request.get('intervention')
            intervened = arm_request.get('arm') == 'edited'
            if intervention:
                edit = _hugsim_expand(json.loads(Path(intervention).read_text()))
                session.edit(edit)
                applied.append(edit)
                intervened = True
                report['branch']['intervention_edit_revision'] = session.revision
            report = _control_loop(ctx, output, session, report, worker_folder, process,
                                   next_index, lambda: intervened)
        report['status'] = 'completed'
        report.setdefault('termination_reason', 'window_completed')
        report.setdefault('interaction_outcome', 'window_exhausted')
    except BaseException as exc:
        report.update(status='failed', error=repr(exc),
                      termination_reason='exception:' + type(exc).__name__)
        raise
    finally:
        if worker_folder.exists():
            _stop_worker(worker_folder, process, log)
        report['implementation_after'] = implementation()
        report['implementation_unchanged'] = report['implementation_after'] == report['implementation_before']
        report['checkpoint_after'] = fingerprint(setup['checkpoint'])
        report['checkpoint_unchanged'] = report['checkpoint_before'] == report['checkpoint_after']
        hashes_after = _hash_tree(snapshot_dir)
        report.setdefault('branch', {})['snapshot_hashes_after'] = hashes_after
        report['branch']['snapshot_immutable'] = hashes_after == hashes_before
        if not report['branch']['snapshot_immutable']:
            report['status'] = 'failed'
            report['termination_reason'] = 'snapshot_mutated_during_branch'
        report['wall_seconds'] = time.monotonic() - began
        atomic_json(output / 'run_report.json', report)
    return report


def _verification_gate(worker_folder, process, record_dir, index):
    """Feed the recorded step-`index` observation through the restored worker and
    compare perception against the record run (same <=1e-4 bar as the existing
    prefix-replay verifier)."""
    packet = record_dir / f'uniad_worker/step_{index:03d}/observation.pkl'
    if not packet.exists():
        raise FileNotFoundError('Record run must keep observation packets: ' + str(packet))
    atomic_json(worker_folder / f'request_{index:03d}.json',
        dict(index=index, packet=str(packet)))
    response = wait_json(worker_folder / f'done_{index:03d}.json', process, timeout=300.)
    diffs = _compare_perception(
        record_dir / f'uniad_worker/step_{index:03d}/perception.npz',
        response['perception'])
    return dict(index=index, tolerance=PERCEPTION_TOLERANCE, differences=diffs,
                passed=all(v <= PERCEPTION_TOLERANCE for v in diffs.values()))


def main():
    request = _hugsim_expand(json.loads(Path(sys.argv[1]).read_text()))
    if request.get('schema') not in ('branch_experiment_v1', 'patch_experiment_v1'):
        raise ValueError('Unexpected branch experiment request')
    mode = request.get('mode')
    if mode != 'record':
        raise ValueError('heading scene v1 requires fresh record; old snapshots forbidden')
    if mode == 'record':
        run_record(request)
    elif mode == 'branch':
        run_branch(request, replay=False)
    elif mode == 'replay-branch':
        run_branch(request, replay=True)
    else:
        raise ValueError('Unknown mode: ' + str(mode))


if __name__ == '__main__':
    main()
