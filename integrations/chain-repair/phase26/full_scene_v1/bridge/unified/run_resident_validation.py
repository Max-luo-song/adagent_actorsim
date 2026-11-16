"""Bounded real 552/reset/551/552 observation and single-worker inference test."""

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
import copy
import hashlib
import json
import os
pass  # Path is provided by the portable boundary above
import pickle
import subprocess
import sys
import time

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent
ROOT = next(p for p in HERE.parents if (p / 'code/UniAD_SIM').is_dir())
ADAPTER = ROOT / 'waymo23-native-import'
sys.path[:0] = [str(HERE), str(BRIDGE/'weather_runtime'), str(ADAPTER), str(ROOT/'integrations/sr-hugsim')]


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def wait_json(path, process, timeout):
    deadline = time.monotonic() + timeout
    progress = time.monotonic() + 15
    while not path.exists():
        if process.poll() is not None:
            raise RuntimeError(f'UniAD exited ({process.returncode}) before {path}; inspect uniad.log')
        if time.monotonic() >= deadline:
            raise TimeoutError(f'UniAD timeout waiting for {path}; inspect uniad.log')
        if time.monotonic() >= progress:
            print(f'Waiting for resident UniAD: {path.name}', flush=True)
            progress += 15
        time.sleep(.1)
    return _hugsim_expand(json.loads(path.read_text()))


def preflight(primary, secondary):
    from scene_edit_session import saf
    from backend import FRAMEWORK_DATA_ROOT
    a, b = saf.SceneSpec.read(primary), saf.SceneSpec.read(secondary)
    # Identity requires a genuinely different scene and different trained model.
    if a.source['scene'] == b.source['scene'] or Path(a.source['checkpoint']).resolve() == Path(b.source['checkpoint']).resolve():
        raise ValueError('Different scene IDs and checkpoint files are required')
    for spec in (a,b):
        if spec.frames <= 81 or spec.fps != 10.:
            raise ValueError('Validation requires frames80/81 and a 10Hz source')
        # FrameworkSceneBackend uses the complete training root, not the legacy
        # native-renderer data_dir (scene552's legacy copy lacks humanpose).
        data_dir = Path(FRAMEWORK_DATA_ROOT) / str(spec.source['scene'])
        for folder in ('humanpose','images','ego_pose','intrinsics','extrinsics'):
            if not (data_dir/folder).is_dir():
                raise FileNotFoundError(data_dir/folder)
        for frame in (0,80,81):
            archived=Path(spec.source['inputs'])/'ego_pose'/f'{spec.source["start_timestep"]+frame:03d}.txt'
            actual=data_dir/'ego_pose'/archived.name
            if not np.allclose(np.loadtxt(archived),np.loadtxt(actual),atol=1e-8,rtol=0):
                raise ValueError('Archived pose and framework dataset disagree')
    return a,b


def compare_perception(before, after):
    row={}
    with np.load(before) as a,np.load(after) as b:
        for key in ('trajectory','occupancy','candidate_boxes','candidate_scores','candidate_labels'):
            if key not in a or key not in b or a[key].shape!=b[key].shape:
                raise RuntimeError(f'Replay output shape/key mismatch: {key}')
            delta=float(np.max(np.abs(a[key].astype(float)-b[key].astype(float)))) if a[key].size else 0.
            row[key]=delta
    return row


def render_inputs(out, primary, secondary):
    from scene_edit_session import UnifiedSceneSession
    from runtime_backend import RuntimeWeatherBackend
    from lifecycle_probe import recorded_pose
    from uniad_bridge import to_uniad
    from build_source_sequence import training_lidar_mount, recorded_kinematics
    count, episodes = [], []
    mount = training_lidar_mount(ROOT/'code/HUGSIM/configs/sim/nuscenes_camera.yaml')
    def factory(spec):
        folder = out / f'weather_load_{len(count)}'
        count.append(folder)
        return RuntimeWeatherBackend(spec, folder)
    with UnifiedSceneSession(factory) as session:
        for index, (name, manifest) in enumerate([
                ('scene552_first',primary),('scene552_reset',None),
                ('scene551_switch',secondary),('scene552_return',primary)]):
            old_backend, old_editor, old_generation = session.backend, session._editor, session.generation
            if manifest is None:
                session.reset()
                if session.backend is not old_backend:
                    raise RuntimeError('Episode reset replaced the backend')
                transition = dict(same_backend=True, weather_reset_ack=session.backend.last_weather_reset_verified)
            else:
                session.load_scene(manifest)
                transition = dict(old_backend_closed=old_backend is None or old_backend.closed,
                                  new_backend=session.backend is not old_backend)
            transition.update(new_generation=session.generation != old_generation,
                              revision_zero=session.revision == 0, journal_empty=not session.edit_journal())
            if old_editor is not None:
                from lifecycle_probe import rejects
                transition['old_editor_rejected'] = rejects(old_editor.objects, RuntimeError)
            if not all(transition.values()):
                raise RuntimeError('Scene transition did not isolate the previous state')
            spec = session.spec
            inputs = Path(spec.source['inputs'])
            raw_start = spec.source['start_timestep']
            poses = np.stack([np.loadtxt(inputs/'ego_pose'/f'{raw_start+i:03d}.txt') for i in range(spec.frames)])
            relative = np.linalg.inv(poses[0])[None] @ poses
            episode = dict(name=name, scene=spec.source['scene'], generation=session.generation,
                checkpoint=spec.source['checkpoint'], transition=transition, frames=[])
            episode['checkpoint_file'] = dict(bytes=Path(spec.source['checkpoint']).stat().st_size,
                mtime_ns=Path(spec.source['checkpoint']).stat().st_mtime_ns)
            episode['source_report_sha256'] = hashlib.sha256(spec.source_report.read_bytes()).hexdigest()
            episode['gaussian_points_by_group'] = {name:int(model._means.shape[0])
                for name,model in session.backend.trainer.models.items() if hasattr(model,'_means')}
            episode['gaussian_sample_sha256'] = {name:hashlib.sha256(
                model._means[:16].detach().cpu().numpy().tobytes()).hexdigest()
                for name,model in session.backend.trainer.models.items() if hasattr(model,'_means')}
            episodes.append(episode)
            for frame in (80,81):
                packet = session.render(relative[frame], frame/spec.fps)
                speed, steer, command = recorded_kinematics(relative, frame, spec.fps, spec.vehicle.wheelbase_m)
                obs, info = to_uniad(packet, mount, speed_mps=speed, steering_rad=steer,
                                    command=command, scene_token=session.generation, dt=.1)
                folder = out / name / f'frame_{frame}'
                folder.mkdir(parents=True,exist_ok=False)
                path = folder/'observation.pkl'
                with path.open('xb') as stream:
                    pickle.dump((obs,info),stream,protocol=4)
                hashes = {}
                for cam, image in obs['rgb'].items():
                    Image.fromarray(image).save(folder/f'{cam}.png')
                    hashes[cam] = hashlib.sha256(image.tobytes()).hexdigest()
                record=session.backend.weather_records[-1]
                if hashes != {cam:c['rgb_sha256'] for cam,c in record['cameras'].items()}:
                    raise RuntimeError('Rendered weather RGB differs from serialized model inputs')
                episode['frames'].append(dict(packet=str(path), timestamp=info['timestamp'],
                    rgb_sha256=hashes, weather_audit=record['audit']))
            if episode['frames'][0]['weather_audit']['window_frames'] != [80,80,80]:
                raise RuntimeError('New episode retained old weather history')
            atomic_json(out/'episodes.json',episodes)
            print(f'{name}: actual scene {spec.source["scene"]} weather observations saved',flush=True)
    # Replay exactly the first inputs after all transitions, with a fresh episode
    # token. This isolates UniAD reset from possible cross-load renderer drift.
    replay = copy.deepcopy(episodes[0])
    replay.update(name='scene552_identical_input_replay',generation='replay-'+episodes[0]['generation'],
                  transition={'identical_input_replay':True},frames=[])
    for i, frame in enumerate(episodes[0]['frames']):
        with Path(frame['packet']).open('rb') as stream:
            obs,info=pickle.load(stream)
        info['scene_token']=replay['generation']
        folder=out/replay['name']/f'frame_{80+i}';folder.mkdir(parents=True)
        path=folder/'observation.pkl'
        with path.open('xb') as stream:pickle.dump((obs,info),stream,protocol=4)
        replay['frames'].append(dict(frame,packet=str(path)))
    episodes.append(replay)
    atomic_json(out/'episodes.json',episodes)
    return episodes


def inference(out, episodes):
    job=out/'uniad_worker';job.mkdir()
    env=dict(os.environ,BASE=str(ROOT),PYTHONUNBUFFERED='1',BEV_EXPERIMENT_SEED='20261021',
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python',UNIAD_INIT_TRACE='1',UNIAD_STARTUP_REPORT=str(job/'startup.json'))
    for key in ('LD_LIBRARY_PATH','PYTHONPATH','PYTHONHOME'):env.pop(key,None)
    command=[str(ROOT/'tools/pixi-uniad-legacy/.pixi/envs/default/bin/python'),
        str(ADAPTER/'waymo172_closed_loop_v1/closed172_worker.py'),'--job',str(job),
        '--planning-bev','current','--model-profile','calibration_geometry_v1',
        '--diagnostic-closed-loop','--session-lifecycle']
    # five_uniad reads the initial packet before invoking the resident callback.
    first=job/'step_000';first.mkdir()
    import shutil
    shutil.copyfile(episodes[0]['frames'][0]['packet'],first/'observation.pkl')
    completed=[]
    with (out/'uniad.log').open('x') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            ready=wait_json(job/'ready.json',process,1200)
            atomic_json(out/'worker_ready.json',ready)
            index=0
            for episode in episodes:
                for frame_index, frame in enumerate(episode['frames']):
                    request=dict(index=index,packet=frame['packet'])
                    if frame_index==0:request['diagnostic_seed']=20261021
                    atomic_json(job/f'request_{index:03d}.json',request)
                    done=wait_json(job/f'done_{index:03d}.json',process,300)
                    if done.get('status')!='ok':raise RuntimeError(str(done))
                    folder=job/f'step_{index:03d}'
                    temporal=_hugsim_expand(json.loads((folder/'temporal_state.json').read_text()))
                    audit=_hugsim_expand(json.loads((folder/'session_lifecycle.json').read_text()))
                    if temporal['history_bev_present'] != (frame_index==1):
                        raise RuntimeError('Resident inference history did not reset/persist correctly')
                    completed.append(dict(index=index,episode=episode['name'],scene=episode['scene'],
                        temporal=temporal,session=audit,done=done))
                    atomic_json(out/'inference_records.json',completed)
                    print(f'Resident UniAD {index+1}/10: {episode["name"]}, history={frame_index==1}',flush=True)
                    index+=1
        finally:
            atomic_json(job/'stop.json',{})
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
    checks=dict(ten_inferences=len(completed)==10,
        one_worker=all(r['session']['identity']==ready['resident_identity'] for r in completed)
            and ready['pid']==process.pid and ready['uniad_model_loads']==1,
        five_clean_resets=all(completed[i]['session']['reset'] and completed[i]['session']['reset_state_cleared'] for i in range(0,10,2)),
        within_episode_history=all(not completed[i]['session']['reset'] for i in range(1,10,2)),
        reset_counter=[r['session']['reset_count'] for r in completed]==[1,1,2,2,3,3,4,4,5,5],
        actual_different_scenes=episodes[0]['scene']!=episodes[2]['scene'] and episodes[0]['checkpoint']!=episodes[2]['checkpoint'],
        loaded_gaussian_models_differ=episodes[0]['gaussian_sample_sha256']!=episodes[2]['gaussian_sample_sha256'],
        returned_model_matches=episodes[0]['gaussian_sample_sha256']==episodes[3]['gaussian_sample_sha256'],
        switched_images_differ=episodes[0]['frames'][0]['rgb_sha256']!=episodes[2]['frames'][0]['rgb_sha256'],
        weather_reset_each_episode=all(e['frames'][0]['weather_audit']['window_frames']==[80,80,80] for e in episodes[:4]))
    comparisons=[]
    for before,after in [(0,8),(1,9)]:
        comparisons.append(compare_perception(job/f'step_{before:03d}/perception.npz',job/f'step_{after:03d}/perception.npz'))
    checks['identical_input_outputs_match']=all(np.isfinite(v) and v<=1e-4 for row in comparisons for v in row.values())
    atomic_json(out/'acceptance.json',dict(passed=all(checks.values()),checks=checks,
        replay_output_differences=comparisons,scope='Real framework 552/reset/551/552 weather observations; one resident UniAD; identical-input replay',
        closed_loop_driving=False,formal_evaluation_admitted=False))
    if not all(checks.values()):raise RuntimeError('Resident validation failed; inspect acceptance.json')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--primary',type=Path,default=BRIDGE.parent.parent/'phase4/dynamic552_frame80/scene_manifest.json')
    parser.add_argument('--secondary',type=Path,default=HERE/'scene_switch_551.json')
    args=parser.parse_args()
    a,b=preflight(args.primary,args.secondary)
    if args.check_only:
        print(json.dumps(dict(passed=True,scenes=[a.source['scene'],b.source['scene']],gpu_executed=False)))
        return
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__).resolve(),ADAPTER/'resident_session_state.py',ADAPTER/'waymo172_closed_loop_v1/closed172_worker.py',
             ADAPTER/'run_reliable_sequence.py',HERE/'scene_edit_session.py',BRIDGE/'weather_runtime/runtime_backend.py',
             BRIDGE/'weather_runtime/weather_worker.py',BRIDGE/'framework_backend/backend.py',
             args.primary.resolve(),args.secondary.resolve(),a.source_report,b.source_report,
             Path(a.source['source_config']),Path(b.source['source_config'])]
    atomic_json(out/'implementation.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    try:
        episodes=render_inputs(out,args.primary,args.secondary)
        inference(out,episodes)
    except Exception as exc:
        atomic_json(out/'failure.json',dict(error=repr(exc),formal_evaluation_admitted=False))
        raise


if __name__=='__main__':main()
