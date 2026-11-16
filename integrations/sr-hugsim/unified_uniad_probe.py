"""Frozen UniAD replay on three matched SceneSession edit arms, no controls executed."""

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

import gc
import json
import os
pass  # Path is provided by the portable boundary above
import pickle
import subprocess
import sys
import time
from uuid import uuid4
import numpy as np
from scene_api import SceneSession,SceneSpec,BASE
from uniad_bridge import to_uniad,placement_reference

ADAPTER=BASE/'waymo23-native-import'
sys.path[:0]=[str(ADAPTER),str(ADAPTER/'waymo172_closed_loop_v1')]
from closed172_common import atomic_json,wait_json
from build_source_sequence import training_lidar_mount,recorded_kinematics
from frozen_bev_experiment import fingerprint

ROLES=('baseline','inserted','moved')
SEED=20261021


def run(request_path,output):
    request=_hugsim_expand(json.loads(Path(request_path).read_text()))
    if request.get('schema')=='unified_uniad_replay_v1':
        from unified_uniad_replay import run as replay
        return replay(request_path,output)
    if set(request)!={'schema','interface_request','frames'} or request['schema']!='unified_uniad_probe_v1' or request['frames']!=5:
        raise ValueError('This diagnostic accepts exactly five frames per arm')
    interface=_hugsim_expand(json.loads(Path(request['interface_request']).read_text()))
    if interface['schema']!='unified_scene_smoke_v1' or not 1<=len(interface['cases'])<=2:
        raise ValueError('Expected one or two shared interface scene manifests')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    os.environ['BASE']=str(BASE)
    os.environ['MPLCONFIGDIR']=str(BASE/'outputs/gpu_test_service/matplotlib_cache')
    checkpoint=BASE/'code/UniAD_SIM/ckpts/uniad_base_e2e.pth'
    report=dict(schema='unified_uniad_diagnostic_v1',status='running',training=False,
        n1_validation_suspended=True,formal_evaluation_admitted=False,closed_loop_run=False,
        control_actions_executed=False,planning_bev='current',perception_bev='history',
        model_input_profile='calibration_geometry_v1',seed=SEED,sequences=[],inference=[])
    atomic_json(output/'run_report.json',report)
    atomic_json(output/'progress.json',dict(stage='hashing_frozen_checkpoint'))
    report['checkpoint_before']=fingerprint(checkpoint)
    mount=training_lidar_mount(BASE/'code/HUGSIM/configs/sim/nuscenes_camera.yaml')
    calibration=_hugsim_expand(json.loads(Path(interface['asset_calibration']).read_text()))
    from asset_placement import terrain_aligned_pose
    from PIL import Image
    with SceneSession() as session:
        for case in interface['cases']:
            original=SceneSpec.read(case['scene_manifest']);scene=original.source['scene']
            manifest=_hugsim_expand(json.loads(Path(case['scene_manifest']).read_text()));manifest['render_scale']=1.
            path=output/f'waymo{scene}_inference_manifest.json';atomic_json(path,manifest)
            atomic_json(output/'progress.json',dict(stage='loading_scene',scene=scene))
            session.load_scene(path)
            audit=_hugsim_expand(json.loads(Path(case['sensor_audit']).read_text()))
            if Path(audit['source_report']).resolve()!=original.source_report:raise ValueError('Wrong sensor audit')
            start=int(audit['automatic_start_frame']);indices=list(range(start,start+request['frames']))
            if indices[-1]>=original.frames:raise ValueError('Frame window exceeds source')
            inputs=Path(original.source['inputs']);raw_start=original.source['start_timestep']
            poses=np.stack([np.loadtxt(inputs/'ego_pose'/f'{raw_start+i:03d}.txt') for i in range(original.frames)])
            relative=np.linalg.inv(poses[0])[None]@poses
            pose=terrain_aligned_pose(audit['proposed_anchor_source_xyz'],audit['heading_source_radians'],
                audit['ground_fit'],calibration['support']['bottom_y'])
            insert=dict(operation='insert',object_id='preview_car',mode='diagnostic_preview',
                asset=interface['asset'],asset_calibration=interface['asset_calibration'],asset_to_world=pose.tolist())
            for role in ROLES:
                if role=='inserted':session.edit(insert)
                if role=='moved':
                    moved=pose.copy();moved[:3,3]+=.5*pose[:3,2]
                    session.edit(dict(operation='set_pose',object_id='preview_car',mode='diagnostic_preview',asset_to_world=moved.tolist()))
                folder=output/'sequences'/str(scene)/role;folder.mkdir(parents=True)
                token=str(uuid4())
                seq=dict(scene=scene,role=role,frames=[],model_input_profile=report['model_input_profile'],
                    planning_bev='current',scene_token=token,training=False,closed_loop_run=False,
                    formal_evaluation_admitted=False,n1_certified=False,control_actions_executed=False)
                ref_request=dict(insert)
                if role=='moved':ref_request['asset_to_world']=moved.tolist()
                reference=placement_reference(ref_request,calibration)
                atomic_json(folder/'asset_reference.json',reference)
                for index,frame in enumerate(indices):
                    atomic_json(output/'progress.json',dict(stage='rendering_unified_observations',scene=scene,role=role,frame=frame))
                    observation=session.render(relative[frame],frame/original.fps)
                    speed,steer,command=recorded_kinematics(relative,frame,original.fps,original.vehicle.wheelbase_m)
                    lo,hi=max(0,frame-1),min(original.frames-1,frame+1)
                    vlo,slo,_=recorded_kinematics(relative,lo,original.fps,original.vehicle.wheelbase_m)
                    vhi,shi,_=recorded_kinematics(relative,hi,original.fps,original.vehicle.wheelbase_m)
                    obs,info=to_uniad(observation,mount,speed_mps=speed,steering_rad=steer,command=command,
                        scene_token=token,dt=original.vehicle.dt,acceleration_mps2=(vhi-vlo)*original.fps/(hi-lo),
                        steering_rate_radps=(shi-slo)*original.fps/(hi-lo))
                    target=folder/f'frame_{index:03d}';target.mkdir()
                    with (target/'observation.pkl').open('wb') as stream:pickle.dump((obs,info),stream,protocol=4)
                    for name,rgb in obs['rgb'].items():Image.fromarray(rgb).save(target/(name+'.png'))
                    seq['frames'].append(dict(frame=frame,timestamp=info['timestamp'],packet=str(target/'observation.pkl')))
                    atomic_json(folder/'sequence_report.json',seq)
                    print('Rendered unified input',scene,role,frame,flush=True)
                report['sequences'].append(str(folder/'sequence_report.json'));atomic_json(output/'run_report.json',report)
    return infer_existing(output,report)


def infer_existing(output,report):
    """Replay already captured observations without loading the renderer again."""
    checkpoint=BASE/'code/UniAD_SIM/ckpts/uniad_base_e2e.pth'
    if not report.get('render_reused_from'):
        import torch
        gc.collect();torch.cuda.empty_cache()
    atomic_json(output/'progress.json',dict(stage='loading_resident_uniad'))
    worker_folder=output/'uniad_worker';(worker_folder/'step_000').mkdir(parents=True)
    first=_hugsim_expand(json.loads(Path(report['sequences'][0]).read_text()))['frames'][0]['packet']
    (worker_folder/'step_000/observation.pkl').symlink_to(first)
    env=dict(os.environ,BASE=str(BASE),PYTHONUNBUFFERED='1',BEV_EXPERIMENT_SEED=str(SEED),
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python')
    for key in ('LD_LIBRARY_PATH','PYTHONPATH','PYTHONHOME'):env.pop(key,None)
    if report.get('render_reused_from'):
        env['UNIAD_INIT_TRACE']='1'
    worker=None
    try:
        with (output/'uniad_worker.log').open('w') as log:
            worker=subprocess.Popen([str(BASE/'tools/pixi-uniad-legacy/.pixi/envs/default/bin/python'),
                str(ADAPTER/'waymo172_closed_loop_v1/closed172_worker.py'),'--job',str(worker_folder),
                '--planning-bev','current','--model-profile','calibration_geometry_v1','--diagnostic-closed-loop'],
                env=env,cwd=BASE,stdout=log,stderr=subprocess.STDOUT)
            ready=wait_json(worker_folder/'ready.json',worker,timeout=1200. if report.get('render_reused_from') else 900.)
            report['worker_ready']=ready
            if ready['planning_bev']!='current' or not ready['diagnostic_closed_loop']:raise ValueError('Wrong inference profile')
            serial=0
            for sequence_path in report['sequences']:
                seq=_hugsim_expand(json.loads(Path(sequence_path).read_text()));entries=[]
                for index,record in enumerate(seq['frames']):
                    atomic_json(output/'progress.json',dict(stage='uniad_inference',scene=seq['scene'],role=seq['role'],frame=record['frame'],completed=serial,total=sum(len(_hugsim_expand(json.loads(Path(p).read_text()))['frames']) for p in report['sequences'])))
                    request=dict(index=serial,packet=record['packet'])
                    if index==0:request['diagnostic_seed']=SEED
                    atomic_json(worker_folder/f'request_{serial:03d}.json',request)
                    result=wait_json(worker_folder/f'done_{serial:03d}.json',worker,timeout=300.)
                    if not np.isclose(result['timestamp'],record['timestamp']):raise ValueError('Model response clock mismatch')
                    entries.append(dict(record,index=serial,perception=result['perception'],reliability=result['reliability']))
                    print('UniAD result',seq['scene'],seq['role'],record['frame'],flush=True)
                    serial+=1
                report['inference'].append(dict(scene=seq['scene'],role=seq['role'],sequence=sequence_path,records=entries))
                atomic_json(output/'run_report.json',report)
    finally:
        atomic_json(worker_folder/'stop.json',{})
        if worker is not None:
            try:worker.wait(timeout=20.)
            except subprocess.TimeoutExpired:
                worker.terminate()
                try:worker.wait(timeout=10.)
                except subprocess.TimeoutExpired:worker.kill();worker.wait()
        report['checkpoint_after']=fingerprint(checkpoint)
        report['checkpoint_unchanged']=report['checkpoint_after']==report['checkpoint_before']
        atomic_json(output/'run_report.json',report)
    if not report['checkpoint_unchanged']:raise RuntimeError('Frozen UniAD checkpoint changed')
    atomic_json(output/'progress.json',dict(stage='posthoc_analysis'))
    from uniad_analysis import analyze
    summary=analyze(output,report)
    report.update(status='unified_uniad_diagnostic_completed',analysis=str(output/'analysis.json'))
    atomic_json(output/'run_report.json',report)
    atomic_json(output/'progress.json',dict(stage='completed'))
    return report
