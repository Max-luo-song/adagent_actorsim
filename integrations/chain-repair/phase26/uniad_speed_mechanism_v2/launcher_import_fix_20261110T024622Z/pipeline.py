"""Provenance, geometry, image/query ablations and matched recorded-history probes."""

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

import argparse,copy,datetime,json,os,pickle,shutil,subprocess,sys,time
pass  # Path is provided by the portable boundary above
import numpy as np
LOCAL=Path(__file__).resolve().parent;ROOT=LOCAL.parents[1]
sys.path[:0]=[str(LOCAL.parent/'light-map551-v4'),str(LOCAL.parent/'light-cutin-diagnostic'),str(ROOT/'waymo23-native-import'),str(ROOT/'integrations/sr-hugsim')]
from common import P26,BUNDLE,asset,read,write,verify
from local_boot import local_archive
from runtime import wait_json,stop_worker
OUT=P26/'uniad_speed_mechanism_v2';A=P26/'light_map551_v4';V1=P26/'uniad_speed_audit_v1'
SOURCE_PACKET=A/'A/uniad_worker/step_000/observation.pkl'

def pins():return [asset(p) for p in [*LOCAL.glob('*.py'),LOCAL/'run.sh']]
def cases():
    base=dict(frame=10,timestamp=0.,reset=True,sequence=None,rgb='render',use_can_bus=False,
        pretrained_cameras=False,planning_bev='current',command=2,speed=4.,override_sdc_velocity=False)
    rows=[]
    for name,options in [
        ('render_base',{}),('source_rgb',dict(rgb='source')),
        ('source_can',dict(rgb='source',use_can_bus=True)),
        ('source_camera_embeddings',dict(rgb='source',use_can_bus=True,pretrained_cameras=True)),
        ('blank_rgb',dict(rgb='blank')),('right_command',dict(command=0)),
        ('sdc_velocity_override',dict(override_sdc_velocity=True))]:
        rows.append(dict(base,name=name,**{})|options)
    # Same recorded 551 RGB/poses, recorded current speed. Only sampling/history policy differs.
    # Three frames per stream, no resets within a stream; not simulated ego execution.
    for stream,frames,bev in [('history_01s',[18,19,20],'history'),('history_05s',[10,15,20],'history'),('current_05s',[10,15,20],'current')]:
        for j,f in enumerate(frames):
            rows.append(dict(base,name=stream+'_'+str(j),sequence=stream,frame=f,timestamp=(f-10)/10.,
                reset=j==0,rgb='source',speed='recorded',use_can_bus=True,planning_bev=bev))
    return rows

def projection_coverage(g):
    # Exact original encoder reference-point sampling, using exported actual matrices.
    x=(np.arange(200)+.5)*.512-51.2;X,Y=np.meshgrid(x,x);Z=np.linspace(.5,7.5,4)-5
    p=np.stack([np.broadcast_to(X,(4,200,200)),np.broadcast_to(Y,(4,200,200)),
        np.broadcast_to(Z[:,None,None],(4,200,200)),np.ones((4,200,200))],-1)
    camera=np.einsum('cij,dhwj->cdhwi',np.asarray(g['lidar_to_image_matrices']),p)
    uv=camera[...,:2]/np.maximum(camera[...,2:3],1e-5);H,W=g['image_shapes'][0][:2]
    valid=(camera[...,2]>1e-5)&(uv[...,0]>0)&(uv[...,0]<W)&(uv[...,1]>0)&(uv[...,1]<H)
    union=valid.any((0,1));front=(Y>0)&(Y<40)&(np.abs(X)<4)
    return dict(per_camera_pillar_fraction=valid.any(1).mean((1,2)).tolist(),union_pillar_fraction=float(union.mean()),
        near_front_corridor_0to40m_halfwidth4m_fraction=float(union[front].mean()),
        valid_projection_is_not_proof_of_good_image_features=True)

def cpu(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    trained={};loaded={}
    exec(compile((V1/'checkpoint_training_config.py').read_text(),'checkpoint_config','exec'),{},trained)
    exec(compile((ROOT/'code/UniAD_SIM/projects/configs/stage2_e2e/base_e2e.py').read_text(),'runtime_config','exec'),{},loaded)
    differences=[]
    def compare(a,b,key='model'):
        if isinstance(a,dict) and isinstance(b,dict):
            for k in sorted(set(a)|set(b)):
                if k not in a or k not in b:differences.append(dict(key=key+'.'+k,missing=True))
                else:compare(a[k],b[k],key+'.'+k)
        elif a!=b:differences.append(dict(key=key,trained=a,runtime=b))
    compare(trained['model'],loaded['model'])
    report=read(A/'A/run_report.json');speeds=[];inferred=[]
    for row in report['trace']:
        folder=A/f'A/uniad_worker/step_{row["index"]:03d}'
        with np.load(folder/'motion_head_raw.npz') as z:inferred.append(float(np.linalg.norm(z['motion_query_boxes'][-1,7:9])))
        speeds.append(row['motion_state'][3])
    result=dict(stage='CPU_geometry_and_configuration_verified',model_configuration_differences=differences,
        projection=projection_coverage(read(A/'A/uniad_worker/step_000/input_geometry.json')),
        actual_speed_quantiles_mps=np.quantile(speeds,[0,.5,1]).tolist(),inferred_SDC_bbox_speed_quantiles_mps=np.quantile(inferred,[0,.5,1]).tolist(),
        inferred_bbox_velocity_is_not_direct_planner_speed_input=True,
        previous_GPU_flag_only_probe=asset(V1/'gpu/result.json'),GPU_executed=False,
        unresolved='Need internal feature/image/history comparisons before assigning primary low-speed cause; low inferred bbox speed alone is not causal proof.')
    write(root/'cpu_report.json',result);return result

def build_packet(case,obs0,info0,data,sources):
    import cv2
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from waymo_adapter import SR_TO_HUGSIM as T
    from uniad_bridge import B
    obs=copy.deepcopy(obs0);info=copy.deepcopy(info0);frame=case['frame']
    info.update(closed_loop_run=False,control_actions_executed=False,offline_speed_mechanism_diagnostic=True)
    if case['rgb']=='source':
        # The old render depth cannot be relabeled as measured depth of recorded RGB.
        obs['depth']={}
        for key in ['depth_alpha','depth_variance','surface_depth','surface_valid']:obs.pop(key,None)
        info['depth_encoding']='unavailable_for_recorded_RGB_diagnostic'
        for name,cid in [('CAM_FRONT',0),('CAM_FRONT_RIGHT',2),('CAM_FRONT_LEFT',1)]:
            path=data/f'images/{frame:03d}_{cid}.jpg';cal=data/f'intrinsics/{cid}.txt'
            sources.extend([asset(path),asset(cal)])
            height,width=obs0['rgb'][name].shape[:2]
            with Image.open(path) as im:rgb=np.asarray(im.convert('RGB').resize((width,height),Image.Resampling.BILINEAR)).copy()
            raw=np.loadtxt(cal);K=np.array([[raw[0]/2,0,raw[2]/2],[0,raw[1]/2,raw[3]/2],[0,0,1.]])
            # The framework resizes to half size then undistorts with this exact K.
            intr=info['cam_params'][name]['intrinsic'];np.testing.assert_allclose(K[:2,2],[intr['cx'],intr['cy']],rtol=0,atol=1e-4)
            obs['rgb'][name]=cv2.undistort(rgb,K,raw[4:9])
        p0=np.loadtxt(data/'ego_pose/000.txt');pose=np.linalg.inv(p0)@np.loadtxt(data/f'ego_pose/{frame:03d}.txt')
        native=T@pose@T.T;mount=np.asarray(info['model_lidar_frame']['virtual_to_ego']);virtual=B@native@B.T@mount
        info.update(ego_pos=native[:3,3].tolist(),ego_rot=Rotation.from_matrix(native[:3,:3]).as_euler('XYZ').tolist(),
            explicit_l2g_r_mat=virtual[:3,:3],explicit_l2g_t=virtual[:3,3],ego_kinematics_source='offline_recorded_pose_diagnostic')
        sources.extend([asset(data/'ego_pose/000.txt'),asset(data/f'ego_pose/{frame:03d}.txt')])
    elif case['rgb']=='blank':
        for name in obs['rgb']:obs['rgb'][name][:]=[124,116,104]
    if case['speed']=='recorded':
        lo=max(frame-1,0);hi=min(frame+1,198)
        p0=np.loadtxt(data/f'ego_pose/{lo:03d}.txt');p1=np.loadtxt(data/f'ego_pose/{hi:03d}.txt')
        speed=float(np.linalg.norm((np.linalg.inv(p0)@p1)[:2,3])/((hi-lo)/10.))
        sources.extend([asset(data/f'ego_pose/{lo:03d}.txt'),asset(data/f'ego_pose/{hi:03d}.txt')])
    else:speed=case['speed']
    case=dict(case,declared_speed_mps=speed)
    info.update(ego_velo=speed,timestamp=case['timestamp'],source_frame=frame,command=case['command'],
        simulation_dt=.1 if case['sequence']=='history_01s' else .5 if case['sequence'] else .1,
        scene_token='speed_mechanism_'+(case['sequence'] or case['name']))
    return (obs,info),case

def prepare(root):
    root=Path(root)
    if (root/'manifest.json').exists():
        plan=read(root/'manifest.json')
        for entry in [*plan['implementation'],*plan['sources'],*plan['packets'],plan['source_prediction']]:verify(entry)
        return plan
    cpu(root)
    with SOURCE_PACKET.open('rb') as f:obs,info=pickle.load(f)
    sources=[asset(SOURCE_PACKET),asset(V1/'gpu/result.json')];data=Path(read(BUNDLE/'source_manifest.json')['raw_data'])
    plan=dict(schema='internal_speed_mechanism_v2',cases=[],packets=[],sources=sources,implementation=pins(),
        seed=read(A/'plan.json')['seed'],source_prediction=asset(A/'A/uniad_worker/step_000/perception.npz'),
        virtual_to_ego=info['model_lidar_frame']['virtual_to_ego'],model_loads=1,closed_loop_driving=False,
        observation_only_recorded_history=True,online_ground_truth_or_future_use_by_driving_policy=False,
        preregistered_reproduction_tolerance_m=1e-4,primary_mechanism_certified=False)
    for i,case in enumerate(cases()):
        packet,case=build_packet(case,obs,info,data,sources);folder=root/f'packets/{i:03d}';folder.mkdir(parents=True,exist_ok=True)
        with (folder/'observation.pkl').open('wb') as f:pickle.dump(packet,f,protocol=4)
        write(folder/'case.json',case);plan['cases'].append(case);plan['packets'].append(asset(folder/'observation.pkl'))
    plan['sources']=list({v['path']:v for v in sources}.values());write(root/'manifest.json',plan)
    write(root/'status.json',dict(stage='CPU_prepared_GPU_required',GPU_executed=False,cases=len(plan['cases']),B_not_started=True))
    return dict(prepared=True,cases=len(plan['cases']),root=str(root))

def summarize(root):
    root=Path(root);plan=read(root/'manifest.json');rows=[];predictions={}
    for i,case in enumerate(plan['cases']):
        folder=root/f'worker/step_{i:03d}'
        with np.load(folder/'perception.npz') as z:p=z['trajectory'].copy()
        with np.load(folder/'planner_ablations.npz') as z:np.testing.assert_allclose(p,z['original'],rtol=0,atol=1e-5)
        speed=float(np.linalg.norm(p[0]+np.asarray(plan['virtual_to_ego'])[:2,3])/.5)
        audit=read(folder/'mechanism_audit.json');features=read(folder/'planner_features.json')
        if not case['reset'] and not audit['before']['bev_present']:raise ValueError('History arm has no BEV history')
        rows.append(dict(case=case,first_reference_speed_mps=speed,trajectory=p.tolist(),features=features,
            audit=asset(folder/'mechanism_audit.json')));predictions[case['name']]=p
    with np.load(plan['source_prediction']['path']) as z:reference=z['trajectory'].copy()
    reproduced=bool(np.max(np.abs(predictions['render_base']-reference))<=plan['preregistered_reproduction_tolerance_m'])
    result=dict(GPU_executed=True,closed_loop_driving=False,baseline_reproduced=reproduced,
        all_original_heads_replayed_exactly=True,cases=rows,primary_mechanism_certified=False,
        note='Interpret image, SDC-feature and direct-BEV paths separately. History cadence and planning-BEV pairs use identical recorded images/poses at matched timestamps where available. Synthetic ablations do not certify a safe driving fix.')
    write(root/'result.json',result)
    if not reproduced:raise ValueError('Baseline did not reproduce; do not interpret mechanism')
    return result

def run(root):
    root=Path(root);prepare(root);plan=read(root/'manifest.json')
    if (root/'result.json').exists():return read(root/'result.json')
    if (root/'worker.log').exists():raise ValueError('Previous partial run retained; inspect worker.log')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no mechanism GPU probe started')
    worker=root/'worker';worker.mkdir(exist_ok=True)
    first=worker/'step_000';first.mkdir(exist_ok=True);shutil.copyfile(plan['packets'][0]['path'],first/'observation.pkl');write(first/'case.json',plan['cases'][0])
    env=dict(os.environ,BASE=str(ROOT),BEV_EXPERIMENT_SEED=str(plan['seed']),PYTHONUNBUFFERED='1',
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python',UNIAD_INIT_TRACE='1')
    for k in ['LD_LIBRARY_PATH','PYTHONPATH','PYTHONHOME']:env.pop(k,None)
    env['PYTHONPATH']=local_archive(A/'boot/manifest.json')
    log=(root/'worker.log').open('x');process=None;began=time.monotonic()
    try:
        process=subprocess.Popen([str(ROOT/'tools/pixi-uniad-legacy/.pixi/envs/default/bin/python'),str(LOCAL/'worker.py'),
            '--job',str(worker),'--planning-bev','current','--model-profile','calibration_geometry_v1','--diagnostic-closed-loop'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        wait_json(worker/'ready.json',process,1200.)
        for i,case in enumerate(plan['cases']):
            folder=worker/f'step_{i:03d}';folder.mkdir(exist_ok=True)
            shutil.copyfile(plan['packets'][i]['path'],folder/'observation.pkl');write(folder/'case.json',case)
            request=dict(index=i,packet=str(folder/'observation.pkl'))
            if case['reset']:request['diagnostic_seed']=plan['seed']
            write(worker/f'request_{i:03d}.json',request);wait_json(worker/f'done_{i:03d}.json',process,300.)
            print('Mechanism probe',i+1,'/',len(plan['cases']),case['name'],flush=True)
    finally:stop_worker(worker,process,log)
    prepare(root);result=summarize(root);result['wall_seconds']=time.monotonic()-began;write(root/'result.json',result)
    write(root/'status.json',dict(stage='GPU_mechanism_probe_complete',result=asset(root/'result.json'),GPU_executed=True,primary_mechanism_certified=False))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['cpu','prepare','probe','inspect']);args=p.parse_args()
    result=cpu(OUT) if args.stage=='cpu' else prepare(OUT) if args.stage=='prepare' else run(OUT) if args.stage=='probe' else summarize(OUT)
    print(json.dumps(result,indent=2))
