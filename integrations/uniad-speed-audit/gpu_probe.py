"""Four isolated conditions in one resident GPU model; saved real image, no simulation."""

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

import argparse,copy,json,os,pickle,shutil,subprocess,sys,time
pass  # Path is provided by the portable boundary above
import numpy as np
LOCAL=Path(__file__).resolve().parent;ROOT=LOCAL.parents[1]
sys.path[:0]=[str(LOCAL.parent/'light-map551-v4'),str(LOCAL.parent/'light-cutin-diagnostic')]
from common import P26,asset,read,write,verify
from local_boot import local_archive
from runtime import wait_json,stop_worker
OUTPUT=P26/'uniad_speed_audit_v1';A=P26/'light_map551_v4'
CASES=[dict(name='off4',use_can_bus=False,declared_speed_mps=4.),dict(name='off16',use_can_bus=False,declared_speed_mps=16.),
       dict(name='on4',use_can_bus=True,declared_speed_mps=4.),dict(name='on16',use_can_bus=True,declared_speed_mps=16.)]

def prepare(root):
    root=Path(root);report=read(root/'report.json')
    if not report['confirmed_configuration_mismatch']:raise ValueError('No confirmed mismatch')
    for entry in report['sources'].values():verify(entry)
    if (root/'gpu/manifest.json').exists():
        plan=read(root/'gpu/manifest.json')
        for entry in [plan['source_packet'],plan['source_prediction'],*plan['packets'],*plan['implementation']]:verify(entry)
        if plan['cases']!=CASES:raise ValueError('GPU probe cases changed')
        return plan
    source=A/'A/uniad_worker/step_000/observation.pkl'
    with source.open('rb') as f:obs,info=pickle.load(f)
    if info['ego_velo']!=4. or info['timestamp']!=0.:raise ValueError('Expected actual A first image/state')
    plan=dict(schema='one_image_CAN_gate_speed_probe_v1',cases=CASES,seed=read(A/'plan.json')['seed'],
        source_packet=asset(source),source_prediction=asset(A/'A/uniad_worker/step_000/perception.npz'),
        implementation=[asset(p) for p in [*LOCAL.glob('*.py'),LOCAL/'run.sh']],packets=[],
        source_image_pose_and_all_other_model_inputs_identical=True,
        model_loads=1,all_cases_full_state_reset=True,online_training=False,closed_loop_driving=False,
        virtual_to_ego=info['model_lidar_frame']['virtual_to_ego'],
        predeclared_reproduction_tolerance=1e-4,disabled_invariance_tolerance=1e-5,
        interpretation='One-image speed feature/configuration intervention. Cannot certify closed-loop recovery or assign the entire low-speed effect to this flag.')
    for i,case in enumerate(CASES):
        step=root/f'gpu/packets/{i:03d}';step.mkdir(parents=True,exist_ok=True)
        current=copy.deepcopy(info);current['ego_velo']=case['declared_speed_mps'];current['scene_token']='offline_speed_audit_'+case['name']
        # The image and pose stay fixed. This is an explicitly counterfactual input,
        # never represented as an actually executed state or a closed-loop episode.
        with (step/'observation.pkl').open('wb') as f:pickle.dump((obs,current),f,protocol=4)
        plan['packets'].append(asset(step/'observation.pkl'))
    write(root/'gpu/manifest.json',plan);return plan

def summarize(root,plan):
    plans={};rows=[]
    for i,case in enumerate(CASES):
        step=root/f'gpu/worker/step_{i:03d}';audit=read(step/'speed_feature_audit.json')
        with np.load(step/'perception.npz') as z:p=z['trajectory'].copy()
        plans[case['name']]=p
        rows.append(dict(case=case,audit=asset(step/'speed_feature_audit.json'),trajectory=p.tolist(),
            successive_node_speeds_mps=(np.linalg.norm(np.diff(p,axis=0),axis=1)/.5).tolist(),
            first_half_second_reference_speed_with_declared_mount_mps=float(np.linalg.norm(p[0]+np.asarray(plan['virtual_to_ego'])[:2,3])/ .5)))
    with np.load(plan['source_prediction']['path']) as z:baseline=z['trajectory'].copy()
    checks=dict(saved_off4_reproduced=bool(np.max(np.abs(plans['off4']-baseline))<=plan['predeclared_reproduction_tolerance']),
        disabled_speed_4_to16_invariant=bool(np.max(np.abs(plans['off4']-plans['off16']))<=plan['disabled_invariance_tolerance']))
    result=dict(checks=checks,comparison_valid=all(checks.values()),cases=rows,
        max_raw_plan_difference_gate_off_speed_change_m=float(np.max(np.abs(plans['off4']-plans['off16']))),
        max_raw_plan_difference_gate_on_speed_change_m=float(np.max(np.abs(plans['on4']-plans['on16']))),
        max_raw_plan_difference_restore_gate_at_speed4_m=float(np.max(np.abs(plans['off4']-plans['on4']))),
        GPU_executed=True,closed_loop_driving=False,production_fix_validated=False,
        conclusion='Use these four one-image comparisons to quantify gate/speed sensitivity. Normal-speed A and any complete CAN-field corrections still require separately frozen closed-loop validation.')
    write(root/'gpu/result.json',result);return result

def run(root):
    root=Path(root);plan=prepare(root);worker=root/'gpu/worker'
    if (root/'gpu/result.json').exists():return read(root/'gpu/result.json')
    if (worker/'ready.json').exists() or (root/'gpu/worker.log').exists():raise ValueError('Existing partial GPU probe retained; inspect before retry')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no GPU probe started')
    env=dict(os.environ,BASE=str(ROOT),PYTHONUNBUFFERED='1',BEV_EXPERIMENT_SEED=str(plan['seed']),
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python',UNIAD_INIT_TRACE='1',UNIAD_STARTUP_REPORT=str(worker/'startup.json'))
    for k in ['LD_LIBRARY_PATH','PYTHONPATH','PYTHONHOME']:env.pop(k,None)
    env['PYTHONPATH']=local_archive(A/'boot/manifest.json')
    first=worker/'step_000';first.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(plan['packets'][0]['path'],first/'observation.pkl');write(first/'case.json',CASES[0])
    log=(root/'gpu/worker.log').open('x');process=None;started=time.monotonic()
    try:
        process=subprocess.Popen([str(ROOT/'tools/pixi-uniad-legacy/.pixi/envs/default/bin/python'),str(LOCAL/'gpu_worker.py'),
            '--job',str(worker),'--planning-bev','current','--model-profile','calibration_geometry_v1','--diagnostic-closed-loop'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        ready=wait_json(worker/'ready.json',process,1200.)
        if ready['planning_bev']!='current' or ready['model_profile']!='calibration_geometry_v1':raise ValueError('Wrong probe model profile')
        for i,case in enumerate(CASES):
            # Existing trace hooks identify the live step by the largest step directory.
            # Never precreate future worker folders while an earlier case is running.
            step=worker/f'step_{i:03d}';step.mkdir(exist_ok=True)
            shutil.copyfile(plan['packets'][i]['path'],step/'observation.pkl');write(step/'case.json',case)
            write(worker/f'request_{i:03d}.json',dict(index=i,packet=str(step/'observation.pkl'),diagnostic_seed=plan['seed']))
            wait_json(worker/f'done_{i:03d}.json',process,300.)
            print('Offline input probe completed:',case['name'],flush=True)
    finally:stop_worker(worker,process,log)
    for entry in [*plan['implementation'],*read(root/'report.json')['sources'].values()]:verify(entry)
    result=summarize(root,plan);result['wall_seconds']=time.monotonic()-started;write(root/'gpu/result.json',result)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    print(json.dumps(prepare(OUTPUT) if args.stage=='prepare' else run(OUTPUT),indent=2))
