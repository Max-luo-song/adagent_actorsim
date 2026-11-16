"""Five-step offline integration check; persistent scene and UniAD processes."""
import argparse
import ast
import copy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def worker_loop(model, namespace):
    from uniad_single import TAIL
    root = Path(os.environ['UNIAD_JOB'])
    atomic_json(root/'ready.json', {'model_loads':1})
    for i in range(int(os.environ['UNIAD_STEPS'])):
        request = root/f'request_{i:03d}.json'
        deadline = time.monotonic()+300
        while not request.exists():
            if time.monotonic()>deadline:
                raise TimeoutError('waiting for simulator observation')
            time.sleep(.1)
        req = json.loads(request.read_text())
        os.environ['UNIAD_PACKET'] = req['packet']
        os.environ['UNIAD_SMOKE_OUTPUT'] = req['output']
        scope = dict(namespace, model=model)
        exec(compile(TAIL,'single_frame_inference','exec'),scope)
        atomic_json(root/f'done_{i:03d}.json',scope['_report'])


def start_worker():
    base = Path(os.environ['BASE'])
    repo = base/'code/UniAD_SIM'
    source = repo/'tools/closeloop/e2e.py'
    sys.path[:0] = [str(repo),str(source.parent)]
    os.chdir(repo)
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    cuts = [i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='cameras' for t in n.targets)]
    if len(cuts)!=1:
        raise RuntimeError('unexpected UniAD initializer structure')
    fn.body = fn.body[:cuts[0]]+ast.parse(
        'from uniad_closed import worker_loop as _run_worker\n_run_worker(model, globals())').body
    sys.argv = [str(source),'projects/configs/stage2_e2e/base_e2e.py',
                'ckpts/uniad_base_e2e.pth','--launcher','none','--eval','bbox',
                '--tmpdir',str(Path(os.environ['UNIAD_JOB'])/'tmp'),
                '--output',os.environ['UNIAD_JOB']]
    exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),
         {'__name__':'__main__','__file__':str(source)})


def six_views(env):
    from prepare_uniad_input import lidar_to_camera
    obs = env.observe()
    info = env.info()
    original, names = dict(env.cameras), dict(env.cam_names)
    views = {names[cid]:copy.deepcopy(c) for cid,c in original.items()}
    try:
        for name,degrees in {'CAM_BACK':180.,'CAM_BACK_LEFT':-120.,'CAM_BACK_RIGHT':120.}.items():
            c = copy.deepcopy(original[0]);a = math.radians(degrees)
            import numpy as np
            rot = np.eye(4)
            rot[:3,:3] = [[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]]
            c['rig'] = rot@c['rig']
            env.cameras,env.cam_names = {0:c},{0:name}
            rear = env.observe()
            for kind in obs:
                obs[kind].update(rear[kind])
            views[name] = c
    finally:
        env.cameras,env.cam_names = original,names
    info['command'] = 2
    info['cam_params'] = {}
    for name,c in views.items():
        w,h,k = c['width'],c['height'],c['K']
        intr = dict(W=w,H=h,cx=float(k[0,2]),cy=float(k[1,2]),
                    fovx=2*math.atan(w/(2*k[0,0])),fovy=2*math.atan(h/(2*k[1,1])))
        info['cam_params'][name] = dict(intrinsic=intr,l2c=lidar_to_camera(c['rig']))
    return obs,info


def wait_for(path, process, limit=300):
    deadline = time.monotonic()+limit
    while not path.exists():
        if process.poll() is not None:
            raise RuntimeError('UniAD process exited; see uniad.log')
        if time.monotonic()>deadline:
            raise TimeoutError('UniAD timeout; see uniad.log')
        time.sleep(.2)
    return json.loads(path.read_text())


def run(steps, remove):
    import numpy as np
    from PIL import Image
    from live_env import make_env
    base = Path(os.environ['BASE'])
    out = base/'outputs'/('waymo23_uniad_closed_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True)
    print('OUTPUT:',out,flush=True)
    checkpoint = ('/inspire/hdd/global_user/guoluosong-253108120129/adscenegraph/'
                  'Omniscenegraph/tmp/outputs/waymo/waymo_23/'
                  '20260418_lidar+cam0_1_2adstudiov_pbr_3view/checkpoint_final.pth')
    env = make_env(base/'code/HUGSIM',checkpoint,base/'waymo23-inputs',cameras=(0,1,2),dt=.1)
    from trajectory_control import traj2control_fixed as traj2control
    if remove is not None:
        from live_delete_check import apply_delete
        apply_delete(env,remove)
    proc = None
    log = (out/'uniad.log').open('w')
    records,images = [],[]
    try:
        child_env = dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',
                         UNIAD_JOB=str(out),UNIAD_STEPS=str(steps))
        command = [str(base/'tools/pixi/bin/pixi'),'run','--manifest-path',
                   str(base/'tools/pixi-uniad-legacy/pixi.toml'),'python',
                   str(Path(__file__).resolve()),'--worker']
        proc = subprocess.Popen(command,env=child_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print('Loading UniAD once; waiting for ready...',flush=True)
        ready = wait_for(out/'ready.json',proc)
        env.reset()
        for i in range(steps):
            obs,info = six_views(env)
            images.append(Image.fromarray(obs['rgb']['CAM_FRONT']))
            folder = out/f'step_{i:03d}';folder.mkdir()
            packet = folder/'observation.pkl'
            with packet.open('wb') as f:
                pickle.dump((obs,info),f,protocol=4)
            atomic_json(out/f'request_{i:03d}.json',dict(packet=str(packet),output=str(folder)))
            result = wait_for(out/f'done_{i:03d}.json',proc)
            trajectory = np.asarray(result['trajectory'],dtype=np.float32)
            if trajectory.ndim!=2 or trajectory.shape[1]!=2 or not np.isfinite(trajectory).all():
                raise RuntimeError('invalid model trajectory')
            acc,rate = map(float,traj2control(trajectory,info))
            if not np.isfinite([acc,rate]).all():
                raise RuntimeError('nonfinite controller output')
            raw = dict(acc=acc,steer_rate=rate)
            action = {k:float(np.clip(v,env.action_space[k].low,env.action_space[k].high))
                      for k,v in raw.items()}
            next_obs,_,terminated,truncated,next_info = env.step(action)
            record = dict(step=i,time_before=info['timestamp'],time_after=next_info['timestamp'],
                          source_frame=info['source_frame'],trajectory=trajectory.tolist(),
                          raw_control=raw,applied_control=action,ego_before=info['ego_pos'],
                          ego_after=next_info['ego_pos'],speed_after=next_info['ego_velo'],
                          terminated=bool(terminated),truncated=bool(truncated),
                          inference_seconds=result['inference_seconds'])
            records.append(record)
            atomic_json(out/'trace.json',records)
            print('step',i,'t',round(next_info['timestamp'],2),'action',action,
                  'speed',round(next_info['ego_velo'],3),flush=True)
            if terminated or truncated:
                raise RuntimeError('unexpected early episode end during short check')
        images.append(Image.fromarray(next_obs['rgb']['CAM_FRONT']))
        images[0].save(out/'front_closed_loop.gif',save_all=True,append_images=images[1:],duration=100,loop=0)
        images[-1].save(out/'final_front.png')
        proc.wait(timeout=30)
        if proc.returncode!=0:
            raise RuntimeError('UniAD exited with an error; see uniad.log')
        report = dict(status='short UniAD closed-loop integration passed',steps=len(records),
                      scene_checkpoint_loads=env.checkpoint_loads,uniad_model_loads=ready['model_loads'],
                      simulated_seconds=next_info['timestamp'],dt=.1,command=2,removed_instance=remove,
                      controller='forward_right_heading_fixed_v1',
                      virtual_rear_views=True,rear_view_quality_validated=False,
                      driving_quality_validated=False,collision_and_scoring_validated=False)
        atomic_json(out/'closed_report.json',report)
        print(json.dumps(report,indent=2));print('OUTPUT:',out)
    except Exception:
        log.flush()
        print('\nUniAD log tail:\n'+'\n'.join((out/'uniad.log').read_text(errors='replace').splitlines()[-35:]),flush=True)
        raise
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        log.close();env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--worker',action='store_true')
    p.add_argument('--steps',type=int,default=5)
    p.add_argument('--remove',type=int)
    a = p.parse_args()
    if a.worker:
        start_worker()
    else:
        if not 1<=a.steps<=10:
            raise ValueError('steps must be 1..10')
        run(a.steps,a.remove)
