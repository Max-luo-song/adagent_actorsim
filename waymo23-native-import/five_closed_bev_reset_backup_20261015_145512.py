import argparse
import json
import math
import os
from pathlib import Path
import pickle


def five_factory(repo, checkpoint, inputs, **kwargs):
    from waymo_five_preview import load_factory
    base = Path(os.environ['BASE'])
    preview = base/'outputs/waymo23_five_preview_20261015_133005_251427'
    source = json.loads((preview/'preview_report.json').read_text())
    if source['trained_cameras']!=[0,1,2,3,4]:
        raise ValueError('expected the five-camera trained checkpoint')
    kwargs['cameras'] = (0,1,2,3,4)
    return load_factory(base/'waymo23-native-import/live_env.py')(
        repo,source['checkpoint'],source['inputs'],**kwargs)


def five_observation(env):
    import numpy as np
    from prepare_uniad_input import lidar_to_camera
    from five_uniad import ORDER
    obs,info = env.observe(),env.info()
    info['command'] = 2
    info['cam_rect'] = np.eye(4)
    info['cam_params'] = {}
    for cid,c in env.cameras.items():
        name = env.cam_names[cid]
        w,h,k = c['width'],c['height'],c['K']
        intr = dict(W=w,H=h,cx=float(k[0,2]),cy=float(k[1,2]),
                    fovx=2*math.atan(w/(2*k[0,0])),
                    fovy=2*math.atan(h/(2*k[1,1])))
        info['cam_params'][name] = dict(
            intrinsic=intr,l2c=lidar_to_camera(c['rig']))
    if set(obs['rgb'])!=set(ORDER):
        raise ValueError('five-camera observations incomplete')
    return obs,info


def five_worker(model, namespace):
    import uniad_single
    from uniad_closed import worker_loop
    from five_uniad import ORDER,adapt_camera_modules
    changes = adapt_camera_modules(model)
    job = Path(os.environ['UNIAD_JOB'])
    (job/'camera_adaptation.json').write_text(json.dumps(changes,indent=2))
    old = "_cameras = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT',\n            'CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']"
    original = uniad_single.TAIL
    if original.count(old)!=1:
        raise ValueError('single-inference camera declaration differs')
    uniad_single.TAIL = original.replace(old,'_cameras = '+repr(ORDER)).replace(
        'virtual_rear_views=True','virtual_rear_views=False')
    from five_ablation_policy import instrument
    uniad_single.TAIL = instrument(uniad_single.TAIL)
    os.environ["ABLATION_MODE"] = "bev_reset"
    try:
        worker_loop(model,namespace)
    finally:
        uniad_single.TAIL = original


def runtime():
    path = Path(os.environ['BASE'])/'waymo23-native-import/uniad_closed_fixed.py'
    text = path.read_text()
    replacements = [
        ('from live_env import make_env',
         'from five_closed_bev_reset import five_factory as make_env'),
        ('obs,info = six_views(env)','obs,info = five_observation(env)'),
        ('from uniad_closed import worker_loop as _run_worker',
         'from five_closed_bev_reset import five_worker as _run_worker'),
        ('virtual_rear_views=True,rear_view_quality_validated=False,',
         'camera_count=5,virtual_rear_views=False,uniad_finetuned_on_waymo=False,bev_history_enabled=False,track_queries_retained=True,diagnostic_only=True,'),
        ("status='short UniAD closed-loop integration passed'",
         "status='five-camera diagnostic loop completed; BEV history disabled'"),
        ("images.append(Image.fromarray(next_obs['rgb']['CAM_FRONT']))",
         "images.append(Image.fromarray(next_obs['rgb']['CAM_FRONT']))\n"
         "        with (out/'final_observation.pkl').open('wb') as f:\n"
         "            pickle.dump((next_obs,next_info),f,protocol=4)"),
        ("        print(json.dumps(report,indent=2));print('OUTPUT:',out)",
         "        print(json.dumps(report,indent=2));print('OUTPUT:',out)\n        return out"),
    ]
    for old,new in replacements:
        if text.count(old)!=1:
            raise ValueError('closed-loop source structure differs: '+old)
        text = text.replace(old,new)
    scope = dict(__name__='five_closed_runtime',
                 __file__=str(Path(__file__).resolve()),
                 five_observation=five_observation)
    exec(compile(text,str(path),'exec'),scope)
    return scope


def make_video(out):
    from PIL import Image,ImageDraw
    packets = sorted(out.glob('step_*/observation.pkl'))+[out/'final_observation.pkl']
    frames = []
    for packet in packets:
        with packet.open('rb') as f:
            obs,info = pickle.load(f)
        canvas = Image.new('RGB',(1440,712),'white')
        draw = ImageDraw.Draw(canvas)
        for i,name in enumerate(['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT']):
            im = Image.fromarray(obs['rgb'][name]).resize(
                (480,320),Image.Resampling.LANCZOS)
            draw.text((i*480+8,8),name,fill='black')
            canvas.paste(im,(i*480,28))
        for i,name in enumerate(['CAM_SIDE_LEFT','CAM_SIDE_RIGHT']):
            im = Image.fromarray(obs['rgb'][name]).resize(
                (720,324),Image.Resampling.LANCZOS)
            draw.text((i*720+8,368),name,fill='black')
            canvas.paste(im,(i*720,388))
        draw.text((8,350),'t='+str(round(info['timestamp'],2))+'s',fill='black')
        frames.append(canvas)
    frames[0].save(out/'five_views_closed_loop.gif',save_all=True,
                   append_images=frames[1:],duration=100,loop=0)
    frames[-1].save(out/'five_views_final.png')
    path = out/'closed_report.json'
    report = json.loads(path.read_text())
    report['five_view_gif'] = 'five_views_closed_loop.gif'
    report['video_frames'] = len(frames)
    path.write_text(json.dumps(report,indent=2))
    print('FIVE-VIEW VIDEO:',out/'five_views_closed_loop.gif')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--steps',type=int,default=10)
    parser.add_argument('--remove',type=int)
    args = parser.parse_args()
    if not 1<=args.steps<=10:
        raise ValueError('steps must be 1..10')
    scope = runtime()
    if args.worker:
        scope['start_worker']()
    else:
        folder = scope['run'](args.steps,args.remove)
        make_video(folder)
