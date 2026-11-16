"""Create six calibrated RGB views for an interface smoke test, not evaluation."""
import copy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import pickle

import numpy as np
from PIL import Image, ImageDraw
from live_env import make_env


def lidar_to_camera(rig):
    native_to_lidar = np.array([[1.,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
    return np.linalg.inv(rig) @ np.linalg.inv(native_to_lidar)


def main():
    base = Path(os.environ['BASE'])
    checkpoint = ('/inspire/hdd/global_user/guoluosong-253108120129/adscenegraph/'
                  'Omniscenegraph/tmp/outputs/waymo/waymo_23/'
                  '20260418_lidar+cam0_1_2adstudiov_pbr_3view/checkpoint_final.pth')
    out = base/'outputs'/('waymo23_uniad_input_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True)
    env = make_env(base/'code/HUGSIM', checkpoint, base/'waymo23-inputs', cameras=(0,1,2))
    try:
        obs, info = env.reset()
        originals = dict(env.cameras)
        names = dict(env.cam_names)
        views = {names[cid]: copy.deepcopy(c) for cid,c in originals.items()}
        virtual = {'CAM_BACK':180., 'CAM_BACK_LEFT':-120., 'CAM_BACK_RIGHT':120.}
        for name, degrees in virtual.items():
            c = copy.deepcopy(originals[0])
            a = math.radians(degrees)
            rot = np.eye(4)
            rot[:3,:3] = [[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]]
            c['rig'] = rot @ c['rig']
            env.cameras = {0:c}
            env.cam_names = {0:name}
            rear = env.observe()
            for kind in obs:
                obs[kind].update(rear[kind])
            views[name] = c
        env.cameras, env.cam_names = originals, names
        info['command'] = 2
        info['cam_rect'] = np.eye(4)
        info['cam_params'] = {}
        order = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT',
                 'CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
        meta = {}
        for name in order:
            c = views[name]
            w,h,k = c['width'],c['height'],c['K']
            l2c = lidar_to_camera(c['rig'])
            intr = dict(W=w,H=h,cx=float(k[0,2]),cy=float(k[1,2]),
                        fovx=2*math.atan(w/(2*k[0,0])),fovy=2*math.atan(h/(2*k[1,1])))
            info['cam_params'][name] = dict(intrinsic=intr,l2c=l2c,v2c=np.linalg.inv(c['rig']))
            resized = np.diag([1600/w,900/h,1.]) @ k
            center = np.linalg.inv(l2c) @ np.array([0.,0.,10.,1.])
            projected = resized @ (l2c @ center)[:3]
            if not np.allclose(projected[:2]/projected[2],resized[:2,2],atol=1e-6):
                raise RuntimeError('projection consistency check failed')
            rgb = obs['rgb'][name]
            Image.fromarray(rgb).save(out/(name+'.png'))
            meta[name] = dict(virtual=name in virtual,width=w,height=h,
                             nonblack_fraction=float((rgb.max(2)>3).mean()),
                             K=k.tolist(),K_1600x900=resized.tolist(),
                             lidar2cam=l2c.tolist(),camera2ego=c['rig'].tolist())
        with (out/'observation.pkl').open('wb') as f:
            pickle.dump((obs,info),f,protocol=4)
        mosaic = Image.new('RGB',(480*3,344*2),'white')
        draw = ImageDraw.Draw(mosaic)
        for i,name in enumerate(order):
            x,y=(i%3)*480,(i//3)*344
            draw.text((x+5,y+5),name+(' (virtual)' if name in virtual else ''),fill='black')
            mosaic.paste(Image.fromarray(obs['rgb'][name]).resize((480,320)),(x,y+24))
        mosaic.save(out/'six_views.png')
        report = dict(status='six-view packet generated; UniAD inference not run',
                      checkpoint_loads=env.checkpoint_loads,command=2,color_order='RGB',
                      virtual_rear_lighting_reference='front camera 0',
                      trained_cameras=[0,1,2],rear_view_quality_validated=False,
                      parser_resize_correction_required=True,
                      collision_and_scoring_validated=False,cameras=meta)
        (out/'input_report.json').write_text(json.dumps(report,indent=2))
        for name in order:
            print(name,'virtual:',meta[name]['virtual'],
                  'nonblack:',round(meta[name]['nonblack_fraction'],3))
        print('OUTPUT:',out)
    finally:
        env.close()


if __name__ == '__main__':
    main()
