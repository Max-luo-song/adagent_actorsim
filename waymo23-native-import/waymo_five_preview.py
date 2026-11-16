from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

NAMES = {0:'CAM_FRONT',1:'CAM_FRONT_LEFT',2:'CAM_FRONT_RIGHT',
         3:'CAM_SIDE_LEFT',4:'CAM_SIDE_RIGHT'}


def load_factory(path):
    text = path.read_text()
    old = "self.cam_names={0:'CAM_FRONT',1:'CAM_FRONT_LEFT',2:'CAM_FRONT_RIGHT'}"
    if text.count(old)!=1:
        raise RuntimeError('live_env camera definition differs; original file unchanged')
    text = text.replace(old,'self.cam_names='+repr(NAMES))
    namespace = dict(__name__='waymo_five_runtime',__file__=str(path))
    exec(compile(text,str(path),'exec'),namespace)
    return namespace['make_env']


def main():
    base = Path(os.environ['BASE'])
    shared = Path('/inspire/hdd/global_user/guoluosong-253108120129/adscenegraph')
    scene = shared/'tempgls/adstudiopbr/outputwaymo3camspbr/waymo_23/20260413_lidar+cam0_1_2_3_4adstudiov_pbr_5view'
    data = shared/'yxtestom/drivestudio/data/waymo/023'
    out = base/'outputs'/('waymo23_five_preview_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    inputs = out/'inputs';inputs.mkdir(parents=True)
    (inputs/'config.yaml').write_bytes((scene/'config.yaml').read_bytes())
    for name in ('intrinsics','extrinsics','ego_pose'):
        (inputs/name).symlink_to(data/name,target_is_directory=True)
    (inputs/'backup').symlink_to(scene/'backup',target_is_directory=True)
    factory = load_factory(base/'waymo23-native-import/live_env.py')
    print('OUTPUT:',out,flush=True)
    env = factory(base/'code/HUGSIM',scene/'checkpoint_final.pth',inputs,
                  cameras=(0,1,2,3,4),scale=.5)
    try:
        obs,info = env.reset()
        counts = {p['name']:len(p['idx']) for p in env.geometry(0)}
        report = dict(stage='five real-camera HUGSIM preview',checkpoint=str(scene/'checkpoint_final.pth'),
                      original_data=str(data),inputs=str(inputs),frame=0,
                      checkpoint_loads=env.checkpoint_loads,trained_cameras=[0,1,2,3,4],
                      virtual_cameras=[],included_gaussians=counts,cameras={},
                      sky_and_affine_enabled=False,camera_refinement=False,
                      pinhole_preview=True,closed_loop_tested=False,quality_verified=False,
                      source_config_sha256=hashlib.sha256((scene/'config.yaml').read_bytes()).hexdigest())
        order = [3,1,0,2,4]
        canvas = Image.new('RGB',(1600,238),'white');draw = ImageDraw.Draw(canvas)
        for j,cid in enumerate(order):
            name = NAMES[cid];c = env.cameras[cid]
            rgb = obs['rgb'][name]
            if not np.isfinite(rgb).all():
                raise RuntimeError('nonfinite image')
            im = Image.fromarray(rgb);im.save(out/(name+'.png'))
            thumb = ImageOps.contain(im,(320,214))
            x = j*320+(320-thumb.width)//2
            y = 24+(214-thumb.height)//2
            canvas.paste(thumb,(x,y));draw.text((j*320+5,5),name,fill='black')
            meta = dict(camera_id=cid,width=c['width'],height=c['height'],
                        K=c['K'].tolist(),rig_camera_to_ego=c['rig'].tolist(),
                        source_distortion=c['distortion'],
                        nonblack_fraction=float((rgb.max(2)>3).mean()))
            report['cameras'][name] = meta
            print(name,'shape:',rgb.shape,'nonblack:',round(meta['nonblack_fraction'],3),flush=True)
        from five_layout import compose_five
        compose_five(out)
        (out/'preview_report.json').write_text(json.dumps(report,indent=2))
        print('Checkpoint loads:',env.checkpoint_loads)
        print('OUTPUT:',out)
    finally:
        env.close()


if __name__=='__main__':
    main()
