import argparse
import ast
import json
import math
import os
from pathlib import Path
import pickle
import sys

ORDER = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_SIDE_LEFT','CAM_SIDE_RIGHT']
EMBEDDING_ROWS = [0,1,2,4,5]


def adapt_camera_modules(model):
    import torch
    changes = []
    for name,module in list(model.named_modules()):
        if hasattr(module,'num_cams'):
            if module.num_cams!=6:
                raise ValueError('unexpected num_cams at '+name)
            module.num_cams = 5
            changes.append(dict(module=name,field='num_cams',old=6,new=5))
        if hasattr(module,'cams_embeds'):
            old = module.cams_embeds
            if not isinstance(old,torch.nn.Parameter) or old.ndim!=2 or old.shape[0]!=6:
                raise ValueError('unexpected camera embedding at '+name)
            rows = torch.tensor(EMBEDDING_ROWS,device=old.device)
            module.cams_embeds = torch.nn.Parameter(
                old.detach().index_select(0,rows).clone(),requires_grad=False)
            changes.append(dict(module=name,field='cams_embeds',rows=EMBEDDING_ROWS))
    if not any(x['field']=='cams_embeds' for x in changes):
        raise ValueError('camera embeddings not found')
    if not any(x['field']=='num_cams' for x in changes):
        raise ValueError('camera-count modules not found')
    return changes


def prepare():
    import numpy as np
    from waymo_five_preview import load_factory
    from prepare_uniad_input import lidar_to_camera
    base = Path(os.environ['BASE'])
    job = Path(os.environ['FIVE_JOB'])
    job.mkdir(parents=True,exist_ok=True)
    preview = base/'outputs/waymo23_five_preview_20261015_133005_251427'
    source = json.loads((preview/'preview_report.json').read_text())
    factory = load_factory(base/'waymo23-native-import/live_env.py')
    env = factory(base/'code/HUGSIM',source['checkpoint'],source['inputs'],
                  cameras=(0,1,2,3,4),scale=.5)
    try:
        obs,info = env.reset()
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
                intrinsic=intr,l2c=lidar_to_camera(c['rig']),
                v2c=np.linalg.inv(c['rig']))
        if set(obs['rgb'])!=set(ORDER):
            raise ValueError('five-camera names mismatch')
        with (job/'observation.pkl').open('wb') as f:
            pickle.dump((obs,info),f,protocol=4)
        (job/'source.json').write_text(json.dumps(source,indent=2))
        print('Prepared five real-camera observations:',job/'observation.pkl')
    finally:
        env.close()



def infer_five(model, namespace):
    from uniad_single import TAIL
    old = "_cameras = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT',\n            'CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']"
    if TAIL.count(old)!=1:
        raise ValueError('single-inference camera declaration differs')
    changes = adapt_camera_modules(model)
    print('Camera adaptations:',json.dumps(changes),flush=True)
    tail = TAIL.replace(old,'_cameras = '+repr(ORDER)).replace(
        'virtual_rear_views=True','virtual_rear_views=False')
    scope = dict(namespace,model=model)
    exec(compile(tail,'five_camera_inference','exec'),scope)
    report = scope['_report']
    report.update(status='five real-camera UniAD inference passed',
                  pretrained_camera_count=6,inference_camera_count=5,
                  camera_adaptations=changes,
                  side_embedding_initialization='pretrained back-left and back-right rows',
                  trained_or_finetuned_on_waymo=False)
    out = Path(os.environ['UNIAD_SMOKE_OUTPUT'])
    (out/'uniad_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


def infer():
    base = Path(os.environ['BASE']);job = Path(os.environ['FIVE_JOB'])
    packet = job/'observation.pkl'
    repo = base/'code/UniAD_SIM';source = repo/'tools/closeloop/e2e.py'
    if not packet.is_file():
        raise FileNotFoundError('run --prepare first: '+str(packet))
    out = job/'prediction';out.mkdir(parents=True,exist_ok=True)
    os.environ['UNIAD_PACKET'] = str(packet)
    os.environ['UNIAD_SMOKE_OUTPUT'] = str(out)
    sys.path[:0] = [str(repo),str(source.parent)]
    os.chdir(repo)
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body
              if isinstance(n,ast.FunctionDef) and n.name=='main')
    cuts = [i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='cameras' for t in n.targets)]
    if len(cuts)!=1:
        raise ValueError('unexpected UniAD initializer structure')
    fn.body = fn.body[:cuts[0]]+ast.parse(
        'from five_uniad import infer_five as _infer_five\n_infer_five(model, globals())').body
    from runtime_bundle import checkpoint_path
    frozen_checkpoint=checkpoint_path() or 'ckpts/uniad_base_e2e.pth'
    sys.argv = [str(source),'projects/configs/stage2_e2e/base_e2e.py',
                str(frozen_checkpoint),'--launcher','none','--eval','bbox',
                '--tmpdir',str(out/'tmp'),'--output',str(out)]
    exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),
         {'__name__':'__main__','__file__':str(source)})


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        infer()
