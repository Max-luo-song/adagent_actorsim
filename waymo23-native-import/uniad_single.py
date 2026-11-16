"""Reuse local UniAD initialization, replacing its FIFO loop in memory only."""
import ast
from datetime import datetime
import os
from pathlib import Path
import sys
import textwrap


TAIL = r'''
import json as _json
import math as _math
import os as _os
import pickle as _pickle
from pathlib import Path as _Path
from queue import Queue as _Queue
import time as _time
import numpy as _np
from dataparser import parse_raw as _parse_raw
from projects.mmdet3d_plugin.uniad.input_geometry import camera_projection as _camera_projection
from runtime_configuration import model_input_profile as _model_input_profile
_profile = _model_input_profile()

_out = _Path(_os.environ['UNIAD_SMOKE_OUTPUT'])
with open(_os.environ['UNIAD_PACKET'], 'rb') as _f:
    _obs, _info = _pickle.load(_f)
_cameras = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT',
            'CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
_norm = dict(mean=_np.array(_profile["mean"]),
             std=_np.array(_profile["std"]),to_rgb=_profile["to_rgb"])
if _info['command'] not in (0,1,2):
    raise ValueError('invalid navigation command')
# Native render RGB -> BGR expected by this mmcv normalization configuration.
_bgr = {'rgb':{_name:_obs['rgb'][_name][...,::-1].copy() for _name in _cameras}}
_data = _parse_raw((_bgr,_info),_cameras,_norm,_Queue())
_data.pop('raw_imgs',None)
_projections, _intrinsics = [], []
for _name in _cameras:
    _im = _obs['rgb'][_name]
    _p = _info['cam_params'][_name]
    _k, _projection = _camera_projection(_p, _im.shape, (_profile['width'], _profile['height']))
    _projections.append(_projection)
    _intrinsics.append(_k)
_data['img_metas'][0][0]['lidar2img'] = _projections
_data['img_metas'][0][0]['intrinsics'] = _intrinsics
print('Input shape:',tuple(_data['img'].shape),flush=True)
torch.cuda.synchronize()
_start = _time.perf_counter()
with torch.no_grad():
    _results = model(return_loss=False,rescale=True,**_data)
torch.cuda.synchronize()
_elapsed = _time.perf_counter()-_start
_traj = _results[0]['planning']['result_planning']['sdc_traj'].detach().cpu().numpy()
if _traj.ndim == 3 and _traj.shape[0] == 1:
    _traj = _traj[0]
if _traj.ndim != 2 or _traj.shape[1] != 2 or len(_traj)<2 or not _np.isfinite(_traj).all():
    raise RuntimeError('invalid planning trajectory: '+repr(_traj))
_np.save(_out/'planning_traj.npy',_traj)
_report = dict(status='single UniAD inference passed',trajectory=_traj.tolist(),
               trajectory_shape=list(_traj.shape),input_shape=list(_data['img'].shape),
               camera_order=_cameras,command=int(_info['command']),
               inference_seconds=_elapsed,projection_resize_corrected=True,
               rgb_to_bgr=True,virtual_rear_views=True,
               rear_view_quality_validated=False,closed_loop_tested=False,
               driving_quality_validated=False,collision_and_scoring_validated=False)
(_out/'uniad_report.json').write_text(_json.dumps(_report,indent=2))
print(_json.dumps(_report,indent=2));print('OUTPUT:',_out)
'''


def replace_loop(tree):
    functions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main']
    if len(functions) != 1:
        raise RuntimeError('expected exactly one UniAD main function')
    fn = functions[0]
    cuts = [i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='cameras' for t in n.targets)]
    if len(cuts) != 1:
        raise RuntimeError('UniAD initialization structure differs; no source files changed')
    fn.body = fn.body[:cuts[0]] + ast.parse(textwrap.dedent(TAIL)).body
    return ast.fix_missing_locations(tree)


def main():
    base = Path(os.environ['BASE'])
    repo = base/'code/UniAD_SIM'
    packet = Path(os.environ['UNIAD_PACKET'])
    source = repo/'tools/closeloop/e2e.py'
    for p in (packet,source,repo/'ckpts/uniad_base_e2e.pth'):
        if not p.is_file():
            raise FileNotFoundError(p)
    out = base/'outputs'/('waymo23_uniad_single_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True)
    os.environ['UNIAD_SMOKE_OUTPUT'] = str(out)
    sys.path[:0] = [str(repo),str(source.parent)]
    os.chdir(repo)
    sys.argv = [str(source),'projects/configs/stage2_e2e/base_e2e.py',
                'ckpts/uniad_base_e2e.pth','--launcher','none','--eval','bbox',
                '--tmpdir',str(out/'tmp'),'--output',str(out)]
    tree = replace_loop(ast.parse(source.read_text()))
    print('OUTPUT:',out,flush=True)
    exec(compile(tree,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})


if __name__ == '__main__':
    main()
