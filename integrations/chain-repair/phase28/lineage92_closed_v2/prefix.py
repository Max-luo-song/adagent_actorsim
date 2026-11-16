"""Declared archived prefix, strict state/input gates, then live feedback."""

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

import copy
import json
pass  # Path is provided by the portable boundary above
import pickle
import numpy as np

P28=Path(__file__).resolve().parents[1]
BASELINE=P28/'runs/heading_fix_scene552_v1'

def required_prefix_files():
    files=[]
    for index in range(40):
        folder=BASELINE/f'uniad_worker/step_{index:03d}'
        files.extend([folder/'observation.pkl',folder/'perception.npz'])
        if index<39:files.extend([folder/'optimization.json',folder/'guard_memory_report.json'])
    return files

def index_from_info(info):
    value=(float(info['timestamp'])-8.)/.1
    index=int(round(value))
    if abs(value-index)>1e-6:raise ValueError('Unexpected source clock')
    return index

def archived_packet(index):
    with (BASELINE/f'uniad_worker/step_{index:03d}/observation.pkl').open('rb') as f:return pickle.load(f)

def bind_packet(original):
    _,first_info=archived_packet(0)
    scene_token=first_info['scene_token']
    def packet(rendered):
        live_obs,live_info=original(rendered)
        index=index_from_info(live_info)
        if index<=39:
            obs,info=archived_packet(index)
            for field in ('ego_velo','ego_steer','explicit_l2g_r_mat','explicit_l2g_t'):
                if not np.allclose(live_info[field],info[field],rtol=0,atol=1e-9):
                    raise RuntimeError('Archived prefix state mismatch: '+field+' at '+str(index))
            # The composed frame must show the same RGB actually fed to the
            # model; copy packet dictionaries rather than altering backend buffers.
            rendered['cameras']={name:dict(camera) for name,camera in rendered['cameras'].items()}
            for name,rgb in obs['rgb'].items():rendered['cameras'][name]['rgb']=rgb.copy()
            return obs,info
        live_info=dict(live_info,scene_token=scene_token)
        return live_obs,live_info
    return packet

def bind_optimizer(original):
    def optimize(reference,occupancy,info,*args,**kwargs):
        index=index_from_info(info)
        if index>=39:return original(reference,occupancy,info,*args,**kwargs)
        folder=BASELINE/f'uniad_worker/step_{index:03d}'
        with np.load(folder/'perception.npz') as data:
            np.testing.assert_array_equal(reference,data['trajectory'])
            np.testing.assert_array_equal(occupancy,data['occupancy'])
        expected=_hugsim_expand(json.loads((folder/'guard_memory_report.json').read_text()))
        actual=copy.deepcopy(kwargs['reliability']);actual.pop('lineage92_counterfactual',None)
        actual=_hugsim_expand(json.loads(json.dumps(actual,default=lambda x:x.tolist() if hasattr(x,'tolist') else float(x))))
        if actual!=expected:
            differing=[key for key in set(actual)|set(expected) if actual.get(key)!=expected.get(key)]
            raise RuntimeError('Archived prefix guard mismatch at '+str(index)+': '+str(differing))
        _,old_info=archived_packet(index)
        for key in ('ego_velo','ego_steer','model_lidar_frame'):
            if json.dumps(info[key],sort_keys=True)!=json.dumps(old_info[key],sort_keys=True):
                raise RuntimeError('Archived optimizer motion state mismatch: '+key)
        result=_hugsim_expand(json.loads((folder/'optimization.json').read_text()))
        result['prefix_replay_audit']=dict(archived_optimization=True,step=index,live_reoptimization=False,
            observation_and_guard_exactly_verified=True)
        return result
    return optimize
