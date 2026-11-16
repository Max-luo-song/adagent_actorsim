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
import math
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

# Only float-valued geometry leaves may differ by 0.1 micrometre.
# Schema, identifiers, booleans, other guard fields and neural arrays remain exact.
GUARD_GEOMETRY_ATOL=1e-7

def verify_guard(actual,expected):
    audit=dict(geometry_float_atol=GUARD_GEOMETRY_ATOL,maximum_absolute_difference=0.0,
        maximum_difference_path=None,differing_float_leaves=0)
    def compare(a,b,path=()):
        where='/'.join(map(str,path))
        if type(a) is not type(b):raise RuntimeError('Archived prefix guard type mismatch: '+where)
        if isinstance(a,dict):
            if a.keys()!=b.keys():raise RuntimeError('Archived prefix guard keys mismatch: '+where)
            for key in a:compare(a[key],b[key],path+(key,))
        elif isinstance(a,list):
            if len(a)!=len(b):raise RuntimeError('Archived prefix guard length mismatch: '+where)
            for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+(i,))
        elif isinstance(a,float):
            if not math.isfinite(a) or not math.isfinite(b):
                raise RuntimeError('Nonfinite prefix guard: '+where)
            delta=abs(a-b)
            allowed=GUARD_GEOMETRY_ATOL if path and path[0] in ('obstacles','geometry_refinement') else 0.0
            if delta>allowed:raise RuntimeError('Archived prefix guard numeric mismatch: '+where+' delta='+str(delta))
            if delta:
                audit['differing_float_leaves']+=1
                if delta>audit['maximum_absolute_difference']:
                    audit.update(maximum_absolute_difference=delta,maximum_difference_path=where)
        elif a!=b:raise RuntimeError('Archived prefix guard value mismatch: '+where)
    compare(actual,expected)
    return audit

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
        guard_audit=verify_guard(actual,expected)
        _,old_info=archived_packet(index)
        for key in ('ego_velo','ego_steer','model_lidar_frame'):
            if json.dumps(info[key],sort_keys=True)!=json.dumps(old_info[key],sort_keys=True):
                raise RuntimeError('Archived optimizer motion state mismatch: '+key)
        result=_hugsim_expand(json.loads((folder/'optimization.json').read_text()))
        result['prefix_replay_audit']=dict(archived_optimization=True,step=index,live_reoptimization=False,
            raw_inputs_exactly_verified=True,guard_numeric_verification=guard_audit)
        return result
    return optimize
