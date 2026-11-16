"""Lossless driver-state checkpoint; model history rebuilt by verified RGB-D replay."""

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

import hashlib,json,os,pickle,shutil
pass  # Path is provided by the portable boundary above
import numpy as np

def save_checkpoint(path, *, implementation, source_result, next_index, token, motion, warm, controller, obstacle_memory, depth_memory, report, minimum_gap):
    payload=dict(schema='closed_loop_driver_state_v1',implementation=implementation,
        source_result=str(source_result),next_index=next_index,token=token,motion=dict(vars(motion)),warm=warm,
        controller=controller,obstacle_memory=obstacle_memory,depth_memory=depth_memory,report=report,minimum_gap=minimum_gap)
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:pickle.dump(payload,f,protocol=4)
    tmp.replace(path)

def load_checkpoint(path,implementation,base,experiment_signature):
    path=Path(path).resolve()
    if Path(base)/'outputs' not in path.parents:raise ValueError('Resume state must be a workspace output')
    with path.open('rb') as f:payload=pickle.load(f)
    if payload.get('schema')!='closed_loop_driver_state_v1' or payload['implementation']!=implementation or payload['report'].get('experiment_signature')!=experiment_signature:
        raise ValueError('Resume implementation fingerprint differs; restart rather than mix code versions')
    return payload

def link_file(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():return
    try:os.link(source,target)
    except OSError:shutil.copy2(source,target)

def copy_frames(payload,output,cameras):
    source=Path(payload['source_result']);output=Path(output)
    for i in range(payload['next_index']):
        for camera in cameras:link_file(source/'closed_loop'/camera/f'{i:03d}.png',output/'closed_loop'/camera/f'{i:03d}.png')
        link_file(source/'comparison_frames'/f'{i:03d}.png',output/'comparison_frames'/f'{i:03d}.png')

def replay_model_history(payload, worker, process, wait_json, atomic_json, atomic_pickle):
    from PIL import Image
    source=Path(payload['source_result']);differences=[]
    for i in range(payload['next_index']):
        prior=source/'uniad_worker'/f'step_{i:03d}';folder=Path(worker)/f'step_{i:03d}';folder.mkdir(exist_ok=True)
        with (prior/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
        with np.load(prior/'observed_depth.npz') as data:depth={name:data[name] for name in data.files}
        obs=dict(depth=depth,rgb={name:np.asarray(Image.open(source/'closed_loop'/name/f'{i:03d}.png')).copy() for name in depth})
        if (prior/'observed_depth_alpha.npz').exists():
            with np.load(prior/'observed_depth_alpha.npz') as data:obs['depth_alpha']={name:data[name] for name in data.files}
        if (prior/'observed_depth_variance.npz').exists():
            with np.load(prior/'observed_depth_variance.npz') as data:obs['depth_variance']={name:data[name] for name in data.files}
        for field in ['surface_depth','surface_valid']:
            if (prior/(field+'.npz')).exists():
                with np.load(prior/(field+'.npz')) as data:obs[field]={name:data[name] for name in data.files}
        packet=folder/'observation.pkl';atomic_pickle(packet,(obs,info));req=dict(index=i,packet=str(packet))
        if i==0:req['diagnostic_seed']=20261021
        atomic_json(Path(worker)/f'request_{i:03d}.json',req);answer=wait_json(Path(worker)/f'done_{i:03d}.json',process,timeout=300.)
        if answer.get('status')!='ok':raise RuntimeError('Resume inference failed')
        with np.load(prior/'perception.npz') as old,np.load(answer['perception']) as new:
            row=dict(index=i)
            for key in ['candidate_boxes','candidate_scores','trajectory','occupancy','geometry_candidate_boxes']:
                if key not in old.files or key not in new.files:raise ValueError('Missing resume comparison field '+key)
                if old[key].shape!=new[key].shape:raise ValueError('Resume prediction shape differs')
                difference=float(np.max(np.abs(old[key].astype(float)-new[key].astype(float)))) if old[key].size else 0.
                if difference>1e-4:raise ValueError('Resume perception/history mismatch: '+key+' '+str(difference))
                row[key+'_max_difference']=difference
            differences.append(row)
        for name in ['sensor_metadata.pkl','sensor_metadata.json','observed_depth.npz']:
            link_file(prior/name,folder/name)
        if (prior/'observed_depth_alpha.npz').exists():link_file(prior/'observed_depth_alpha.npz',folder/'observed_depth_alpha.npz')
        if (prior/'observed_depth_variance.npz').exists():link_file(prior/'observed_depth_variance.npz',folder/'observed_depth_variance.npz')
        for field in ['surface_depth','surface_valid']:
            if (prior/(field+'.npz')).exists():link_file(prior/(field+'.npz'),folder/(field+'.npz'))
        packet.unlink()
    atomic_json(Path(worker)/'resume_verification.json',dict(frames=len(differences),rows=differences,history_restored_by_replay=True,control_actions_during_replay=0))
