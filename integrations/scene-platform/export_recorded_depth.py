"""Export actual archived RGB-D observations; no renderer/inference/control runs."""

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

import argparse,pickle,types,sys
pass  # Path is provided by the portable boundary above
import numpy as np
from platform_io import ROOT,read,write,asset
from optional_sensors import Recorder

def export(job,output,indices=(0,10,20,30)):
    job=Path(job);out=Path(output)
    if out.exists():raise FileExistsError('Existing offline export retained: '+str(out))
    task=read(job/'task.json');arm='B' if task['edits'] else 'A';run=job/'work'/arm;report=read(run/'run_report.json')
    sys.path.insert(0,str(ROOT/'integrations/sr-hugsim'))
    from uniad_bridge import B,T
    recorder=Recorder(dict(enabled=True,provider='camera_depth_pointcloud_v1'),out/'pointclouds')
    session=types.SimpleNamespace(spec=types.SimpleNamespace(fps=10),generation='recorded_observation',revision=1 if task['edits'] else 0,backend=types.SimpleNamespace())
    sources=[];rows=[]
    for index in indices:
        path=run/f'uniad_worker/step_{index:03d}/observation.pkl'
        with path.open('rb') as stream:obs,info=pickle.load(stream)
        state=next(o for o in report['observations'] if o['index']==index);ego=np.asarray(state['pose']);mount=np.asarray(info['model_lidar_frame']['virtual_to_ego'])
        if not np.isclose(info['timestamp'],index*.1):raise ValueError('Archived local clock mismatch')
        if not np.allclose(B@(T@ego@T.T)@B.T@mount,np.block([[np.asarray(info['explicit_l2g_r_mat']),np.asarray(info['explicit_l2g_t'])[:,None]],[np.array([[0.,0.,0.,1.]])]]),atol=1e-5):raise ValueError('Archived ego/calibration mismatch')
        cameras={}
        for name,depth in obs['depth'].items():
            camera=info['cam_params'][name];k=camera['intrinsic'];fx=k['W']/(2*np.tan(k['fovx']/2));fy=k['H']/(2*np.tan(k['fovy']/2))
            cam_ego=T.T@B.T@mount@np.linalg.inv(np.asarray(camera['l2c']))
            cameras[name]=dict(depth=depth,depth_alpha=obs['depth_alpha'][name],depth_encoding=info['depth_encoding'],
                K=np.array([[fx,0,k['cx']],[0,fy,k['cy']],[0,0,1.]]),camera_to_world=ego@cam_ego)
        frame=report['initial_context']['source_frame']+index
        packet=dict(source_frame=frame,timestamp_s=frame/10,ego_to_world=ego,cameras=cameras,camera_order=list(cameras),model_cache_token='archived:'+str(index))
        rows.append(recorder.record(session,packet));sources.append(asset(path))
    result=dict(schema='archived_depth_pointcloud_export_v1',source_run=asset(run/'run_report.json'),source_observations=sources,
        source_scene=task['scene_id'],source_revision=session.revision,provider='camera_depth_pointcloud_v1',
        point_counts=[r['point_count'] for r in rows],all_frames_nonempty=all(r['point_count']>0 for r in rows),
        archived_data_reused=True,new_GPU_render=False,new_UniAD_inference=False,new_controls=False,
        edited_vs_original_sync_certified=False,physical_lidar_certified=False,
        explanation='Actual edited-run camera depth exported with archived calibration; no independent scan or paired original-scene comparison.',
        manifest=asset(out/'pointclouds/manifest.json'))
    write(out/'report.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job');p.add_argument('output');a=p.parse_args()
    print(__import__('json').dumps(export(a.job,a.output),indent=2))
