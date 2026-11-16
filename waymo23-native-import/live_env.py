"""Resident SR assets in a HUGSimEnv subclass; action/observation validation only.

Collision, navigation commands and scoring are deliberately unvalidated (None).
No changes to the existing HUGSIM source files or original checkpoints.
"""

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

import argparse
from datetime import datetime
import json
import math
import os
pass  # Path is provided by the portable boundary above
import sys
import time

import numpy as np
import torch
import yaml

from export_snapshot import (camera_metadata,node_geometry,WAYMO_TO_HUGSIM,
                             OPENCV_TO_WAYMO,C0)
from scene_math import matrix_quat,quat_mul,pbr_color,lighting,sh_eval
import torch.nn.functional as F


class Motion:
    """Same integration order as the supplied HUGSIM step(): v, steer, x/z, yaw."""
    def __init__(self,dt=.1,wheelbase=2.7,start_speed=2.):
        if not (np.isfinite([dt,wheelbase,start_speed]).all() and dt>0 and wheelbase>0):
            raise ValueError('invalid motion parameters')
        self.dt,self.wheelbase,self.start_speed=dt,wheelbase,start_speed
        self.reset()

    def reset(self):
        self.x=self.z=self.yaw=self.steer=self.timestamp=self.last_accel=self.last_rate=0.
        self.speed=self.start_speed

    def advance(self,action):
        acc,rate=float(action['acc']),float(action['steer_rate'])
        if not np.isfinite([acc,rate]).all():raise ValueError('nonfinite action')
        self.timestamp+=self.dt
        self.last_accel,self.last_rate=acc,rate
        self.speed+=acc*self.dt;self.steer+=rate*self.dt
        self.x+=self.speed*math.sin(self.yaw)*self.dt
        self.z+=self.speed*math.cos(self.yaw)*self.dt
        self.yaw+=self.speed*math.tan(self.steer)/self.wheelbase*self.dt

    def state(self):
        return dict(x=self.x,z=self.z,yaw=self.yaw,speed=self.speed,steer=self.steer,
                    timestamp=self.timestamp)


def frame_at(t,fps,frames):
    if not (math.isfinite(t) and t>=0 and fps>0 and frames>0):raise ValueError('invalid scene time')
    frame=int(math.floor(t*fps+1e-6))
    return min(frame,frames-1),t>=(frames-1)/fps-1e-9


def make_env(repo,checkpoint,inputs,scale=.5,dt=.1,fps=10.,start_speed=2.,cameras=(0,),camera_names=None,vehicle=None):
    repo,inputs=Path(repo).resolve(),Path(inputs).resolve()
    sys.path[:0]=[str(repo),str(repo/'sim')]
    os.chdir(repo)
    import gymnasium as gym
    from gymnasium import spaces
    from hugsim_env.envs.hug_sim import HUGSimEnv as NativeHUGSimEnv
    import gaussian_renderer as renderer
    from scene.cameras import Camera

    class SourceAssetHUGSimEnv(NativeHUGSimEnv):
        def __init__(self):
            gym.Env.__init__(self)
            self.cfg=yaml.safe_load((inputs/'config.yaml').read_text())
            print('Loading source checkpoint once...',flush=True)
            ckpt=torch.load(checkpoint,map_location='cuda',weights_only=False)
            self.models,self.source_step=ckpt['models'],int(ckpt['step'])
            self.checkpoint_loads=1
            self.frames=len(self.models['RigidNodes']['instances_trans'])
            self.fps=fps
            self.kin=yaml.safe_load((repo/'configs/sim/kinematic.yaml').read_text())
            if vehicle is not None:
                dt_config = vehicle.dt
                self.kin.update(min_acc=vehicle.min_acc_mps2, max_acc=vehicle.max_acc_mps2,
                                max_steer=vehicle.max_steer_rate_degrees)
                self.motion=Motion(dt_config,vehicle.wheelbase_m,start_speed)
            else:
                self.motion=Motion(dt,self.kin['Lr']+self.kin['Lf'],start_speed)
            self.cameras={};self.episode=0;self.finished=False
            self.target=torch.tensor(WAYMO_TO_HUGSIM,dtype=torch.float32,device='cuda')
            self.world_q=matrix_quat(self.target[:3,:3])
            self.pc=renderer.GaussianModel(0,affine=False)
            self.static={};self.frame_cache=None;self.cached_frame=None
            self.names={'Background':2,'RoadNodes':0,'RigidNodes':13,'DeformableNodes':18,'SMPLNodes':11}
            for name,state in self.models.items():
                if '_means' in state and len(state['_means']):
                    if name not in self.names:raise ValueError('unsupported populated node: '+name)
            for cid in cameras:
                c=camera_metadata(inputs,self.cfg,self.models,0,cid,scale,False)
                c['rig']=WAYMO_TO_HUGSIM@np.loadtxt(inputs/'extrinsics'/f'{cid}.txt')@OPENCV_TO_WAYMO
                self.cameras[cid]=c
            source_start=self.cfg['data']['start_timestep']
            e0=np.loadtxt(inputs/'ego_pose'/f'{source_start:03d}.txt')
            points=[]
            for i in range(self.frames):
                pose=np.linalg.inv(e0)@np.loadtxt(inputs/'ego_pose'/f'{source_start+i:03d}.txt')
                points.append((WAYMO_TO_HUGSIM@pose)[:3,3])
            self.reference=np.array(points)
            self.cam_names=dict(camera_names) if camera_names is not None else {0:'CAM_FRONT',1:'CAM_FRONT_LEFT',2:'CAM_FRONT_RIGHT'}
            self.action_space=spaces.Dict({
                'acc':spaces.Box(float(self.kin['min_acc']),float(self.kin['max_acc']),shape=(),dtype=np.float32),
                'steer_rate':spaces.Box(-math.radians(self.kin['max_steer']),math.radians(self.kin['max_steer']),shape=(),dtype=np.float32)})
            obs_spaces={}
            for kind,high,dtype in [('rgb',255,np.uint8),('semantic',19,np.uint8),('depth',np.inf,np.float32)]:
                items={}
                for cid,c in self.cameras.items():
                    shape=(c['height'],c['width'],3) if kind=='rgb' else (c['height'],c['width'])
                    items[self.cam_names[cid]]=spaces.Box(0,high,shape=shape,dtype=dtype)
                obs_spaces[kind]=spaces.Dict(items)
            self.observation_space=spaces.Dict(obs_spaces)
            self.original_concatenate=renderer.concatenate_all
            self.safe_concatenate=lambda fg:self.original_concatenate(fg) if fg else [None]*6
            renderer.concatenate_all=self.safe_concatenate

        def ego_pose(self):
            m=self.motion;pose=np.eye(4);c,s=math.cos(m.yaw),math.sin(m.yaw)
            pose[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]]
            nearest=np.argmin((self.reference[:,0]-m.x)**2+(self.reference[:,2]-m.z)**2)
            pose[:3,3]=[m.x,self.reference[nearest,1],m.z]
            return pose

        @torch.no_grad()
        def geometry(self,frame):
            if self.cached_frame==frame:return self.frame_cache
            parts=[]
            for name,state in self.models.items():
                if '_means' not in state or not len(state['_means']):continue
                if name in self.static:
                    parts.append(self.static[name]);continue
                xyz,q,valid=node_geometry(name,state,self.cfg['model'][name],frame,self.frames,self.source_step)
                idx=valid.nonzero().flatten()
                if not len(idx):continue
                raw_scale=state['_scales']
                if raw_scale.shape[-1]==1:raw_scale=raw_scale.expand(-1,3)
                logits=xyz.new_full((len(idx),20),-12);logits[:,self.names[name]]=12
                part=dict(name=name,xyz=xyz[idx],idx=idx,
                    target_xyz=xyz[idx]@self.target[:3,:3].T,
                    target_q=F.normalize(quat_mul(self.world_q.expand_as(q[idx]),q[idx]),dim=-1),
                    scales=raw_scale[idx],alpha=state['_opacities'][idx].reshape(-1,1),logits=logits)
                if name in ('Background','RoadNodes'):self.static[name]=part
                parts.append(part)
            self.cached_frame,self.frame_cache=frame,parts
            return parts

        @torch.no_grad()
        def observe(self):
            frame,_=frame_at(self.motion.timestamp,self.fps,self.frames)
            parts=self.geometry(frame);ego=self.ego_pose()
            obs={k:{} for k in ('rgb','semantic','depth')}
            for cid,c in self.cameras.items():
                c2w=ego@c['rig']
                source_c2w=np.linalg.inv(WAYMO_TO_HUGSIM)@c2w
                position=torch.tensor(source_c2w[:3,3],dtype=torch.float32,device='cuda')
                camera_index=frame*len(self.cfg['data']['pixel_source']['cameras'])+self.cfg['data']['pixel_source']['cameras'].index(cid)
                light=lighting(self.models['GlobalLighting'],frame,self.frames,camera_index) if 'GlobalLighting' in self.models else None
                colors=[]
                for part in parts:
                    state=self.models[part['name']];idx=part['idx']
                    if part['name']=='Background' and self.cfg['model']['Background'].get('ctrl',{}).get('use_pbr',False):
                        rgb=pbr_color(state,position,light)[idx]
                    else:
                        coeff=torch.cat([state['_features_dc'][idx,None,:],state['_features_rest'][idx]],1)
                        rgb=torch.sigmoid(coeff[:,0]) if coeff.shape[1]==1 else (sh_eval(coeff,F.normalize(part['xyz']-position,dim=-1))+.5).clamp(0,1)
                    colors.append(((rgb-.5)/C0)[:,None,:])
                pc=self.pc
                pc._xyz=torch.cat([p['target_xyz'] for p in parts]);pc._rotation=torch.cat([p['target_q'] for p in parts])
                pc._scaling=torch.cat([p['scales'] for p in parts]);pc._opacity=torch.cat([p['alpha'] for p in parts])
                pc._feats3D=torch.cat([p['logits'] for p in parts]);pc._features_dc=torch.cat(colors)
                pc._features_rest=pc._xyz.new_empty((len(pc._xyz),0,3))
                k=np.eye(4);k[:3,:3]=c['K']
                cam=Camera(width=c['width'],height=c['height'],image=np.zeros((c['height'],c['width'],3)),
                    K=k,c2w=c2w,image_name='resident',dynamics={})
                result=renderer.render(viewpoint=cam,prev_viewpoint=None,pc=pc,dynamic_gaussians={},
                    unicycles={},bg_color=torch.zeros(3,device='cuda'),planning=[])
                name=self.cam_names[cid]
                obs['rgb'][name]=(result['render'].permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8)
                obs['depth'][name]=result['depth'][0].cpu().numpy().astype(np.float32)
                if not np.isfinite(obs['depth'][name]).all():raise RuntimeError('nonfinite depth')
                obs['semantic'][name]=result['feats'].argmax(0).cpu().numpy().astype(np.uint8)
                if getattr(self, 'capture_edit_evidence', False):
                    if not hasattr(self, '_edit_validation_views'):
                        self._edit_validation_views = {}
                    self._edit_validation_views[name] = dict(
                        K=np.asarray(c['K']).copy(), camera_to_world=source_c2w.copy(),
                        rgb=obs['rgb'][name].astype(np.float32)/255.,
                        depth=obs['depth'][name].copy(),
                        depth_encoding=result.get('depth_encoding','unknown'),
                        depth_second_moment=result['depth_second_moment'][0].detach().cpu().numpy().copy(),
                        surface_depth=result['surface_depth'].detach().cpu().numpy().copy(),
                        surface_valid=result['surface_valid'].detach().cpu().numpy().copy(),
                        alpha=result['alphas'].detach().cpu().numpy().reshape(c['height'], c['width']).copy())
            return obs

        def info(self):
            from edit_trust_adapter import runtime_edit_metadata
            frame,_=frame_at(self.motion.timestamp,self.fps,self.frames)
            return dict(**self.motion.state(),ego_pos=self.ego_pose()[:3,3].tolist(),
                ego_rot=[0.,self.motion.yaw,0.],ego_velo=self.motion.speed,ego_steer=self.motion.steer,
                accelerate=self.motion.last_accel,steer_rate=self.motion.last_rate,
                source_frame=frame,checkpoint_loads=self.checkpoint_loads,episode=self.episode,
                collision=None,rc=None,command=None,evaluation_valid=False,
                edit_trust=runtime_edit_metadata(self),
                integration_stage='action_visual_validation',semantic_kind='coarse_node_categories',
                camera_refinement=False,source_fps_assumption=self.fps)

        def reset(self,seed=None,options=None):
            gym.Env.reset(self,seed=seed)
            self.motion.reset();self.finished=False;self.episode+=1
            self.cached_frame=None;self.frame_cache=None
            return self.observe(),self.info()

        def step(self,action):
            if self.finished:raise RuntimeError('episode ended; call reset')
            self.motion.advance(action)
            _,truncated=frame_at(self.motion.timestamp,self.fps,self.frames)
            self.finished=truncated
            return self.observe(),0.0,False,truncated,self.info()

        def close(self):
            if renderer.concatenate_all is self.safe_concatenate:renderer.concatenate_all=self.original_concatenate
            self.models={};self.static={};self.frame_cache=None;self.pc=None
            torch.cuda.empty_cache()

    return SourceAssetHUGSimEnv()


def main():
    from PIL import Image,ImageDraw
    p=argparse.ArgumentParser()
    p.add_argument('--base',default=_hugsim_expand('${HUGSIM_ROOT}'))
    p.add_argument('--steps',type=int,default=10)
    a=p.parse_args();base=Path(a.base).resolve()
    if not 1<=a.steps<=30:raise ValueError('steps must be 1..30')
    checkpoint=_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/adscenegraph/Omniscenegraph/tmp/outputs/waymo/waymo_23/20260418_lidar+cam0_1_2adstudiov_pbr_3view/checkpoint_final.pth')
    if not torch.cuda.is_available():raise RuntimeError('existing HUGSIM CUDA environment required')
    out=base/'outputs'/('waymo23_live_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True,exist_ok=False)
    env=make_env(base/'code/HUGSIM',checkpoint,base/'waymo23-inputs')
    started=time.perf_counter();trace=[]
    try:
        obs,initial=env.reset();Image.fromarray(obs['rgb']['CAM_FRONT']).save(out/'reset.png')
        for _ in range(a.steps):obs,_,_,_,straight=env.step({'acc':0.,'steer_rate':0.})
        first=obs['rgb']['CAM_FRONT'].copy();Image.fromarray(first).save(out/'straight.png')
        obs,reset=env.reset()
        if reset['steer']!=0 or reset['timestamp']!=0 or reset['ego_pos']!=initial['ego_pos']:
            raise RuntimeError('reset did not restore initial state')
        frames=[Image.fromarray(obs['rgb']['CAM_FRONT'])]
        for i in range(a.steps):
            obs,reward,terminated,truncated,info=env.step({'acc':1.,'steer_rate':.15})
            if not env.observation_space.contains(obs):raise RuntimeError('observation contract failed')
            trace.append(info);frames.append(Image.fromarray(obs['rgb']['CAM_FRONT']))
            print('step',i+1,'time',round(info['timestamp'],3),'pose',info['ego_pos'],'source frame',info['source_frame'],flush=True)
            if terminated or truncated:break
        after=obs['rgb']['CAM_FRONT'];Image.fromarray(after).save(out/'controlled.png')
        difference=np.abs(first.astype(np.int16)-after.astype(np.int16))
        state_delta=float(np.linalg.norm(np.array(straight['ego_pos'])-np.array(info['ego_pos'])))
        if state_delta<1e-5 or env.checkpoint_loads!=1:raise RuntimeError('action/state or resident-load check failed')
        w,h=frames[0].size;canvas=Image.new('RGB',(2*w,h+24),'white')
        canvas.paste(Image.fromarray(first),(0,24));canvas.paste(Image.fromarray(after),(w,24))
        draw=ImageDraw.Draw(canvas);draw.text((5,5),'A: zero controls',fill='black');draw.text((w+5,5),'B: accelerate + steer',fill='black')
        canvas.save(out/'action_comparison.png')
        frames[0].save(out/'controlled.gif',save_all=True,append_images=frames[1:],duration=100,loop=0)
        report=dict(status='resident reset/step state checks passed; visual review required',
            checkpoint_loads=env.checkpoint_loads,steps=len(trace),state_delta_m=state_delta,
            changed_pixels=int((difference.max(2)>5).sum()),elapsed_seconds=time.perf_counter()-started,
            fixed_driving_model_connected=False,collision_and_scoring_validated=False,
            sky_and_affine_enabled=False,camera_refinement=False,source_fps_assumption=10.,trace=trace)
        report['action_visual_response_detected']=report['changed_pixels']>=20
        (out/'live_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));print('OUTPUT:',out)
        if not report['action_visual_response_detected']:raise RuntimeError('insufficient image response to different actions; inspect saved report')
    finally:env.close()


if __name__=='__main__':main()
