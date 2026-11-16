"""HUGSIM geometry/rasterization + frozen SR PBR; diagnostic managed objects.

No AST-loaded model code. Full MultiTrainer is not this backend. Native renderer
hooks are scoped to an observation and restored even on failure. One native
session per process is enforced because HUGSIM also has process-global state.
"""
from __future__ import annotations

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

from dataclasses import dataclass
import importlib.util
import inspect
import os
pass  # Path is provided by the portable boundary above
import sys
from threading import RLock
from types import SimpleNamespace
import numpy as np
from waymo_adapter import SR_TO_HUGSIM as T

BASE=Path(__file__).resolve().parents[2]
_LOCK=RLock()
_ACTIVE=None


@dataclass
class Asset:
    model: object
    raw_rest: object
    source_pose: np.ndarray
    native_pose: object
    path: Path


class NativeSceneBackend:
    def __init__(self, spec):
        global _ACTIVE
        self.env=None;self.assets={};self.spec=spec;self.light=None;self.material=None;self.closed=False
        with _LOCK:
            if _ACTIVE is not None:raise RuntimeError('Use one native scene session per process')
            _ACTIVE=self
        old_cwd=Path.cwd()
        try:
            archive=Path(spec.source['inputs'])/'backup/datasets/dataset_meta.py'
            if not archive.is_file():
                raise FileNotFoundError('Missing archived training camera metadata before GPU load: '+str(archive))
            import torch
            self.torch=torch
            if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
            os.environ['BASE']=str(BASE)
            sys.path[:0]=[str(BASE/'code/HUGSIM'),str(BASE/'code/HUGSIM/sim')]
            import waymo552_three
            import gaussian_renderer
            from gsplat import spherical_harmonics
            self.renderer=gaussian_renderer;self.sh=spherical_harmonics
            module_path=BASE/'integrations/sr-origin-main/models/legacy_pbr.py'
            module_spec=importlib.util.spec_from_file_location('unified_frozen_sr_pbr',module_path)
            self.pbr=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(self.pbr)
            self.env=waymo552_three.make_env(scale=spec.scale,source_report=str(spec.source_report),vehicle=spec.vehicle)
            self.env.fps=spec.fps;self.env.capture_edit_evidence=True
            if self.env.frames!=spec.frames:raise ValueError('Checkpoint and metadata frame counts differ')
            state=self.env.models['Background'];count=len(state['_means'])
            for name,shape in self.pbr.MATERIAL_SHAPES.items():
                if name not in state or tuple(state[name].shape)!=(count,)+shape:
                    raise ValueError('Invalid background material: '+name)
            self.material=SimpleNamespace(**state,num_points=count,num_asg_lobes=4)
            light_state=self.env.models['GlobalLighting'];n=len(light_state['appearance_embedding.weight'])
            if n!=spec.frames*len(self.env.cfg['data']['pixel_source']['cameras']):
                raise ValueError('Lighting image index schema differs from metadata')
            self.light=self.pbr.GlobalLightingModule(class_name='GlobalLighting',n=n,device=torch.device('cuda'),
                **self.env.cfg['model']['GlobalLighting']['params']).cuda()
            self.light.load_state_dict(light_state,strict=True);self.light.eval().requires_grad_(False)
            scope=inspect.unwrap(self.env.observe.__func__).__globals__
            def color(received,position,coeff):
                if received is not self.env.models['Background']:raise ValueError('Unexpected material source')
                cam=torch.eye(4,device=position.device);cam[:3,3]=position
                return self.pbr.compute_pbr_color(self.material,SimpleNamespace(camtoworlds=cam),coeff,self.sh)
            def lighting(received,frame,frames,image_id):
                if received is not self.env.models['GlobalLighting']:raise ValueError('Unexpected light source')
                return self.pbr.lighting_for_frame(self.light,frame,image_id,n,'cuda')
            scope.update(pbr_color=color,lighting=lighting)
        except BaseException:
            self.close()
            raise
        finally:
            os.chdir(old_cwd)

    def _sync_edit_metadata(self):
        self.env._unverified_scene_edits=[dict(kind='insert',object_id=oid,
            status='diagnostic_preview_only',asset=str(asset.path),
            asset_to_source=asset.source_pose.tolist()) for oid,asset in self.assets.items()]

    def _posed(self, model, raw_rest, pose, path):
        from asset_radiance import sh_rotation
        native=T@pose
        matrix=sh_rotation(native[:3,:3],int(model.active_sh_degree))
        rest=self.torch.einsum('ij,njc->nic',raw_rest.new_tensor(matrix[1:,1:]),raw_rest).contiguous()
        native_tensor=model._xyz.new_tensor(native)
        model._features_rest=rest
        return Asset(model,raw_rest,pose.copy(),native_tensor,path)

    def apply(self, edit):
        with _LOCK,self.torch.no_grad():
            if edit.operation=='remove':
                del self.assets[edit.object_id];self._sync_edit_metadata();return
            if edit.operation=='set_pose':
                old=self.assets[edit.object_id]
                value=self._posed(old.model,old.raw_rest,edit.asset_to_world,old.path)
            else:
                from scene.obj_model import ObjModel
                params,_=self.torch.load(edit.asset/'gs.pth',map_location='cuda',weights_only=False)
                if int(params[0])!=3:raise ValueError('Managed library assets currently require SH degree 3')
                model=ObjModel(int(params[0]),feat_mutable=False);model.restore(params,None)
                model._xyz=model._xyz.clone()
                center=edit.calibration['support']['raw_recenter_xz']
                model._xyz[:,0]-=center[0];model._xyz[:,2]-=center[1]
                raw_rest=model._features_rest.detach().clone()
                value=self._posed(model,raw_rest,edit.asset_to_world,edit.asset)
            self.assets[edit.object_id]=value
            self._sync_edit_metadata()

    def _render_hook(self, original):
        def render(*args,**kwargs):
            if not self.assets:return original(*args,**kwargs)
            camera,pc=kwargs['viewpoint'],kwargs['pc']
            previous=pc.active_sh_degree,pc._features_rest,camera.dynamics
            models=dict(kwargs.get('dynamic_gaussians',{}));poses=dict(camera.dynamics)
            for oid,asset in self.assets.items():
                key='unified_preview_'+oid
                if key in models or key in poses:raise ValueError('Managed object ID collision')
                models[key]=asset.model;poses[key]=asset.native_pose
            try:
                pc.active_sh_degree=3
                pc._features_rest=pc._features_dc.new_zeros((len(pc._xyz),15,3))
                camera.dynamics=poses;kwargs['dynamic_gaussians']=models
                return original(*args,**kwargs)
            finally:
                pc.active_sh_degree,pc._features_rest,camera.dynamics=previous
        return render

    def render(self, pose, timestamp):
        with _LOCK,self.torch.no_grad():
            old_pose_fn=self.env.ego_pose;old_timestamp=self.env.motion.timestamp
            native=T@pose@T.T
            matches=np.allclose(old_pose_fn(),native) and np.isclose(old_timestamp,timestamp)
            original=self.renderer.render
            try:
                self.env.ego_pose=lambda:native.copy();self.env.motion.timestamp=timestamp
                self.renderer.render=self._render_hook(original)
                self.env._edit_validation_views={}
                obs=self.env.observe()
                cameras={}
                for cid,calibration in self.env.cameras.items():
                    name=self.env.cam_names[cid];view=self.env._edit_validation_views[name]
                    from depth_contract import expected_depth,depth_variance,EXPECTED
                    sensor_depth=expected_depth(obs['depth'][name],view['alpha'],view['depth_encoding'])
                    cameras[name]=dict(rgb=obs['rgb'][name],depth=sensor_depth,depth_alpha=view['alpha'],depth_encoding=EXPECTED,surface_depth=view['surface_depth'],surface_valid=view['surface_valid'],depth_variance=depth_variance(view['depth_second_moment'],view['alpha'],sensor_depth),K=view['K'],
                        camera_to_world=view['camera_to_world'],source_camera_id=cid)
                return dict(cameras=cameras,ego_state=dict(speed_mps=self.env.motion.speed,
                    steering_rad=self.env.motion.steer,simulation_state_matches_view=bool(matches)))
            finally:
                self.renderer.render=original;self.env.ego_pose=old_pose_fn;self.env.motion.timestamp=old_timestamp

    def state(self):return self.env.motion.state()

    def current_view(self):return self.env.motion.timestamp,T.T@self.env.ego_pose()@T

    def advance(self, action):
        self.env.motion.advance(dict(acc=action['acc_mps2'],steer_rate=action['steer_rate_radps']))

    def reset(self):
        self.assets.clear();self._sync_edit_metadata();self.env.motion.reset();self.env.cached_frame=None;self.env.frame_cache=None

    def close(self):
        global _ACTIVE
        with _LOCK:
            if self.closed:return
            self.closed=True
            try:
                self.assets.clear();self.light=None;self.material=None
                if self.env is not None:self.env.close()
            finally:
                self.env=None
                if _ACTIVE is self:_ACTIVE=None
