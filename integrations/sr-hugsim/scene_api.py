"""Uniform diagnostic scene interface. NumPy-only contracts; CUDA is lazy.

World/ego axes: first Waymo ego, metres, x forward/y left/z up.
Asset axes remain the calibrated asset's x longitudinal/y down/z lateral.
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
import json
import math
pass  # Path is provided by the portable boundary above
import re
import sys
from uuid import uuid4
import numpy as np
from waymo_adapter import rigid, frame_index

BASE=Path(__file__).resolve().parents[2]
ADAPTER=BASE/'waymo23-native-import'
if str(ADAPTER) not in sys.path: sys.path.insert(0,str(ADAPTER))
from runtime_configuration import preflight_source, load_vehicle_config

CAMERA_ORDER=('CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT')
BACKEND='hugsim_geometry_sr_pbr_v1'


def resolve_file(value, parent):
    path=Path(value)
    if not path.is_absolute(): path=parent/path
    path=path.resolve()
    if not path.is_file(): raise FileNotFoundError(path)
    return path


@dataclass(frozen=True)
class SceneSpec:
    manifest: Path
    source_report: Path
    source: dict
    fps: float
    frames: int
    scale: float
    vehicle: object
    camera_order: tuple

    @classmethod
    def read(cls, path):
        path=Path(path).resolve(); data=_hugsim_expand(json.loads(path.read_text()))
        allowed={'schema','backend','source_report','fps','fps_authority','render_scale','camera_order','vehicle_config'}
        if set(data)-allowed or data.get('schema')!='unified_scene_v1' or data.get('backend')!=BACKEND:
            raise ValueError('Unsupported scene manifest/schema/backend')
        source_path=resolve_file(data['source_report'],path.parent)
        source=_hugsim_expand(json.loads(source_path.read_text()))
        calibration=preflight_source(source)
        count=source.get('frames')
        if isinstance(count,bool) or not isinstance(count,int) or count<1:
            raise ValueError('Invalid scene frame count')
        for key in ('checkpoint','source_config'):
            if not Path(source[key]).is_file(): raise FileNotFoundError(source[key])
        fps=float(data['fps']);scale=float(data['render_scale'])
        if not np.isfinite(fps) or fps<=0 or not np.isfinite(scale) or not 0<scale<=1:
            raise ValueError('Invalid FPS/render scale')
        if not np.isclose(fps,calibration['source_fps']) or data.get('fps_authority')!='source_metadata_assumption':
            raise ValueError('Clock disagrees with source metadata or has an unsupported authority')
        order=tuple(data['camera_order'])
        if len(order)!=3 or set(order)!=set(calibration['cameras']):
            raise ValueError('Camera order must contain each configured camera once')
        vehicle=load_vehicle_config(resolve_file(data['vehicle_config'],path.parent))
        return cls(path,source_path,source,fps,count,scale,vehicle,order)

    def summary(self):
        return dict(scene=self.source['scene'],source_report=str(self.source_report),fps=self.fps,
            frames=self.frames,camera_order=list(self.camera_order),world_frame='first_waymo_ego_z_up_meters',
            timestamp_frequency_independently_validated=False,backend=BACKEND,
            full_sr_trainer=False,training=False,formal_evaluation_admitted=False)


@dataclass(frozen=True)
class EditRequest:
    operation: str
    object_id: str
    asset_to_world: np.ndarray | None=None
    asset: Path | None=None
    calibration: dict | None=None

    @classmethod
    def read(cls, value):
        if not isinstance(value,dict):raise ValueError('Edit request must be a mapping')
        common={'operation','object_id','mode'}
        op=value.get('operation')
        fields={'insert':{'asset','asset_calibration','asset_to_world'},'set_pose':{'asset_to_world'},'remove':set()}
        if op not in fields or set(value)!=(common|fields[op]):
            raise ValueError('Unsupported operation or missing/unknown edit fields')
        if value['mode']!='diagnostic_preview':
            raise ValueError('This interface currently supports diagnostic previews only; N1 is suspended')
        oid=value['object_id']
        if not isinstance(oid,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}',oid):
            raise ValueError('Invalid managed object ID')
        pose=rigid(value['asset_to_world']).copy() if op!='remove' else None
        asset=calibration=None
        if op=='insert':
            from asset_calibration import load
            asset=Path(value['asset']).resolve()
            calibration=load(Path(value['asset_calibration']).resolve(),asset)
        return cls(op,oid,pose,asset,calibration)


def validate_action(action, vehicle, speed, steer):
    if not isinstance(action,dict) or set(action)!={'acc_mps2','steer_rate_radps'}:
        raise ValueError('Expected acc_mps2 and steer_rate_radps')
    acc=float(action['acc_mps2']);rate=float(action['steer_rate_radps'])
    if not np.isfinite([acc,rate]).all() or not vehicle.min_acc_mps2<=acc<=vehicle.max_acc_mps2 or abs(rate)>math.radians(vehicle.max_steer_rate_degrees)+1e-10:
        raise ValueError('Action exceeds the shared vehicle limits')
    if not -1e-10<=speed+acc*vehicle.dt<=vehicle.max_speed_mps+1e-10 or abs(steer+rate*vehicle.dt)>math.radians(vehicle.max_steer_degrees)+1e-10:
        raise ValueError('Action would exceed speed/steering limits')
    return dict(acc_mps2=acc,steer_rate_radps=rate)


class SceneSession:
    """One backend per session; load_scene/reset invalidate consumer history tokens.

    The interface does not instantiate UniAD, certify placement or grant formal
    evaluation admission. Managed edits operate on inserted objects only.
    """
    def __init__(self, backend_factory=None):
        if backend_factory is None:
            from native_scene_backend import NativeSceneBackend
            backend_factory=NativeSceneBackend
        self.factory=backend_factory;self.backend=None;self.spec=None
        self.generation=None;self.revision=0;self.objects={}

    def load_scene(self, manifest):
        spec=SceneSpec.read(manifest)  # CPU preflight must succeed before closing the old scene.
        self.close()
        backend=None
        try:
            backend=self.factory(spec)
        except Exception:
            if backend is not None:backend.close()
            raise
        self.backend=backend;self.spec=spec;self.generation=str(uuid4());self.revision=0;self.objects={}
        return self.describe()

    def _require(self):
        if self.backend is None:raise RuntimeError('Load a scene first')

    def describe(self):
        self._require()
        return dict(self.spec.summary(),history_token=self.generation,edit_revision=self.revision,
            managed_objects=sorted(self.objects),capabilities=['insert_calibrated_asset','set_managed_pose',
            'remove_managed_object','render_at_pose','observe','step','reset','load_scene'],
            native_object_editing=False,perception_connected=False,placement_certified=False)

    def edit(self, request):
        self._require(); edit=EditRequest.read(request)
        if edit.operation=='insert' and edit.object_id in self.objects:raise ValueError('Object ID already exists')
        if edit.operation!='insert' and edit.object_id not in self.objects:raise KeyError(edit.object_id)
        # Backend stages expensive loading/rotation before atomic registry replacement.
        self.backend.apply(edit)
        if edit.operation=='remove':del self.objects[edit.object_id]
        elif edit.operation=='insert':self.objects[edit.object_id]=dict(asset=str(edit.asset),asset_to_world=edit.asset_to_world.tolist())
        else:self.objects[edit.object_id]['asset_to_world']=edit.asset_to_world.tolist()
        self.revision+=1
        return dict(object_id=edit.object_id,operation=edit.operation,edit_revision=self.revision,
                    status='diagnostic_unverified',n1_certified=False,formal_evaluation_admitted=False)

    def _packet(self, raw, timestamp, ego_pose):
        frame=frame_index(timestamp,self.spec.fps,self.spec.frames)
        cameras={}
        for name in self.spec.camera_order:
            view=raw['cameras'][name];rgb=np.asarray(view['rgb']);depth=np.asarray(view['depth'])
            k=np.asarray(view['K'],dtype=float);c2w=rigid(view['camera_to_world'])
            if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or depth.shape!=rgb.shape[:2] or not np.isfinite(depth).all() or np.any(depth<0) or k.shape!=(3,3) or not np.isfinite(k).all() or np.any(np.diag(k)[:2]<=0):
                raise ValueError('Backend returned an invalid RGB/depth/calibration packet: '+name)
            cameras[name]=dict(rgb=rgb,depth=depth.astype(np.float32,copy=False),K=k,
                camera_to_world=c2w,camera_to_ego=np.linalg.inv(ego_pose)@c2w,
                source_camera_id=view['source_camera_id'])
            if 'depth_alpha' in view:
                alpha=np.asarray(view['depth_alpha'],dtype=np.float32)
                if alpha.shape!=depth.shape or not np.isfinite(alpha).all() or np.any(alpha<0) or np.any(alpha>1.00001):raise ValueError('Invalid depth opacity')
                cameras[name]['depth_alpha']=alpha.copy()
                cameras[name]['depth_encoding']=view.get('depth_encoding','unknown')
            if 'depth_variance' in view:
                variance=np.asarray(view['depth_variance'],dtype=np.float32)
                if variance.shape!=depth.shape or not np.isfinite(variance).all() or np.any(variance<0):raise ValueError('Invalid depth variance')
                cameras[name]['depth_variance']=variance.copy()
            if 'surface_depth' in view:
                surface=np.asarray(view['surface_depth'],dtype=np.float32);valid=np.asarray(view['surface_valid'],dtype=bool)
                if surface.shape!=depth.shape or valid.shape!=depth.shape or not np.isfinite(surface).all() or np.any(surface<0):raise ValueError('Invalid front surface sensor')
                cameras[name].update(surface_depth=surface.copy(),surface_valid=valid.copy())
        if set(raw['cameras'])!=set(cameras):raise ValueError('Unexpected backend cameras')
        return dict(schema='unified_observation_v1',scene=self.spec.source['scene'],timestamp_s=float(timestamp),
            source_frame=frame,history_token=self.generation,edit_revision=self.revision,
            model_cache_token=f'{self.generation}:{self.revision}',ego_to_world=rigid(ego_pose).copy(),
            ego_state=dict(raw['ego_state']),camera_order=list(self.spec.camera_order),cameras=cameras,
            managed_objects=_hugsim_expand(json.loads(json.dumps(self.objects))),backend=BACKEND,full_sr_trainer=False,
            edit_trust=dict(edited=bool(self.objects),status='diagnostic_unverified' if self.objects else 'unedited',
                            placement_certified=False,n1_certified=False),
            formal_evaluation_admitted=False,training=False,perception_connected=False,
            timestamp_frequency_independently_validated=False)

    def render(self, ego_to_world, timestamp_s):
        """Render a supplied source-world ego pose/time without advancing simulation."""
        self._require();pose=rigid(ego_to_world).copy()
        frame_index(timestamp_s,self.spec.fps,self.spec.frames)
        return self._packet(self.backend.render(pose,float(timestamp_s)),float(timestamp_s),pose)

    def observe(self):
        self._require();timestamp,pose=self.backend.current_view()
        return self.render(pose,timestamp)

    def step(self, action):
        self._require();state=self.backend.state()
        action=validate_action(action,self.spec.vehicle,state['speed'],state['steer'])
        frame_index(state['timestamp']+self.spec.vehicle.dt,self.spec.fps,self.spec.frames)
        self.backend.advance(action)
        return self.observe()

    def reset(self):
        self._require();self.backend.reset();self.objects={};self.revision=0;self.generation=str(uuid4())
        return self.observe()

    def close(self):
        backend=self.backend
        self.backend=None;self.spec=None;self.generation=None;self.objects={};self.revision=0
        if backend is not None:backend.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
