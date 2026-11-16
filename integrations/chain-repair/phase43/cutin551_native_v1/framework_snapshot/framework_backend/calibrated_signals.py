"""Registered, photo-calibrated 3D lens patches in the shared Gaussian raster.

Original scene parameters remain intact. Three compact opaque lens surfaces
cover the original lit lens when changing state. No screen-space painting.
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

import copy,json
pass  # Path is provided by the portable boundary above
import numpy as np
from scipy.spatial.transform import Rotation


def load_profiles(spec):
    result={}
    for path in (Path(__file__).parent/'signal_profiles').glob('*.json'):
        profile=_hugsim_expand(json.loads(path.read_text()))
        if profile['scene']!=spec.source.get('scene'):continue
        source=spec.source.get('checkpoint')
        if source is None or Path(source).resolve()!=Path(profile['checkpoint']).resolve():continue
        stat=Path(source).stat()
        if stat.st_size!=profile['checkpoint_size'] or stat.st_mtime_ns!=profile['checkpoint_mtime_ns']:
            raise ValueError('calibrated signal checkpoint provenance changed')
        build_geometry(profile)  # validate before installing render hooks
        result[profile['profile_id']]=profile
    return result


def build_geometry(profile):
    if profile['schema']!='calibrated_signal_lenses_v1':raise ValueError('invalid lens profile')
    green=np.asarray(profile['green_center_source'],dtype=float);up=np.asarray(profile['up_source'],dtype=float);normal=np.asarray(profile['normal_towards_viewer_source'],dtype=float)
    if any(v.shape!=(3,) or not np.isfinite(v).all() for v in [green,up,normal]):raise ValueError('invalid lens pose')
    if not np.isclose(np.linalg.norm(up),1) or not np.isclose(np.linalg.norm(normal),1) or abs(up@normal)>1e-6:raise ValueError('lens basis is not orthonormal')
    radius=float(profile['lens_radius_m']);spacing=float(profile['lens_spacing_m']);step=float(profile['gaussian_spacing_m']);sigma=float(profile['gaussian_sigma_m']);alpha=float(profile['gaussian_opacity']);bias=float(profile['surface_front_bias_m'])
    if not np.isfinite([radius,spacing,step,sigma,alpha,bias]).all() or not 0<radius<spacing/2 or not 0<step<=radius or not 0<sigma<radius or not 0<alpha<1 or not 0<=bias<=.25:raise ValueError('invalid lens dimensions')
    right=np.cross(up,normal);r=np.column_stack([right,up,normal]);quat=Rotation.from_matrix(r).as_quat()[[3,0,1,2]]
    centers=np.stack([green+up*spacing*2,green+up*spacing,green])+normal*bias
    xy=np.array([(x,y) for x in np.arange(-radius,radius+step/2,step) for y in np.arange(-radius,radius+step/2,step) if x*x+y*y<=radius*radius])
    means=np.concatenate([center+xy[:,0,None]*right+xy[:,1,None]*up for center in centers])
    slots=np.repeat(np.arange(3),len(xy));n=len(means)
    return dict(means=means,scales=np.tile([sigma,sigma,.004],(n,1)),quats=np.tile(quat,(n,1)),opacities=np.full((n,1),alpha),slots=slots,centers=centers)


class CalibratedSignals:
    def __init__(self,backend,spec,timestamp):
        self.backend=backend;self.trainer=backend.trainer;self.timestamp=timestamp;self.profiles=load_profiles(spec);self.programs={};self.receipts=[]
        self.geometry={key:build_geometry(p) for key,p in self.profiles.items()}
        self.original_collect=self.trainer.collect_gaussians;self.trainer.collect_gaussians=self.collect;self.closed=False

    def objects(self):
        return [dict(object_id='signal:'+key,kind='calibrated_signal_lens_patch',group='CalibratedSignals',profile_id=key,
            point_count=len(self.geometry[key]['means']),emitter_candidate_points=len(self.geometry[key]['means']),
            operations=['set_signal','set_signal_program','restore_signal'],color_edit_supported=True,
            selection='photo-calibrated three-lens Gaussian surface',selection_visually_certified=False,
            three_bulb_semantics_verified=False,red_slot_directly_observed_lit=p.get('red_slot_directly_observed_lit',False)) for key,p in self.profiles.items()]

    def collect(self,*args,**kwargs):
        gs,freeze=self.original_collect(*args,**kwargs);self.receipts=[]
        if not self.programs:return gs,freeze
        from traffic_lights import state_at,COLORS
        cam=kwargs.get('cam',args[0] if args else None);camera_position=cam.camtoworlds[..., :3,3]
        fields=[]
        for key,program in sorted(self.programs.items()):
            state=state_at(program,self.timestamp());profile=self.profiles[key];geometry=self.geometry[key]
            towards=np.asarray(camera_position.detach().cpu())-geometry['centers'][2]
            front=float(np.dot(towards,np.asarray(profile['normal_towards_viewer_source'])))>0
            self.receipts.append(dict(object_id='signal:'+key,requested_state=state,front_facing=front,
                appended_gaussians=len(geometry['means']) if front and state!='original' else 0))
            if state=='original' or not front:continue
            colors=np.full((len(geometry['means']),3),.008)
            if state in ('red','yellow','green'):
                colors[geometry['slots']==('red','yellow','green').index(state)]=COLORS[state]
            def tensor(value):return gs._means.new_tensor(value)
            fields.append(dict(_means=tensor(geometry['means']),_scales=tensor(geometry['scales']),_quats=tensor(geometry['quats']),
                _opacities=tensor(geometry['opacities']),_rgbs=tensor(colors)))
        if not fields:return gs,freeze
        merged=copy.copy(gs);torch=self.backend.torch
        for name in ('_means','_scales','_quats','_opacities','_rgbs'):setattr(merged,name,torch.cat([getattr(gs,name)]+[f[name] for f in fields]))
        n=sum(len(f['_means']) for f in fields);labels=self.trainer.pts_labels
        label=self.trainer.gaussian_classes['TrafficLightNodes']
        self.trainer.pts_labels=torch.cat([labels,labels.new_full((n,),label)])
        if hasattr(self.trainer,'dynamic_pts_mask'):self.trainer.dynamic_pts_mask=self.trainer.pts_labels!=self.trainer.gaussian_classes.get('Background',0)
        return merged,torch.cat([freeze,freeze.new_zeros(n)])

    def clear(self):self.programs.clear();self.receipts=[]
    def close(self):
        if self.closed:return
        self.clear();self.trainer.collect_gaussians=self.original_collect;self.closed=True
