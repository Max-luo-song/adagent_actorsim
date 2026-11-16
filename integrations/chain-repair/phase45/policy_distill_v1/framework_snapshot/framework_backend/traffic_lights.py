"""Diagnostic signal-color programs on supported native light Gaussians.

Overrides final emitted RGB only, after learned time/view-dependent color.
No weights, geometry, alpha, background or unselected points are modified.
Chromatic canonical points are emitter CANDIDATES, not certified bulb masks.
"""
import copy
import math
from numbers import Real
import numpy as np

STATES=('red','yellow','green','off','original')
SIGNAL_OPERATIONS={'set_signal':{'state'},'set_signal_program':{'phases','outside_state'},'restore_signal':set()}
COLORS={'red':(1.,.015,.005),'yellow':(1.,.72,.005),'green':(.005,1.,.04),'off':(.01,.01,.01)}


def finite(value):return isinstance(value,Real) and not isinstance(value,(bool,np.bool_)) and math.isfinite(value)


def validate_signal(command,*,duration=None):
    op=command['operation']
    if op not in SIGNAL_OPERATIONS:raise ValueError('unsupported signal operation')
    if op=='set_signal':
        if command['state'] not in STATES:raise ValueError('unknown signal state')
    if op=='set_signal_program':
        phases=command['phases'];outside=command['outside_state']
        if outside not in STATES or not isinstance(phases,list) or not 1<=len(phases)<=128:raise ValueError('invalid signal program')
        previous=0.
        for phase in phases:
            if not isinstance(phase,dict) or set(phase)!={'start_s','end_s','state'}:raise ValueError('invalid phase fields')
            start,end=phase['start_s'],phase['end_s']
            if not finite(start) or not finite(end) or start<0 or start<previous or end<=start:
                raise ValueError('signal phases must be finite, ordered, non-overlapping half-open intervals')
            if phase['state'] not in STATES:raise ValueError('unknown phase state')
            if duration is not None and end>duration+1e-9:raise ValueError('signal phase exceeds scene clock')
            previous=end
    return copy.deepcopy(command)


def state_at(program,timestamp):
    if program['operation']=='set_signal':return program['state']
    for phase in program['phases']:
        if phase['start_s']<=timestamp<phase['end_s']:return phase['state']
    return program['outside_state']


def emitter_candidates(dc):
    rgb=(dc.detach()*.28209479177387814+.5).clamp(0,1)
    return (rgb.amax(-1)-rgb.amin(-1)>.2)&(rgb.amax(-1)>.5)


class TrafficLights:
    def __init__(self,backend,spec):
        self.backend=backend;self.spec=spec;self.torch=backend.torch
        self.model=backend.trainer.models['TrafficLightNodes'];self.programs={};self.timestamp=None;self.last=[];self.geometry={};self.closed=False
        model=self.model;torch=self.torch
        self.point_ids=model.point_ids.flatten().long()
        dc=model._features_dc.detach();rgb=(dc*.28209479177387814+.5).clamp(0,1)
        if rgb.shape!=(len(self.point_ids),3):raise ValueError('traffic light color/identity shape mismatch')
        # Conservative visual candidate selector; grey housing stays unmodified.
        self.emitter=emitter_candidates(dc)
        self.original_get=model.get_gaussians;self.original_render=backend.render
        from calibrated_signals import CalibratedSignals
        self.calibrated=CalibratedSignals(backend,spec,lambda:self.timestamp)
        self.had_model_override='get_gaussians' in model.__dict__;self.had_render_override='render' in backend.__dict__
        def get_gaussians(*args,**kwargs):return self.override(self.original_get(*args,**kwargs))
        def render(pose,timestamp):
            self.timestamp=float(timestamp);self.last=[]
            return self.original_render(pose,timestamp)
        model.get_gaussians=get_gaussians;backend.render=render

    def objects(self):
        rows=[]
        for index in range(self.model.instances_trans.shape[1]):
            count=int((self.point_ids==index).sum());candidates=int(((self.point_ids==index)&self.emitter).sum())
            valid=self.model.instances_fv[:,index].nonzero().flatten().detach().cpu().tolist()
            rows.append(dict(object_id=f'native:TrafficLightNodes:{index}',kind='native_signal',group='TrafficLightNodes',model_index=index,
                point_count=count,emitter_candidate_points=candidates,valid_source_frames=valid,
                operations=list(SIGNAL_OPERATIONS) if candidates else [],
                color_edit_supported=bool(candidates),selection='canonical_dc_chromatic_candidates_v1',
                selection_visually_certified=False,three_bulb_semantics_verified=False))
        return rows+self.calibrated.objects()

    @property
    def has_programs(self):return bool(self.programs or self.calibrated.programs)

    def validate(self,command):
        validate_signal(command,duration=self.spec.frames/self.spec.fps)
        if command['object_id'].startswith('signal:'):
            key=command['object_id'].split(':',1)[1]
            if key not in self.calibrated.profiles:raise ValueError('unknown calibrated signal profile')
            return key
        index=int(command['object_id'].split(':')[-1])
        if not 0<=index<self.model.instances_trans.shape[1]:raise ValueError('traffic light instance does not exist')
        if not bool(((self.point_ids==index)&self.emitter).any()):
            raise ValueError('light has no supported emitter candidate Gaussians; refusing a false edit')
        return index

    def apply(self,command):
        index=self.validate(command)
        programs=self.calibrated.programs if isinstance(index,str) else self.programs
        before=copy.deepcopy(programs)
        try:
            if command['operation']=='restore_signal':programs.pop(index,None)
            else:programs[index]=copy.deepcopy(command)
            self.backend.clear_observation_history()
        except Exception:
            programs.clear();programs.update(before);raise
        return dict(model_index=index,emitter_candidate_points=len(self.calibrated.geometry[index]['means']) if isinstance(index,str) else int(((self.point_ids==index)&self.emitter).sum()),
            geometry_changed=isinstance(index,str),geometry_mode='added compact 3D lens patch' if isinstance(index,str) else 'unchanged native geometry',weights_changed=False,background_changed=False,
            visual_scope='registered photo-calibrated 3D lens surface' if isinstance(index,str) else 'native emitter candidate RGB override; housing geometry and other points unchanged',
            program_active=index in programs,signal_rules_connected=False)

    def override(self,gaussians):
        if self.programs and self.timestamp is None:raise RuntimeError('signal override requires renderer timestamp')
        n=len(self.point_ids);mask=getattr(self.model,'filter_mask',None)
        if mask is None or mask.shape!=(n,) or mask.dtype!=self.torch.bool:
            raise RuntimeError('cannot prove traffic Gaussian output ordering')
        ids=self.point_ids[mask];emitter=self.emitter[mask]
        if gaussians['_rgbs'].shape!=(len(ids),3):raise RuntimeError('traffic Gaussian output alignment mismatch')
        self.geometry={}
        for index in ids[emitter].unique().tolist():
            selected=(ids==index)&emitter&(gaussians['_opacities'].reshape(-1)>1e-6)
            if bool(selected.any()):
                self.geometry[index]=dict(means=gaussians['_means'][selected].detach().cpu().numpy(),
                    scales=gaussians['_scales'][selected].detach().cpu().numpy())
        if not self.programs:return gaussians
        result=dict(gaussians);colors=gaussians['_rgbs'].clone();receipts=[]
        for index,program in sorted(self.programs.items()):
            state=state_at(program,self.timestamp);selected=(ids==index)&emitter
            if state!='original':colors[selected]=colors.new_tensor(COLORS[state])
            active=int((selected & (gaussians['_opacities'].reshape(-1)>1e-6)).sum())
            receipts.append(dict(object_id=f'native:TrafficLightNodes:{index}',requested_state=state,
                valid_gaussian_points=active,camera_visibility_verified=False,semantic_control_connected=False))
        result['_rgbs']=colors;self.last=receipts
        return result

    def snapshot(self,timestamp,cameras=None):
        targets=[]
        geometries=dict(self.geometry)
        for key,geometry in self.calibrated.geometry.items():geometries['signal:'+key]=geometry
        for index,geometry in geometries.items():
            rois={}
            for name,cam in (cameras or {}).items():
                means=geometry['means'];local=np.c_[means,np.ones(len(means))]@np.linalg.inv(np.asarray(cam['camera_to_world'])).T
                good=local[:,2]>1e-3
                if not good.any():continue
                xyz=local[good,:3];k=np.asarray(cam['K']);uv=xyz@k.T;uv=uv[:,:2]/uv[:,2:]
                radius=3*geometry['scales'][good].max(-1)*max(k[0,0],k[1,1])/xyz[:,2]+8
                low=(uv-radius[:,None]).min(0);high=(uv+radius[:,None]).max(0)
                shape=np.asarray(cam['rgb']).shape;h,w=shape[:2]
                box=[int(np.floor(low[0])),int(np.floor(low[1])),int(np.ceil(high[0])),int(np.ceil(high[1]))]
                rois[name]=dict(box_xyxy=box,intersects_image=bool(high[0]>=0 and high[1]>=0 and low[0]<w and low[1]<h))
            identity=index if isinstance(index,str) else f'native:TrafficLightNodes:{index}'
            lenses={}
            if isinstance(index,str):
                for name,cam in (cameras or {}).items():
                    xyz=np.c_[geometry['centers'],np.ones(3)]@np.linalg.inv(np.asarray(cam['camera_to_world'])).T
                    k=np.asarray(cam['K']);uv=xyz[:,:3]@k.T;uv=uv[:,:2]/uv[:,2:]
                    radius=self.calibrated.profiles[index.split(':',1)[1]]['lens_radius_m']
                    lenses[name]=[dict(state=state,pixel=pixel.tolist(),radius_px=float(k[0,0]*radius/point[2])) for state,pixel,point in zip(['red','yellow','green'],uv,xyz)]
            targets.append(dict(object_id=identity,valid_emitter_candidate_points=len(geometry['means']),
                projected_candidate_rois=rois,projected_lenses=lenses,occlusion_visibility_certified=False))
        return dict(projected_targets=targets,programs=[dict(object_id=f'native:TrafficLightNodes:{i}',state=state_at(p,timestamp),program=copy.deepcopy(p))
            for i,p in sorted(self.programs.items())]+[dict(object_id='signal:'+key,state=state_at(program,timestamp),program=copy.deepcopy(program)) for key,program in sorted(self.calibrated.programs.items())],last_render_receipts=copy.deepcopy(self.last)+copy.deepcopy(self.calibrated.receipts),
            semantic_control_connected=False,selection_visually_certified=False)

    def clear(self):self.programs.clear();self.calibrated.clear();self.last=[];self.geometry={};self.timestamp=None

    def close(self):
        if self.closed:return
        self.clear();self.calibrated.close()
        if self.had_model_override:self.model.get_gaussians=self.original_get
        else:delattr(self.model,'get_gaussians')
        if self.had_render_override:self.backend.render=self.original_render
        else:delattr(self.backend,'render')
        self.closed=True
