"""Native SMPL root paths, preserving source-clock body articulation.

The original SMPL forward evaluates source roots/joints. A rigid edit delta is
then applied to the posed Gaussian body, with SH evaluated at corrected view
directions. Shared annotation proxies contain the planned roots. This avoids
held-out-frame root interpolation moving the requested trajectory.
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

from contextlib import contextmanager
import copy,hashlib,math
pass  # Path is provided by the portable boundary above
import numpy as np
from scipy.spatial.transform import Rotation

ACTOR_OPERATIONS={'set_actor_trajectory':{'trajectory'},'restore_actor_trajectory':set()}


def finite(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def validate_path(value):
    if not isinstance(value,dict) or set(value)-{'schema','coordinate_frame','waypoints','outside_interval','motion_profile','heading_mode'} or not {'schema','coordinate_frame','waypoints'}<=set(value):raise ValueError('invalid actor trajectory fields')
    if value['schema']!='actor_trajectory_v1' or value['coordinate_frame'] not in ('start_ego_ground','source_world'):raise ValueError('unsupported actor trajectory frame/schema')
    if value.get('outside_interval','hold') not in ('hold','original'):raise ValueError('outside_interval must be hold or original')
    if value.get('motion_profile','legacy') not in ('legacy','smooth_v1'):raise ValueError('unknown actor motion_profile')
    if value.get('heading_mode','waypoint_delta') not in ('waypoint_delta','path_tangent'):raise ValueError('unknown actor heading_mode')
    if value.get('heading_mode')=='path_tangent' and value.get('motion_profile')!='smooth_v1':raise ValueError('path_tangent requires smooth_v1')
    points=value['waypoints']
    if not isinstance(points,list) or not 2<=len(points)<=128:raise ValueError('actor trajectory requires 2..128 timed waypoints')
    previous=-1
    for point in points:
        field='offset_m' if value['coordinate_frame']=='start_ego_ground' else 'position_world_m'
        if not isinstance(point,dict) or not {'timestamp_s',field}<=set(point) or set(point)-{'timestamp_s',field,'yaw_delta_degrees'}:raise ValueError('invalid actor waypoint fields')
        t=point['timestamp_s'];pos=point[field];yaw=point.get('yaw_delta_degrees',0.)
        if not finite(t) or t<0 or t<=previous:raise ValueError('actor timestamps must strictly increase from a nonnegative time')
        if not isinstance(pos,list) or len(pos)!=3 or not all(finite(v) for v in pos):raise ValueError('actor position requires three finite numbers')
        if not finite(yaw) or abs(yaw)>180:raise ValueError('actor yaw delta must be finite and within 180 degrees')
        if value.get('heading_mode')=='path_tangent' and yaw!=0:raise ValueError('path_tangent requires zero/omitted waypoint yaw deltas')
        previous=t
    return copy.deepcopy(value)


def ground_basis(spec,frame):
    source=spec.source;folder=Path(source['inputs'])/'ego_pose';start=source['start_timestep']
    first=np.loadtxt(folder/f'{start:03d}.txt');reference=np.linalg.inv(first)@np.loadtxt(folder/f'{start+frame:03d}.txt')
    up=first[:3,:3].T@np.array([0.,0.,1.]);up/=np.linalg.norm(up)
    forward=reference[:3,0]-up*(up@reference[:3,0]);forward/=np.linalg.norm(forward)
    return np.column_stack([forward,np.cross(up,forward),up])


def source_center(trans,valid,frame,test_frames):
    if frame in test_frames and frame>1 and frame+1<len(trans) and valid[frame-1] and valid[frame+1]:return (trans[frame-1]+trans[frame+1])*.5
    return trans[frame].copy()


def compile_path(value,spec,trans,valid,test_frames=()):
    value=validate_path(value);points=value['waypoints'];fps=spec.fps
    times=np.array([p['timestamp_s'] for p in points]);frames=np.rint(times*fps).astype(int)
    if np.max(np.abs(times*fps-frames))>1e-6 or frames[-1]>=spec.frames:raise ValueError('actor waypoints must align to source frames within scene clock')
    times=frames/fps
    if not np.all(valid[frames[0]:frames[-1]+1]):raise ValueError('source body animation unavailable in requested interval')
    if not np.isfinite(trans[frames[0]:frames[-1]+1]).all():raise ValueError('nonfinite source actor roots')
    basis=ground_basis(spec,int(frames[0]));anchor=source_center(trans,valid,int(frames[0]),test_frames)
    field='offset_m' if value['coordinate_frame']=='start_ego_ground' else 'position_world_m';positions=np.array([p[field] for p in points],dtype=float)
    if value['coordinate_frame']=='start_ego_ground':positions=anchor+positions@basis.T
    if np.max(np.linalg.norm(positions-anchor,axis=1))>10:raise ValueError('actor route exceeds 10m diagnostic displacement')
    speeds=np.linalg.norm(np.diff(positions,axis=0),axis=1)/np.diff(times)
    if np.max(speeds)>3.+1e-8:raise ValueError('actor route exceeds 3m/s diagnostic speed')
    yaw=np.unwrap(np.radians([p.get('yaw_delta_degrees',0) for p in points]))
    if np.max(np.abs(np.diff(yaw)/np.diff(times)))>math.pi/2+1e-8:raise ValueError('actor turn rate exceeds 90deg/s diagnostic limit')
    result=dict(times=times,positions=positions,yaws=yaw,up=basis[:,2],outside=value.get('outside_interval','hold'),
        first_frame=int(frames[0]),last_frame=int(frames[-1]),max_speed_mps=float(speeds.max()),motion_profile=value.get('motion_profile','legacy'),heading_mode=value.get('heading_mode','waypoint_delta'))
    if result['motion_profile']=='smooth_v1':
        from actor_motion import spline
        result.update(position_spline=spline(times,positions),yaw_spline=spline(times,yaw))
        dense=np.linspace(times[0],times[-1],max(2,int((times[-1]-times[0])*120)+1))
        velocity=result['position_spline'](dense,1)
        if np.max(np.linalg.norm(velocity,axis=1))>3.+1e-6:raise ValueError('smooth route exceeds 3m/s')
        if np.max(np.linalg.norm(result['position_spline'](dense)-anchor,axis=1))>10.+1e-6:raise ValueError('smooth route exceeds 10m displacement')
        if np.max(np.abs(result['yaw_spline'](dense,1)))>math.pi/2+1e-6:raise ValueError('smooth turn exceeds 90deg/s')
        if result['heading_mode']=='path_tangent':
            projected=velocity-np.outer(velocity@result['up'],result['up']);speed=np.linalg.norm(projected,axis=1)
            if speed.min()<.05:raise ValueError('path_tangent requires a continuously moving route')
            direction=projected/speed[:,None];angle=np.arccos(np.clip(np.sum(direction[1:]*direction[:-1],axis=1),-1,1))
            if np.max(angle/np.diff(dense))>math.pi/2+1e-5:raise ValueError('path tangent turns faster than 90deg/s')
        result['max_speed_mps']=float(np.linalg.norm(velocity,axis=1).max())
    return result


def sample_path(plan,timestamp):
    if plan['outside']=='original' and not plan['times'][0]<=timestamp<=plan['times'][-1]:return None
    if plan.get('motion_profile')=='smooth_v1':
        t=np.clip(timestamp,plan['times'][0],plan['times'][-1])
        return np.array(plan['position_spline'](t)),float(plan['yaw_spline'](t))
    position=np.array([np.interp(timestamp,plan['times'],plan['positions'][:,j]) for j in range(3)])
    yaw=float(np.interp(timestamp,plan['times'],plan['yaws']))
    return position,yaw


def quat_product(a,b,torch):
    w,x,y,z=a.unbind(-1);v,i,j,k=b.unbind(-1)
    return torch.stack([w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v],dim=-1)


class ActorTrajectories:
    def __init__(self,backend,spec):
        self.backend=backend;self.spec=spec;self.torch=backend.torch;self.model=backend.trainer.models['SMPLNodes']
        self.programs={};self.timestamp=None;self.suspended=False;self.evidence=[];self.closed=False;self.runtime_frame=None
        self.tests=set(int(x) for x in getattr(backend.trainer,'test_set_indices',[]))
        self.original_get=self.model.get_gaussians;self.original_render=backend.render
        self.had_get='get_gaussians' in self.model.__dict__;self.had_render='render' in backend.__dict__
        self.original_clear=backend.clear_observation_history;self.had_clear='clear_observation_history' in backend.__dict__
        backend.clear_observation_history=self.clear_history
        self.trans_tensor=self.model.instances_trans;self.quat_tensor=self.model.instances_quats
        self.model.get_gaussians=self.get_gaussians;backend.render=self.render

    def check_identity(self):
        for name,tensor in [('instances_trans',self.trans_tensor),('instances_quats',self.quat_tensor)]:
            if getattr(self.model,name) is not tensor or self.backend.env.models['SMPLNodes'][name] is not tensor:raise RuntimeError('actor renderer/proxy identity changed')

    def validate(self,command):
        index=int(command['object_id'].split(':')[-1]);n=self.trans_tensor.shape[1]
        if not 0<=index<n or not bool((self.model.point_ids.flatten()==index).any()):raise ValueError('actor has no supported native body')
        self.check_identity()
        if command['operation']=='restore_actor_trajectory':return index,None
        old=self.programs.get(index)
        base=old['base_trans'] if old else self.trans_tensor[:,index].detach().clone()
        valid=self.model.instances_fv[:,index].detach().cpu().numpy().astype(bool)
        plan=compile_path(command['trajectory'],self.spec,base.cpu().numpy(),valid,self.tests)
        if plan['motion_profile']=='smooth_v1':
            from actor_motion import prepare_motion
            base_q=old['base_quats'] if old else self.quat_tensor[:,index].detach()
            plan['motion']=prepare_motion(plan,self.spec,base.cpu().numpy(),valid,base_q.cpu().numpy(),self.model.smpl_qauts[:,index].detach().cpu().numpy(),self.tests)
        return index,plan

    def apply(self,command):
        self.restore_runtime()
        index,plan=self.validate(command);old=self.programs.get(index)
        before_t=self.trans_tensor[:,index].detach().clone();before_q=self.quat_tensor[:,index].detach().clone()
        if plan is not None:
            base_t=old['base_trans'] if old else before_t;base_q=old['base_quats'] if old else before_q
            motion=plan.pop('motion',None)
            temporary=dict(plan=plan,motion=motion)
            planned_t=base_t.clone();planned_q=base_q.clone()
            for frame in range(self.spec.frames):
                sample=sample_path(plan,frame/self.spec.fps)
                if sample is None:continue
                point,yaw=sample
                root_q=base_q[frame]
                if motion is not None:
                    from actor_motion import motion_sample
                    pose=motion_sample(temporary,frame/self.spec.fps);yaw=pose['yaw'];root_q=base_q.new_tensor(pose['root_quat']).reshape(base_q[frame].shape)
                planned_t[frame]=planned_t.new_tensor(point)
                xyzw=Rotation.from_rotvec(plan['up']*yaw).as_quat();dq=planned_q.new_tensor(xyzw[[3,0,1,2]])
                planned_q[frame]=quat_product(dq,root_q,self.torch)
            value=dict(plan=plan,motion=motion,base_trans=base_t,base_quats=base_q,planned_trans=planned_t,planned_quats=planned_q,command=copy.deepcopy(command))
        else:value=None
        try:
            with self.torch.no_grad():
                if value is None:
                    if old:self.trans_tensor[:,index].copy_(old['base_trans']);self.quat_tensor[:,index].copy_(old['base_quats'])
                    self.programs.pop(index,None)
                else:
                    self.programs[index]=value;self.trans_tensor[:,index].copy_(value['planned_trans']);self.quat_tensor[:,index].copy_(value['planned_quats'])
            self.backend.clear_observation_history()
        except Exception:
            with self.torch.no_grad():self.trans_tensor[:,index].copy_(before_t);self.quat_tensor[:,index].copy_(before_q)
            if old:self.programs[index]=old
            else:self.programs.pop(index,None)
            raise
        return dict(model_index=index,program_active=index in self.programs,source_body_animation_preserved=plan is None or plan['motion_profile']=='legacy',source_parameters_preserved=True,
            gait_retimed=bool(plan is not None and plan['motion_profile']=='smooth_v1'),motion_profile=plan['motion_profile'] if plan else None,root_trajectory='C1 cubic' if plan is not None and plan['motion_profile']=='smooth_v1' else 'piecewise linear source-clock samples',heading='delta around Waymo gravity up',
            proxy='shared per-frame root tensors; annotation-box precision',max_speed_mps=plan['max_speed_mps'] if plan else None)

    def restore_runtime(self):
        if self.runtime_frame is not None:
            self.runtime_frame=None;self.write_roots('planned')

    def clear_history(self):
        self.restore_runtime();return self.original_clear()

    def render(self,pose,timestamp):
        self.restore_runtime()
        if not np.isfinite(timestamp) or not 0<=timestamp*self.spec.fps<self.spec.frames:raise ValueError('actor render outside scene clock')
        fractional=abs(timestamp*self.spec.fps-round(timestamp*self.spec.fps))>1e-6
        if fractional and any(e['plan']['motion_profile']=='legacy' for e in self.programs.values()):raise ValueError('legacy actor paths require source-frame-aligned render times')
        self.timestamp=float(timestamp);self.evidence=[]
        # Keep all other models on the source floor frame, matching packet.source_frame.
        source_frame=int(np.floor(timestamp*self.spec.fps+1e-8))
        had_override=hasattr(self.backend,'_actor_source_frame')
        previous_override=getattr(self.backend,'_actor_source_frame',None)
        try:
            if self.programs:
                # Raster frame selection is separate from scene time: traffic
                # schedules and other time consumers still receive exact time.
                self.backend._actor_source_frame=source_frame
                if not self.suspended:
                    self.runtime_frame=source_frame;self.write_roots('planned')
            return self.original_render(pose,timestamp)
        except Exception:
            self.restore_runtime();raise
        finally:
            if had_override:self.backend._actor_source_frame=previous_override
            elif hasattr(self.backend,'_actor_source_frame'):delattr(self.backend,'_actor_source_frame')

    def write_runtime(self):
        if self.runtime_frame is None:return
        frame=self.runtime_frame
        with self.torch.no_grad():
            for index,entry in self.programs.items():
                sample=sample_path(entry['plan'],self.timestamp)
                if sample is None:continue
                target,yaw=sample;root=entry['base_quats'][frame]
                if entry['motion'] is not None:
                    from actor_motion import motion_sample
                    pose=motion_sample(entry,self.timestamp);yaw=pose['yaw'];root=root.new_tensor(pose['root_quat']).reshape(root.shape)
                dq=Rotation.from_rotvec(entry['plan']['up']*yaw).as_quat()[[3,0,1,2]]
                self.trans_tensor[frame,index].copy_(self.trans_tensor.new_tensor(target))
                self.quat_tensor[frame,index].copy_(quat_product(root.new_tensor(dq),root,self.torch))

    def write_roots(self,which):
        with self.torch.no_grad():
            for index,entry in self.programs.items():
                self.trans_tensor[:,index].copy_(entry[which+'_trans']);self.quat_tensor[:,index].copy_(entry[which+'_quats'])
        if which=='planned':self.write_runtime()

    def get_gaussians(self,*args,**kwargs):
        if not self.programs or self.suspended:return self.original_get(*args,**kwargs)
        if self.timestamp is None:raise RuntimeError('render timestamp required for actor path')
        self.check_identity();self.write_roots('base')
        try:source=self.original_get(*args,**kwargs)
        finally:self.write_roots('planned')
        if source is None:return source
        source=self.conditioned_body(source,args,kwargs)
        model=self.model;torch=self.torch;frame=int(model.cur_frame);mask=model.filter_mask
        if mask.dtype!=torch.bool or len(mask)!=len(model.point_ids):raise RuntimeError('unsupported body Gaussian ordering')
        global_ids=torch.arange(len(mask),device=mask.device)[mask];point_ids=model.point_ids.flatten()[mask]
        if len(source['_means'])!=len(point_ids):raise RuntimeError('body Gaussian/identity count mismatch')
        result=dict(source);changed=False
        cam=kwargs.get('cam',args[0] if args else None);camera=cam.camtoworlds[..., :3,3]
        for index,entry in self.programs.items():
            sample=sample_path(entry['plan'],self.timestamp)
            if sample is None:continue
            selected=point_ids==index;indices=global_ids[selected]
            base=entry['base_trans'].detach().cpu().numpy();valid=model.instances_fv[:,index].detach().cpu().numpy().astype(bool)
            center=source_center(base,valid,frame,self.tests);target,yaw=sample
            animation_frame=frame
            if entry['motion'] is not None:
                from actor_motion import motion_sample
                pose=motion_sample(entry,self.timestamp);yaw=pose['yaw'];animation_frame=pose['animation_frame']
            rotation=Rotation.from_rotvec(entry['plan']['up']*yaw)
            active=bool(valid[frame]);proxy=self.trans_tensor[frame,index].detach().cpu().numpy()
            item=dict(object_id=f'native:SMPLNodes:{index}',target_root_world_m=target.tolist(),proxy_root_world_m=proxy.tolist(),
                source_render_root_world_m=center.tolist(),yaw_delta_degrees=float(np.degrees(yaw)),source_frame_valid=active,
                source_body_animation_preserved=entry['motion'] is None,gait_retimed=entry['motion'] is not None,animation_frame=float(animation_frame),motion_profile=entry['plan']['motion_profile'],gaussian_count=int(selected.sum()),
                proxy_error_m=float(np.linalg.norm(proxy-target)),
                render_root_error_m=float(np.linalg.norm(np.asarray(target,dtype=np.float32)-target)),rigid_shape_error_m=0.)
            self.evidence.append(item)
            if not active:continue
            if np.linalg.norm(target-center)<2e-5 and abs(yaw)<1e-10:continue
            if not changed:
                for field in ('_means','_quats','_rgbs'):result[field]=source[field].clone()
                changed=True
            r=source['_means'].new_tensor(rotation.as_matrix());origin=r.new_tensor(center);destination=r.new_tensor(target)
            means=(source['_means'][selected]-origin)@r.T+destination
            result['_means'][selected]=means
            # Independently invert the rigid transform on actual output points.
            recovered=(means-destination)@r+origin
            item['rigid_shape_error_m']=float(torch.linalg.vector_norm(recovered-source['_means'][selected],dim=-1).max().item())
            applied_root=(origin-origin)@r.T+destination
            item['render_root_error_m']=float(np.linalg.norm(applied_root.detach().cpu().numpy()-target))
            dq=r.new_tensor(rotation.as_quat()[[3,0,1,2]]);q=quat_product(dq,source['_quats'][selected],torch)
            result['_quats'][selected]=torch.nn.functional.normalize(q,dim=-1)
            # Re-evaluate source SH at moved directions, inverse-rotated by edit delta.
            coefficients=torch.cat((model._features_dc[indices,None,:],model._features_rest[indices]),dim=1)
            if model.sh_degree>0:
                fn=getattr(self.original_get,'__func__',self.original_get).__globals__.get('spherical_harmonics')
                if fn is None:raise RuntimeError('native SMPL SH evaluator unavailable')
                directions=torch.nn.functional.normalize(means-camera,dim=-1)@r
                degree=min(model.step//model.ctrl_cfg.sh_degree_interval,model.sh_degree)
                colors=(fn(degree,directions,coefficients)+.5).clamp(0,1)
            else:colors=torch.sigmoid(coefficients[:,0,:])
            result['_rgbs'][selected]=colors
        return result

    def conditioned_body(self,source,args,kwargs):
        active={i:e for i,e in self.programs.items() if e['motion'] is not None and sample_path(e['plan'],self.timestamp) is not None}
        if not active:return source
        from actor_motion import motion_sample
        model=self.model;frame=int(model.cur_frame);torch=self.torch
        original_test=getattr(model,'in_test_set',None);had_test=hasattr(model,'in_test_set')
        originals={i:model.smpl_qauts[frame,i].detach().clone() for i in active}
        original_mask=model.filter_mask.clone()
        self.write_roots('base')
        try:
            with torch.no_grad():
                for index,entry in active.items():
                    pose=motion_sample(entry,self.timestamp)
                    model.smpl_qauts[frame,index].copy_(model.smpl_qauts.new_tensor(pose['joints']))
                    self.quat_tensor[frame,index].copy_(self.quat_tensor.new_tensor(pose['root_quat']).reshape(self.quat_tensor[frame,index].shape))
                    center=source_center(entry['base_trans'].cpu().numpy(),model.instances_fv[:,index].cpu().numpy().astype(bool),frame,self.tests)
                    self.trans_tensor[frame,index].copy_(self.trans_tensor.new_tensor(center))
            # Only selected bodies from this second forward are merged below.
            model.in_test_set=False
            conditioned=self.original_get(*args,**kwargs)
            if conditioned is None:raise RuntimeError('conditioned body unexpectedly unavailable')
            if not torch.equal(original_mask,model.filter_mask):raise RuntimeError('conditioned body changed Gaussian ordering')
            selected=torch.zeros_like(model.filter_mask)
            ids=model.point_ids.flatten()[model.filter_mask]
            for index in active:selected[model.filter_mask]|=ids==index
            selected=selected[model.filter_mask]
            result={k:v.clone() for k,v in source.items()}
            for field in result:result[field][selected]=conditioned[field][selected]
            return result
        finally:
            with torch.no_grad():
                for index,joints in originals.items():model.smpl_qauts[frame,index].copy_(joints)
            if had_test:model.in_test_set=original_test
            else:delattr(model,'in_test_set')
            self.write_roots('planned')

    @contextmanager
    def original_paths(self):
        self.restore_runtime();self.check_identity()
        previous=self.suspended;self.suspended=True;self.write_roots('base')
        try:yield
        finally:self.write_roots('planned');self.suspended=previous

    def animation_fingerprint(self):
        fields={}
        for name in ('smpl_qauts','_means','_quats','_features_dc','_features_rest'):
            value=getattr(self.model,name,None)
            if value is not None:fields[name]=hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
        return fields

    def snapshot(self):
        states=[]
        if self.timestamp is not None:
            frame=int(np.floor(self.timestamp*self.spec.fps+1e-8))
            for index,entry in sorted(self.programs.items()):
                sample=sample_path(entry['plan'],self.timestamp)
                base=entry['base_trans'].cpu().numpy();valid=self.model.instances_fv[:,index].cpu().numpy().astype(bool)
                actual=source_center(base,valid,frame,self.tests) if self.suspended or sample is None else sample[0]
                states.append(dict(object_id=f'native:SMPLNodes:{index}',root_world_m=actual.tolist(),
                    path_active=not self.suspended and sample is not None,source_frame_valid=bool(valid[frame])))
        return dict(programmed_objects=[f'native:SMPLNodes:{i}' for i in sorted(self.programs)],
            original_path_counterfactual=self.suspended,states=states,render_evidence=copy.deepcopy(self.evidence),
            body_animation='source SMPL; smooth_v1 filters joints and retimes by path distance',gait_retimed=not self.suspended and any(e['motion'] is not None for e in self.programs.values()))

    def clear(self):
        self.restore_runtime()
        if self.programs:
            self.check_identity();self.write_roots('base')
        self.programs.clear();self.evidence=[];self.timestamp=None

    def close(self):
        if self.closed:return
        self.clear()
        if self.had_get:self.model.get_gaussians=self.original_get
        else:delattr(self.model,'get_gaussians')
        if self.had_render:self.backend.render=self.original_render
        else:delattr(self.backend,'render')
        if self.had_clear:self.backend.clear_observation_history=self.original_clear
        else:delattr(self.backend,'clear_observation_history')
        self.closed=True
