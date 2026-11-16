"""Native environment collision kernels and scripted actor binding; no controller policy."""
import ast,contextlib,math
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from shapely.geometry import Polygon as ShapelyPolygon
from common import ROOT,BUNDLE,SOURCE,SCORER,configure


def functions(path,names,scope):
    tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise ValueError('Native kernel names changed')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope);return scope


def native_kernels():
    import torch
    scope=dict(np=np,torch=torch,math=math,ShapelyPolygon=ShapelyPolygon)
    functions(SCORER,['create_rectangle','bg_collision_det'],scope)
    functions(ROOT/'code/HUGSIM/sim/hugsim_env/envs/hug_sim.py',['fg_collision_det'],scope)
    return scope


def actor_arrays(session):
    """Export effective rendered poses on the source clock, including heldout interpolation."""
    from native_car_edit import effective_source
    result={}
    for name,state in session.backend.env.models.items():
        if name not in ('RigidNodes','DeformableNodes','SMPLNodes'):continue
        values={k:v.detach().cpu().numpy().copy() for k,v in state.items() if k in ('instances_trans','instances_quats','instances_fv','instances_size')}
        if set(values)!=set(('instances_trans','instances_quats','instances_fv','instances_size')):continue
        q=values['instances_quats']
        if q.ndim==4:q=q[:,:,0,:]
        t=values['instances_trans'];valid=values['instances_fv']
        for index in range(t.shape[1]):
            original_t=t[:,index].copy();original_q=q[:,index].copy()
            active=name=='RigidNodes' and index in session.vehicles.programs
            for frame in session.vehicles.tests:
                if active and frame>=session.vehicles.activation_frame:continue
                t[frame,index],q[frame,index]=effective_source(original_t,original_q,valid[:,index],frame,session.vehicles.tests)
        values['instances_quats']=q
        for k,v in values.items():result[name+'__'+k]=v
    return result


def install_trajectory(session,track_path,start_s):
    """Reuse original native program's shared tensors/SH context; activate after common prefix."""
    import torch
    from native_car_edit import VehiclePrograms
    from native_radiance import radiance_state
    class Programs(VehiclePrograms):
        @contextlib.contextmanager
        def render_context(self,frame):
            if frame<self.activation_frame:yield
            else:
                with super().render_context(frame):yield
    previous=session.vehicles
    session.editor.require_current()
    if session.editor.started or session.backend._last is not None:raise RuntimeError('Trajectory binding requires a fresh pre-run scene')
    owner=Programs(session.backend,session.spec);owner.activation_frame=int(round(start_s*session.spec.fps))
    # The no-edit owner is also given activation_frame for canonical exports.
    with np.load(track_path) as z:track={k:z[k].copy() for k in z.files}
    index=3;m=owner.model;base_t=m.instances_trans[:,index].detach().clone();base_q=m.instances_quats[:,index].detach().clone()
    np.testing.assert_array_equal(base_t.cpu(),track['original_trans']);np.testing.assert_array_equal(base_q.cpu(),track['original_quats'])
    trans=base_t.new_tensor(track['edited_trans']);quats=base_q.new_tensor(track['edited_quats'])
    prefix=slice(0,owner.activation_frame+1)
    if not torch.equal(base_t[prefix],trans[prefix]) or not torch.equal(base_q[prefix],quats[prefix]):raise ValueError('Actor intervention changes the common prefix')
    owner.programs[index]=dict(base_t=base_t,base_q=base_q,t=trans,q=quats,appearance=radiance_state(m,index),audit=dict(scripted_intervention=True))
    with torch.no_grad():m.instances_trans[:,index].copy_(trans);m.instances_quats[:,index].copy_(quats)
    if session.backend.env.models['RigidNodes']['instances_trans'] is not m.instances_trans:raise RuntimeError('Proxy and renderer tensors split')
    session.backend.clear_observation_history();session.vehicles=owner;session.backend.vehicle_programs=owner
    old_revision=session.revision;session.revision+=1
    receipt=dict(schema='scripted_native_actor_intervention_v1',object_id='native:RigidNodes:3',model_index=3,point_count_unchanged=True,
        previous_revision=old_revision,edit_revision=session.revision,
        renderer_state_collision_shared_tensor=True,activation_frame=owner.activation_frame,
        prefix_tensor_values_unchanged=True,source_visibility_and_sizes_unchanged=True,
        heldout_prefix_uses_original_renderer=True,sh_rotation_context_reused=True)
    session.editor.journal.append(dict(receipt));return receipt


class Evaluation:
    def __init__(self,tracks,vehicle,background_opacity,device='cuda'):
        import torch
        self.tracks=tracks;self.vehicle=vehicle;self.kernels=native_kernels();self.device=device
        background=np.load(BUNDLE/'background_sr_world.npy')
        with np.load(BUNDLE/'point_semantics/background_semantics.npz') as z:labels=z['class_id']
        # Keep the native >1 and !=10 predicate; unknown semantics remain disclosed.
        opacity=np.asarray(background_opacity).reshape(-1)
        if len(opacity)!=len(background) or not np.isfinite(opacity).all():raise ValueError('Native background opacity/geometry mismatch')
        opaque=opacity>.8
        points=background[(labels>1)&(labels!=10)&opaque]
        self.points=torch.as_tensor(points,dtype=torch.float32,device=device)
        self.known_points=torch.as_tensor(background[(labels>1)&(labels!=10)&(labels!=255)&opaque],dtype=torch.float32,device=device)
        self.unknown_background_points=int(np.sum(labels==255))
        route=np.load(BUNDLE/'native_dense_route.npy');self.route=route[:,[0,2],3]
    def boxes(self,frame):
        rows=[]
        for name in ['RigidNodes','DeformableNodes','SMPLNodes']:
            key=name+'__instances_trans'
            if key not in self.tracks:continue
            t=self.tracks[key];q=self.tracks[name+'__instances_quats'];size=self.tracks[name+'__instances_size'];valid=self.tracks[name+'__instances_fv']
            lo=int(np.floor(frame));hi=int(np.ceil(frame));u=frame-lo
            for i in np.flatnonzero(valid[lo]|valid[hi]):
                if valid[lo,i] and valid[hi,i] and lo!=hi:
                    trans=t[lo,i]*(1-u)+t[hi,i]*u;rot=Slerp([0,1],Rotation.from_quat(q[[lo,hi],i][:,[1,2,3,0]]))([u])[0]
                else:
                    j=lo if valid[lo,i] else hi;trans=t[j,i];rot=Rotation.from_quat(q[j,i][[1,2,3,0]])
                forward=rot.as_matrix()[:,0];l,w,h=size[i]
                rows.append(dict(group=name,index=int(i),box=[*trans.tolist(),float(w),float(l),float(h),float(np.arctan2(forward[1],forward[0]))]))
        return rows
    def check(self,pose,frame,background=True):
        import torch
        v=self.vehicle;pose=np.asarray(pose);yaw=float(np.arctan2(pose[1,0],pose[0,0]))
        ego=[*pose[:3,3],2*v.half_width_m,2*v.half_length_m,v.body_height_m,yaw]
        objects=self.boxes(frame);overlaps=[r for r in objects if self.kernels['fg_collision_det'](ego,[r['box']])]
        # Stock canonical vertex order is part of bg_collision_det's O/A/B/C contract.
        canonical=np.array([[.5,.5,0],[.5,-.5,0],[.5,.5,1],[.5,-.5,1],[-.5,-.5,0],[-.5,.5,0],[-.5,-.5,1],[-.5,.5,1]])
        verts=canonical*np.array([2*v.half_length_m,2*v.half_width_m,v.body_height_m]);world=verts@pose[:3,:3].T+pose[:3,3]
        bg=bool(self.kernels['bg_collision_det'](self.points,torch.as_tensor(world,dtype=torch.float32,device=self.device))) if background else False
        known_bg=bool(self.kernels['bg_collision_det'](self.known_points,torch.as_tensor(world,dtype=torch.float32,device=self.device))) if background else False
        native_xy=np.array([-pose[1,3],pose[0,3]]);dist=np.linalg.norm(self.route-native_xy,axis=1);closest=int(np.argmin(dist));rc=(closest+1)/(.9*len(self.route))
        target=next((r for r in objects if r['group']=='RigidNodes' and r['index']==3),None)
        relative=None
        if target:
            relative=(np.linalg.inv(pose)@np.r_[target['box'][:3],1])[:3].tolist()
            t=self.tracks['RigidNodes__instances_trans'];i=min(max(int(round(frame)),1),len(t)-1)
            target['backward_difference_velocity_source_mps']=((t[i,3]-t[i-1,3])/.1).tolist()
            target['backward_difference_speed_mps']=float(np.linalg.norm((t[i,3]-t[i-1,3])/.1))
        return dict(foreground_collision=bool(overlaps),overlaps=overlaps,background_collision=bg,
            background_collision_with_known_semantics=known_bg,background_collision_depends_on_unknown_semantics=bg and not known_bg,
            collision=bg or bool(overlaps),route_distance_m=float(dist.min()),route_completion=rc,
            target_world=target,target_relative_forward_left_up_m=relative,
            unknown_background_points=self.unknown_background_points,
            native_background_opacity_threshold=.8,
            actual_hd_road_boundary_departure=None,physical_road_boundary_verified=False,
            native_route_deviation=bool(dist.min()>10.),native_route_completed=bool(rc>=1.))
