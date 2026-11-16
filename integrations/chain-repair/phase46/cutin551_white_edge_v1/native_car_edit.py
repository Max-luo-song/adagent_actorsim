"""Original-RigidNodes trajectory edit: shared tensors, revision, reset and SH."""
from contextlib import contextmanager
import copy,re
import numpy as np
from scipy.spatial.transform import Rotation
from scenario import compile_vehicle_path

def qmat(q):
 a=np.asarray(q,dtype=float).reshape(4);return Rotation.from_quat(a[[1,2,3,0]]).as_matrix()

def effective_source(t,q,visible,frame,tests):
 if frame in tests and frame>1 and frame+1<len(t) and visible[frame-1] and visible[frame+1]:
  a=np.asarray(q[frame-1],dtype=float);b=np.asarray(q[frame+1],dtype=float)
  from scipy.spatial.transform import Slerp
  rotation=Slerp([0.,1.],Rotation.from_quat(np.stack([a,b])[:,[1,2,3,0]]))([.5]).as_quat()[0][[3,0,1,2]]
  return (t[frame-1]+t[frame+1])*.5,rotation
 return t[frame].copy(),q[frame].copy()

class VehiclePrograms:
 def __init__(self,backend,spec):
  self.backend=backend;self.spec=spec;self.model=backend.trainer.models['RigidNodes'];self.programs={};self.tests=set(map(int,backend.trainer.test_timesteps if hasattr(backend.trainer,'test_timesteps') else getattr(backend.dataset,'test_timesteps',[])))
  for k in ('instances_trans','instances_quats','instances_fv','instances_size'):
   if backend.env.models['RigidNodes'][k] is not getattr(self.model,k):raise RuntimeError('Vehicle renderer/proxy identity mismatch')
 def apply(self,command):
  index=int(command['object_id'].split(':')[-1]);m=self.model;torch=self.backend.torch
  if not 0<=index<m.instances_trans.shape[1]:raise ValueError('Native vehicle does not exist')
  previous=self.programs.get(index);base_t=previous['base_t'] if previous else m.instances_trans[:,index].detach().clone();base_q=previous['base_q'] if previous else m.instances_quats[:,index].detach().clone()
  if command['operation']=='restore_actor_trajectory':planned_t,planned_q,audit=base_t,base_q,dict(restored=True)
  else:
   visible=m.instances_fv[:,index].detach().cpu().numpy()
   if not visible.all():raise ValueError('Vehicle path requires complete source visibility')
   t,q,audit=compile_vehicle_path(command['trajectory'],base_t.cpu().numpy(),base_q.cpu().numpy(),self.spec.fps);planned_t=base_t.new_tensor(t);planned_q=base_q.new_tensor(q)
  from native_radiance import radiance_state
  appearance=previous['appearance'] if previous else radiance_state(m,index)
  old_t=m.instances_trans[:,index].detach().clone();old_q=m.instances_quats[:,index].detach().clone()
  try:
   with torch.no_grad():m.instances_trans[:,index].copy_(planned_t);m.instances_quats[:,index].copy_(planned_q)
   self.backend.clear_observation_history()
  except BaseException:
   with torch.no_grad():m.instances_trans[:,index].copy_(old_t);m.instances_quats[:,index].copy_(old_q)
   raise
  if command['operation']=='restore_actor_trajectory':self.programs.pop(index,None)
  else:self.programs[index]=dict(base_t=base_t,base_q=base_q,t=planned_t,q=planned_q,appearance=appearance,audit=audit)
  return dict(model_index=index,program_active=index in self.programs,proxy_tensor_identity_preserved=True,source_gaussians_preserved=True,source_visibility_preserved=True,**audit)
 @contextmanager
 def render_context(self,frame):
  torch=self.backend.torch;m=self.model;saved=[]
  try:
   for index,entry in self.programs.items():
    # Make target pose exact at held-out source frames; other actors untouched.
    if frame in self.tests and frame>1 and frame+1<self.spec.frames and bool(m.instances_fv[frame-1,index] and m.instances_fv[frame+1,index]):
     for j in (frame-1,frame+1):
      saved.append(('root',index,j,m.instances_trans[j,index].detach().clone(),m.instances_quats[j,index].detach().clone()))
      with torch.no_grad():m.instances_trans[j,index].copy_(entry['t'][frame]);m.instances_quats[j,index].copy_(entry['q'][frame])
    appearance=entry['appearance']
    if appearance is not None:
     indices,rest,degree=appearance
     base_t,base_q=effective_source(entry['base_t'].cpu().numpy(),entry['base_q'].cpu().numpy(),m.instances_fv[:,index].cpu().numpy(),frame,self.tests)
     delta=qmat(entry['q'][frame].cpu().numpy())@qmat(base_q).T
     if np.allclose(delta,np.eye(3),rtol=0.,atol=1e-12):continue
     from asset_radiance import sh_rotation
     mat=rest.new_tensor(sh_rotation(delta,degree)[1:,1:]);saved.append(('sh',indices,m._features_rest[indices].detach().clone()))
     with torch.no_grad():m._features_rest[indices]=torch.einsum('ij,njc->nic',mat,rest)
   yield
  finally:
   with torch.no_grad():
    for row in reversed(saved):
     if row[0]=='root':_,index,j,t,q=row;m.instances_trans[j,index].copy_(t);m.instances_quats[j,index].copy_(q)
     else:_,indices,features=row;m._features_rest[indices]=features
 def restore(self):
  with self.backend.torch.no_grad():
   for index,e in self.programs.items():self.model.instances_trans[:,index].copy_(e['base_t']);self.model.instances_quats[:,index].copy_(e['base_q'])
  self.programs.clear()
 def export(self):
  m=self.model;trans=m.instances_trans.detach().cpu().numpy().copy();quats=m.instances_quats.detach().cpu().numpy().copy();valid=m.instances_fv.detach().cpu().numpy().copy();size=m.instances_size.detach().cpu().numpy().copy()
  for i in range(trans.shape[1]):
   if i in self.programs:continue
   source_t=trans[:,i].copy();source_q=quats[:,i].copy()
   for f in self.tests:
    if 0<=f<len(trans):trans[f,i],quats[f,i]=effective_source(source_t,source_q,valid[:,i],f,self.tests)
  return dict(translations=trans,quaternions_wxyz=quats,visible=valid,size_lwh=size)

def make_types():
 from scene_edit_session import UnifiedSceneSession
 from sensor_backend import ClosedLoopFrameworkBackend
 class Backend(ClosedLoopFrameworkBackend):
  def render(self,pose,timestamp):
   owner=getattr(self,'vehicle_programs',None)
   if owner is None:return super().render(pose,timestamp)
   frame=min(max(int(round(timestamp*self.spec.fps)),0),self.frames-1)
   with owner.render_context(frame):return super().render(pose,timestamp)
 class Session(UnifiedSceneSession):
  def load_scene(self,manifest):
   result=super().load_scene(manifest);self.vehicles=VehiclePrograms(self.backend,self.spec);self.backend.vehicle_programs=self.vehicles;return result
  def edit(self,value):
   if not isinstance(value,dict):raise ValueError('Edit must be a mapping')
   if not (value.get('operation') in ('set_actor_trajectory','restore_actor_trajectory') and str(value.get('object_id','')).startswith('native:RigidNodes:')):return super().edit(value)
   editor=self.editor;editor.require_current()
   if editor.started or self.backend._last is not None:raise RuntimeError('Vehicle edits require pre-run phase')
   expected={'schema','operation','object_id','mode','expected_revision','expected_generation'}|({'trajectory'} if value['operation']=='set_actor_trajectory' else set())
   if set(value)!=expected or value['schema']!='scene_edit_v1' or value['mode']!='diagnostic_preview':raise ValueError('Invalid native vehicle edit contract')
   if not re.fullmatch(r'native:RigidNodes:(0|[1-9][0-9]*)',value['object_id']) or type(value['expected_revision']) is not int or not isinstance(value['expected_generation'],str):raise ValueError('Invalid vehicle identity/version')
   if value['expected_revision']!=self.revision or value['expected_generation']!=self.generation:raise ValueError('Stale native vehicle edit')
   previous=self.revision;evidence=self.vehicles.apply(value);self.revision+=1
   receipt=dict(schema='scene_edit_receipt_v1',command=copy.deepcopy(value),generation=self.generation,previous_revision=previous,edit_revision=self.revision,evidence=evidence,status='diagnostic_unverified',formal_evaluation_admitted=False);editor.journal.append(copy.deepcopy(receipt));return receipt
  def _packet(self,*args,**kwargs):
   packet=super()._packet(*args,**kwargs)
   if hasattr(self,'vehicles') and self.vehicles.programs:
    packet['native_vehicle_trajectories']={'objects':['native:RigidNodes:'+str(i) for i in self.vehicles.programs],'clock':'source_time_only'};packet['edit_trust'].update(edited=True,status='diagnostic_unverified')
   return packet
  def reset(self):self.vehicles.restore();result=super().reset();self.vehicles=VehiclePrograms(self.backend,self.spec);self.backend.vehicle_programs=self.vehicles;return result
  def close(self):
   if getattr(self,'vehicles',None) is not None:self.vehicles.restore();self.backend.vehicle_programs=None;self.vehicles=None
   super().close()
 return Backend,Session
