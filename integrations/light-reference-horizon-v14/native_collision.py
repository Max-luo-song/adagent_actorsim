"""Restore the unmodified native postprocessor after the verified common prefix."""
import functools,inspect,time,hashlib
import numpy as np
POLICY=dict(enabled_after_common_index=-1,implementation='unchanged UniAD collision_optimization',external_obstacle_mpc=False)

def install(core,protocol,read,write):
 head=core.planning_head;forward=head.forward;native=head.collision_optimization;previous_flag=head.use_col_optim;active={}
 @functools.wraps(native)
 def collision(trajectory,occupancy):
  active['before']=trajectory.detach().clone();active['method_calls']+=1
  return native(trajectory,occupancy)
 @functools.wraps(forward)
 def planning(*args,**kwargs):
  folder=max(protocol.JOB.glob('step_[0-9][0-9][0-9]'));index=int(folder.name.split('_')[-1]);enabled=index>POLICY['enabled_after_common_index'];head.use_col_optim=enabled
  bound=inspect.signature(forward).bind(*args,**kwargs);bound.apply_defaults();occ=bound.arguments.get('occ_mask');active.clear();active['method_calls']=0;began=time.monotonic();result=forward(*args,**kwargs)
  after=result['sdc_traj'];before=active.get('before',after).detach().clone()
  if after.shape!=(1,head.planning_steps,2) or not np.isfinite(after.detach().cpu().numpy()).all():raise ValueError('Invalid native collision trajectory')
  occ_available=occ is not None;near=[]
  if occ_available:
   masks=occ.squeeze(2) if occ.ndim==5 and occ.shape[2]==1 else occ
   if tuple(masks.shape)!=(1,5,head.bev_h,head.bev_w):raise ValueError('Unexpected native occupancy layout')
   for t in range(head.planning_steps):
    import torch
    cells=torch.nonzero(masks[0,min(t+1,4)],as_tuple=False)[:,[1,0]].to(before)
    if len(cells):
     cells[:,0]=(cells[:,0]-head.bev_h//2)*.5+.25;cells[:,1]=(cells[:,1]-head.bev_w//2)*.5+.25
     near.append(int((((before[0,t,:2]-cells[:,:2])**2).sum(-1)<head.occ_filter_range**2).sum()))
    else:near.append(0)
  pre=before.cpu().numpy().copy();post=after.detach().cpu().numpy().copy();np.savez_compressed(folder/'native_collision_plans.npz',before=pre,after=post)
  write(folder/'native_collision_audit.json',dict(plans_sha256=hashlib.sha256((folder/'native_collision_plans.npz').read_bytes()).hexdigest(),schema='native_collision_postprocessor_v1',index=index,enabled=enabled,native_code_modified=False,policy=POLICY,occupancy_available=occ_available,occupied_cells=int((occ!=0).sum()) if occ_available else None,near_reference_cells_per_step=near,native_method_calls=active['method_calls'],native_filter_requires_solver=bool(enabled and sum(near)>0),trajectory_max_change_m=float(np.max(np.abs(post-pre))),skip_reason='common_prefix_matches_original_A' if not enabled else 'occupancy_unavailable_native_skip' if not occ_available else 'no_occupancy_inside_native_filter' if not sum(near) else None,wall_seconds=time.monotonic()-began,missing_occupancy_replaced=False,ground_truth_used=False))
  return result
 head.forward=planning;head.collision_optimization=collision
 def cleanup():head.forward=forward;head.collision_optimization=native;head.use_col_optim=previous_flag
 return cleanup
