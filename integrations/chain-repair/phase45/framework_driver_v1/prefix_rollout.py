"""Exact control-prefix reuse inside one fixed current MPC problem."""
from collections import OrderedDict
import numpy as np
class PrefixRollout:
 def __init__(self,speed,steer,offset,dt,wheelbase,interval,interpolation,capacity=128):
  if capacity<1:raise ValueError('Positive prefix capacity required')
  self.speed=speed;self.steer=steer;self.offset=np.array(offset,dtype=float,copy=True);self.dt=dt;self.wheelbase=wheelbase;self.interval=interval;self.interpolation=interpolation;self.substeps=int(round(interval/dt));self.cache=OrderedDict();self.capacity=capacity;self.metrics=dict(queries=0,reused_ticks=0,integrated_ticks=0)
  if interpolation not in ('constant','linear') or dt<=0 or interval<=0 or not np.isclose(self.substeps*dt,interval):raise ValueError('Invalid integration configuration')
 def __call__(self,controls):
  from empty_controller import integrate_state,rollout
  controls=np.asarray(controls).reshape(6,2)
  if controls.dtype!=np.float64 or not np.isfinite(controls).all():return rollout(controls,self.speed,self.steer,self.offset,self.dt,self.wheelbase,self.interval,self.interpolation)
  total=6*self.substeps;start=0;cached=None;keys=[(k,np.ascontiguousarray(controls[:k]).tobytes()) for k in range(1,7)]
  for k,key in reversed(list(enumerate(keys,1))):
   if key in self.cache:
    cached=self.cache[key];self.cache.move_to_end(key);start=total if k==6 else k*self.substeps if self.interpolation=='constant' else max(0,(k-1)*self.substeps)
    if start:break
  if cached is not None and start:
   states=[row.copy() for row in cached[0][:start+1]];lidar=[row.copy() for row in cached[1][:start+1]];state=states[-1].copy()
  else:
   state=np.array([*(-self.offset),0.,self.speed,self.steer],dtype=float);states=[state.copy()];lidar=[np.zeros(2)]
  for tick in range(start,total):
   action=np.array([np.interp(tick*self.dt,np.arange(6)*self.interval,controls[:,j]) for j in range(2)]) if self.interpolation=='linear' else controls[tick//self.substeps]
   state=integrate_state(state,action,self.dt,self.wheelbase);states.append(state.copy());yaw=state[2];ox=np.cos(yaw)*self.offset[0]+np.sin(yaw)*self.offset[1];oy=-np.sin(yaw)*self.offset[0]+np.cos(yaw)*self.offset[1];lidar.append(state[:2]+[ox,oy])
  states=np.asarray(states);lidar=np.asarray(lidar);saved=(states.copy(),lidar.copy());saved[0].flags.writeable=False;saved[1].flags.writeable=False
  for key in keys:
   self.cache[key]=saved;self.cache.move_to_end(key)
   if len(self.cache)>self.capacity:self.cache.popitem(last=False)
  self.metrics['queries']+=1;self.metrics['reused_ticks']+=start;self.metrics['integrated_ticks']+=total-start
  return states,lidar
