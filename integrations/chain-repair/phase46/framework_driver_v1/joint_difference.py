"""Bound-aware joint objective/constraint differencing, exact SciPy stencil."""
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.optimize._slsqp_py import _epsilon
class JointDifference:
 def __init__(self,objective,constraints,bounds):
  self.objective=objective;self.constraints=constraints;self.bounds=np.asarray(bounds,dtype=float).T;self.last=None;self.derivative=None;self.metrics=dict(stencils=0,cache_hits=0,joint_values=0)
 def value(self,x):
  self.metrics['joint_values']+=1
  return np.r_[float(self.objective(x)),np.asarray(self.constraints(x),dtype=float)]
 def ensure(self,x):
  x=np.clip(np.asarray(x,dtype=float),*self.bounds);key=x.tobytes()
  if key==self.last:self.metrics['cache_hits']+=1;return self.derivative
  f0=self.value(x);jac=approx_derivative(self.value,x,method='2-point',abs_step=_epsilon,bounds=self.bounds,f0=f0);self.last=key;self.derivative=jac;self.metrics['stencils']+=1;return jac
 def gradient(self,x):return self.ensure(x)[0].copy()
 def jacobian(self,x):return self.ensure(x)[1:].copy()
def solve(minimize,objective,seed,constraints,bounds,options):
 engine=JointDifference(objective,constraints,bounds)
 result=minimize(objective,seed,method='SLSQP',bounds=bounds,jac=engine.gradient,constraints=[dict(type='ineq',fun=constraints,jac=engine.jacobian)],options=options)
 result.joint_difference_metrics=engine.metrics.copy();return result
