"""SLSQP change of variables from physical units to physical-bound units."""
import numpy as np

def solve(minimize,objective,x0,constraints,bounds,options):
 x0=np.asarray(x0,dtype=float);limits=np.asarray(bounds,dtype=float);scale=np.max(np.abs(limits),axis=1);scale=np.where(scale>0.,scale,1.)
 if limits.shape!=(len(x0),2) or not np.isfinite(limits).all():raise ValueError('Finite physical bounds required for scaling')
 result=minimize(lambda z:objective(z*scale),x0/scale,method='SLSQP',bounds=[tuple(row) for row in limits/scale[:,None]],constraints=[{'type':'ineq','fun':lambda z:constraints(z*scale)}],options=options)
 result.x=np.asarray(result.x)*scale
 return result
