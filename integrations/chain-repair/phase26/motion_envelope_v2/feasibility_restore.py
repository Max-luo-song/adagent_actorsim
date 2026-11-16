"""Phase-I search: unchanged physical and obstacle constraints, never relaxed."""
import time,numpy as np
from scipy.optimize import minimize

def restore(trials,constraints,objective,bounds,maxiter=100,starts=3):
 candidates=sorted([r for r in trials if np.isfinite(r.get('minimum_constraint',np.nan)) and np.isfinite(r.get('controls')).all()],key=lambda r:r['minimum_constraint'],reverse=True)[:starts];attempts=[];best=None
 for row in candidates:
  initial=np.asarray(row['controls']).ravel();began=time.perf_counter()
  def violation(x):
   values=np.asarray(constraints(x));bad=np.minimum(values,0.);return float(np.dot(bad,bad)) if np.isfinite(values).all() else 1e20
  solved=minimize(violation,initial,method='SLSQP',bounds=bounds,options={'maxiter':maxiter,'ftol':1e-12,'disp':False});residual=float(np.min(constraints(solved.x)))
  attempts.append(dict(seed=row['seed'],minimum_constraint=residual,iterations=int(getattr(solved,'nit',0)),wall_seconds=time.perf_counter()-began,feasible=residual>=-1e-5))
  if residual>=-1e-5 and np.isfinite(solved.x).all() and all(lo-1e-8<=value<=hi+1e-8 for value,(lo,hi) in zip(solved.x,bounds)):
   # A restoration candidate is admitted only after every original row passes.
   cost=float(objective(solved.x))
   if best is None or cost<best['objective']:best=dict(x=solved.x.copy(),objective=cost,minimum_constraint=residual,seed=row['seed'])
 if best is not None:
  refined=minimize(objective,best['x'],method='SLSQP',bounds=bounds,constraints=[dict(type='ineq',fun=constraints)],options={'maxiter':min(40,maxiter),'ftol':1e-6,'disp':False})
  if np.isfinite(refined.x).all() and float(np.min(constraints(refined.x)))>=-1e-5 and all(lo-1e-8<=value<=hi+1e-8 for value,(lo,hi) in zip(refined.x,bounds)) and float(objective(refined.x))<best['objective']:best.update(x=refined.x.copy(),objective=float(objective(refined.x)),minimum_constraint=float(np.min(constraints(refined.x))))
 return best,attempts
