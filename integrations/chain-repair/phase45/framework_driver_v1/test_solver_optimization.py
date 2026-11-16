from bootstrap import *
import unittest,numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.optimize._slsqp_py import _epsilon
from scipy.optimize import minimize
from joint_difference import JointDifference,solve
from consensus_search import ConsensusSearch
from ranked_search import candidates
from mpc_schedule import POLICY
class DifferenceTests(unittest.TestCase):
 def test_exact_scipy_stencil_at_zero_and_bounds(self):
  fun=lambda x:np.sin(x[0])+x[1]**2+np.exp(x[2]);con=lambda x:np.r_[x,1.-x,np.minimum(x[0],x[1])-.1]
  for x in (np.array([0.,1.,.4]),np.array([1.,0.,1.]),np.array([.3,.8,.2])):
   engine=JointDifference(fun,con,[(0.,1.)]*3);a=approx_derivative(fun,x,method='2-point',abs_step=_epsilon,bounds=(np.zeros(3),np.ones(3)));b=approx_derivative(con,x,method='2-point',abs_step=_epsilon,bounds=(np.zeros(3),np.ones(3)))
   self.assertEqual(a.tobytes(),engine.gradient(x).tobytes());self.assertEqual(b.tobytes(),engine.jacobian(x).tobytes());self.assertEqual(engine.metrics['stencils'],1);self.assertEqual(engine.metrics['cache_hits'],1)
 def test_derivatives_are_current_and_returned_arrays_not_shared(self):
  engine=JointDifference(lambda x:float(x@x),lambda x:np.r_[x,1-x],[(0.,1.)]*2);x=np.array([.2,.4]);a=engine.gradient(x);a[:]=999;self.assertLess(engine.gradient(x).max(),2);before=engine.metrics['stencils'];x[0]+=.01;engine.jacobian(x);self.assertEqual(engine.metrics['stencils'],before+1)
 def test_real_slsqp_solution_and_constraints_match(self):
  f=lambda x:float((x[0]-.8)**2+2*(x[1]-.6)**2);c=lambda x:np.r_[1.-x.sum(),x];bounds=[(0.,1.)]*2;options=dict(maxiter=220,ftol=1e-6,disp=False)
  a=minimize(f,[.1,.2],method='SLSQP',bounds=bounds,constraints=[dict(type='ineq',fun=c)],options=options);b=solve(minimize,f,[.1,.2],c,bounds,options);self.assertEqual(a.x.tobytes(),b.x.tobytes());self.assertEqual(a.fun,b.fun);self.assertGreaterEqual(c(b.x).min(),-1e-5)
class RankingTests(unittest.TestCase):
 def test_current_feasible_seed_cost_orders_only_expansion(self):
  search=ConsensusSearch(POLICY,7.);seeds=[(n,np.zeros((6,2))) for n in ['warm','coast','escape_left','escape_right','bad','expensive','cheap']];scored=[];visited=[]
  def score(name,seed):scored.append(name);return {'bad':(-1.,0.),'expensive':(1.,9.),'cheap':(1.,2.)}[name]
  for name,seed,budget,stage in candidates(search,seeds,220,score):
   visited.append(name);self.assertEqual(budget,220)
  self.assertEqual(visited,['warm','coast','escape_left','escape_right','cheap','expensive','bad']);self.assertEqual(set(scored),{'bad','expensive','cheap'});self.assertFalse(search.audit['candidate_pruning'])
 def test_quality_and_agreement_rules_unchanged_and_lateral_coverage_retained(self):
  search=ConsensusSearch(POLICY,7.);seeds=[(n,np.zeros((6,2))) for n in ['warm','coast','escape_left','escape_right','extra']];visited=[]
  for name,seed,budget,stage in candidates(search,seeds,220,lambda *a:(_ for _ in ()).throw(AssertionError('Unnecessary ranking'))):
   visited.append(name);search.observe(dict(start_key=name,seed=name,objective=20.,feasible=True,solver_success=True,used_verified_seed=False,states=[[0.,0.,0.,8.]],native_alignment_proxy=dict(local_progress_proxy=1.,dac_proxy=1.,comfort_native_kinematic_flag=1.)))
  self.assertEqual(visited,['warm','coast','escape_left','escape_right']);self.assertTrue(search.audit['agreement_passed'])
 def test_unpromising_seed_order_is_preserved_after_promotions(self):
  search=ConsensusSearch(POLICY,7.);search.rows=[dict(seed='existing',objective=10.,feasible=True,solver_success=False,states=[[0.,0.,0.,8.]])];seeds=[(n,None) for n in ['warm','coast','high','middle','cheap']];scores={'high':(1.,90.),'middle':(1.,30.),'cheap':(1.,12.)};visited=[n for n,*_ in candidates(search,seeds,220,lambda n,s:scores[n])];self.assertEqual(visited[-3:],['cheap','high','middle']);self.assertTrue(search.audit['unpromoted_order_preserved'])
 def test_nonfinite_seed_score_is_retained_last_without_json_infinity(self):
  import json
  search=ConsensusSearch(POLICY,7.);seeds=[('warm',None),('coast',None),('nan',None),('finite',None)];visited=[n for n,*_ in candidates(search,seeds,220,lambda n,s:((float('nan'),float('inf')) if n=='nan' else (1.,2.)))];self.assertEqual(visited[-2:],['finite','nan']);json.dumps(search.audit,allow_nan=False)
 def test_changed_problem_never_reuses_old_ranking(self):
  seeds=[('warm',None),('coast',None),('left',None),('right',None)];orders=[]
  for sign in (1.,-1.):
   search=ConsensusSearch(POLICY,7.);orders.append([n for n,*_ in candidates(search,seeds,220,lambda name,seed:(1.,sign*(1 if name=='left' else 2)))])
  self.assertNotEqual(orders[0],orders[1])
if __name__=='__main__':unittest.main(verbosity=2)
