from bootstrap import *
import unittest,numpy as np
from consensus_search import ConsensusSearch
from mpc_schedule import POLICY
class ConsensusTests(unittest.TestCase):
 def row(self,name,cost,speed=8.,key=None):return dict(seed=name,objective=cost,states=[[0.,0.,0.,speed,0.]],feasible=True,solver_success=True,used_verified_seed=False,start_key=key or name,native_alignment_proxy=dict(local_progress_proxy=1.,dac_proxy=1.,comfort_native_kinematic_flag=1.))
 def seeds(self):return [(n,np.zeros((6,2))) for n in ['warm','route','coast','aux_left','aux_right']]
 def test_two_converged_distinct_primaries_stop_at_original_budget(self):
  search=ConsensusSearch(POLICY,7.);visited=[]
  for name,seed,budget,stage in search.candidates(self.seeds(),220):visited.append(name);self.assertEqual(budget,220);search.observe(self.row(name,50. if name=='warm' else 50.1))
  self.assertEqual(visited,['warm','coast']);self.assertTrue(search.audit['agreement_passed'])
 def test_poor_quality_or_disagreement_searches_each_remaining_start_once(self):
  for case in ['cost','stop','same_key','infeasible','failed_solver']:
   search=ConsensusSearch(POLICY,7.);visited=[]
   for name,seed,budget,stage in search.candidates(self.seeds(),220):
    visited.append(name);row=self.row(name,20. if name=='warm' else 40. if case=='cost' else 20.1,speed=0. if case=='stop' else 8.,key='same' if case=='same_key' else name)
    if case=='infeasible':row['feasible']=False
    if case=='failed_solver':row['solver_success']=False
    search.observe(row)
   self.assertEqual(len(visited),5);self.assertEqual(len(set(visited)),5);self.assertTrue(search.audit['expanded'])
 def test_explicit_neural_stop_can_accept_two_safe_converged_stops(self):
  search=ConsensusSearch(POLICY,7.,neural_stop=True);visited=[]
  for name,seed,budget,stage in search.candidates(self.seeds(),220):visited.append(name);search.observe(self.row(name,20.,speed=0.))
  self.assertEqual(len(visited),2)
 def test_agreement_after_expansion_keeps_best_and_original_budget(self):
  search=ConsensusSearch(POLICY,6.);visited=[]
  for name,seed,budget,stage in search.candidates(self.seeds(),220):
   visited.append(name);self.assertEqual(budget,220)
   search.observe(self.row(name,50. if name=='warm' else 40. if name=='coast' else 40.1,speed=6.8))
  self.assertEqual(visited,['warm','coast','route']);self.assertTrue(search.audit['expanded']);self.assertTrue(search.audit['agreement_passed'])
 def test_lower_bad_quality_outlier_cannot_be_ignored(self):
  search=ConsensusSearch(POLICY,7.);visited=[]
  for name,seed,budget,stage in search.candidates(self.seeds(),220):
   visited.append(name);search.observe(self.row(name,10. if name=='warm' else 20.,speed=0. if name=='warm' else 8.))
  self.assertEqual(len(visited),5);self.assertFalse(search.audit['agreement_passed'])
 def test_a_new_best_requires_independent_agreement(self):
  search=ConsensusSearch(POLICY,7.);visited=[]
  for name,seed,budget,stage in search.candidates(self.seeds(),220):
   visited.append(name);cost={'warm':50.,'coast':40.,'route':30.,'aux_left':30.1,'aux_right':30.}[name];search.observe(self.row(name,cost))
  self.assertEqual(visited,['warm','coast','route','aux_left'])
 def test_both_lateral_basins_are_visited_before_agreement(self):
  search=ConsensusSearch(POLICY,7.);visited=[]
  seeds=[(n,np.zeros((6,2))) for n in ['warm','coast','route','escape_left','escape_right']]
  for name,seed,budget,stage in search.candidates(seeds,220):
   visited.append(name);search.observe(self.row(name,50. if name in ('warm','coast') else 49.9))
  self.assertEqual(visited,['warm','coast','escape_left','escape_right']);self.assertTrue(search.audit['agreement_passed'])
if __name__=='__main__':unittest.main(verbosity=2)
