from bootstrap import *
import copy,unittest
from search_acceptance import validate_document
class SearchAcceptanceTests(unittest.TestCase):
 def document(self):
  row=dict(baseline_matches_frozen_v2=True,both_selected=True,independent_passed=True,forecast_recovery=False,identical_inputs=True,objective_increase_fraction=.001,minimum_constraint=.1,iteration_budget_preserved=True,lateral_seeds_visited=True,baseline_seconds=100.,candidate_seconds=50.)
  return dict(schema='framework_search_migration_acceptance_v3',passed=True,physics_and_objective_preserved=True,paired_frames=[copy.deepcopy(row),copy.deepcopy(row)],held_out_expansion=dict(kind='archived_scheduling_replay',expanded=True,original_starts_preserved=True,original_iteration_budget_preserved=True,same_archived_best=True),new_gpu_executed=False,whole_loop_speedup_verified=False)
 def test_fast_frame_slowdown_is_explicitly_allowed_only_with_aggregate_gain(self):
  d=self.document();d['paired_frames'][0].update(baseline_seconds=20.,candidate_seconds=60.);self.assertTrue(validate_document(d))
 def test_quality_or_coverage_cannot_be_traded_for_speed(self):
  for key,value in [('baseline_matches_frozen_v2',False),('independent_passed',False),('identical_inputs',False),('objective_increase_fraction',.02),('minimum_constraint',-.1),('iteration_budget_preserved',False),('lateral_seeds_visited',False)]:
   d=self.document();d['paired_frames'][0][key]=value
   with self.assertRaises(ValueError):validate_document(d)
 def test_no_gain_or_missing_full_search_fallback_cannot_be_accepted(self):
  for mode in ['time','fallback','gpu_claim']:
   d=self.document()
   if mode=='time':
    for r in d['paired_frames']:r['candidate_seconds']=101.
   elif mode=='fallback':d['held_out_expansion']['original_starts_preserved']=False
   else:d['whole_loop_speedup_verified']=True
   with self.assertRaises(ValueError):validate_document(d)
if __name__=='__main__':unittest.main(verbosity=2)
