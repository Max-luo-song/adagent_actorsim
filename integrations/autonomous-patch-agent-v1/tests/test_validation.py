import copy,unittest
from helpers import config,metrics
from validation import acceptance
class AcceptanceTests(unittest.TestCase):
 def result(self,n,b=None,r=None):return acceptance(n,b or metrics(),r or metrics(),config()['acceptance'],{'status':'failed','missing_cases':['original172']})
 def test_small_positive_gain_passes_local_only(self):
  r=self.result(metrics(.26001));self.assertTrue(r['accepted']);self.assertFalse(r['global_patch_validated']);self.assertFalse(r['formal_certification']);self.assertEqual(r['strict_matrix_status'],'failed')
 def test_equal_or_worse_target_is_rejected(self):
  for hd in (.26,.259):self.assertFalse(self.result(metrics(hd))['accepted'])
 def test_original_regression_or_safety_overrides_target_gain(self):
  for name in ('collision','map_departure','unresolved_stop','original_score'):
   n=metrics(.27)
   if name=='original_score':n['original551']['score']['hdscore']-=.00001
   else:n['cutin551'][name]=True
   self.assertFalse(self.result(n)['accepted'])
 def test_lateral_drift_cannot_accumulate_across_rounds(self):
  champion=metrics(.261);champion['cutin551']['final_abs_lateral_m']=1.2
  n=metrics(.262);n['cutin551']['final_abs_lateral_m']=1.4
  self.assertFalse(self.result(n,champion,metrics())['accepted'])
 def test_incomplete_and_nonfinite_metrics_fail(self):
  for kind in ('controls','observations','nan'):
   n=metrics(.27)
   if kind=='nan':n['cutin551']['score']['hdscore']=float('nan')
   else:n['cutin551'][kind]-=1
   self.assertFalse(self.result(n)['accepted'])
 def test_increment_must_beat_current_champion_not_only_v13(self):
  self.assertFalse(self.result(metrics(.261),metrics(.262),metrics(.26))['accepted'])
if __name__=='__main__':unittest.main()
