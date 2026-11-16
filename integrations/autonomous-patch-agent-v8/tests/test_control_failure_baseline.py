import copy,unittest
from convergence import initial,observe
from test_convergence_rotation import config,metrics

class RecoveryConvergenceTests(unittest.TestCase):
 def test_failed_target_is_kept_failed_and_recovery_does_not_count_as_plateau(self):
  p=config()['convergence'];p['allow_control_failure_baseline']=True;b=metrics(.2);b['cutin551'].update(status='terminated',controls=116,observations=117,termination_reason='no_jointly_verified_action',error_stage=None)
  ledger=initial('failure',b,p,True);self.assertEqual(b['cutin551']['status'],'terminated');self.assertTrue(ledger['recovery_baseline'])
  r=observe(ledger,'recovered',metrics(.27),dict(accepted=True,reasons=[]),.2,True)
  self.assertEqual(r['classification'],'failure_recovery_not_a_comparable_gain');self.assertEqual(ledger['small_gain_streak'],0);self.assertFalse(ledger['recovery_baseline']);self.assertEqual(ledger['best_score'],.27)
 def test_infrastructure_exit_never_becomes_baseline(self):
  p=config()['convergence'];p['allow_control_failure_baseline']=True;b=metrics(.2);b['cutin551'].update(status='failed',error_stage='worker_boot')
  with self.assertRaises(Exception):initial('bad',b,p,True)
