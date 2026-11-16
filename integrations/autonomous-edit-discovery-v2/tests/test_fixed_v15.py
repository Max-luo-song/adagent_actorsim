import copy
import unittest
from common_flow import assess_failure,qualify_fixed


def metric(hd=.26):return dict(status='completed',controls=188,observations=189,collision=False,map_departure=False,unresolved_stop=False,score=dict(hdscore=hd,pdms=1.,rc=hd),termination_reason='source_scene_time_window_complete',error_stage=None)

class FixedDefaultTests(unittest.TestCase):
 def test_known_v15_long_brake_pass_is_not_a_failure(self):
  self.assertFalse(assess_failure(metric(.2717112138441646),metric(.2763164886550826))['candidate'])
 def test_repeatable_early_control_exit_is_admitted_without_v17(self):
  b=metric(.18);b.update(status='terminated',controls=116,observations=117,termination_reason='no_jointly_verified_action')
  self.assertTrue(qualify_fixed(b,copy.deepcopy(b),metric(.276),True)['qualified'])
 def test_worker_failure_is_not_control_failure(self):
  b=metric();b.update(status='failed',error_stage='worker_boot')
  self.assertFalse(assess_failure(b,metric(.276))['candidate'])
 def test_missing_feasible_witness_blocks_admission(self):
  b=metric(.18);b.update(status='terminated',controls=116,observations=117,termination_reason='no_jointly_verified_action')
  self.assertFalse(qualify_fixed(b,b,metric(.276),False)['qualified'])
 def test_nonrepeatable_control_exit_blocks_admission(self):
  b=metric(.18);b.update(status='terminated',controls=116,observations=117,termination_reason='no_jointly_verified_action');r=copy.deepcopy(b);r['controls']=117;r['observations']=118
  self.assertFalse(qualify_fixed(b,r,metric(.276),True)['qualified'])

class NativeEnvironmentTests(unittest.TestCase):
 def test_only_coordinator_keeps_api_credentials(self):
  import os
  from unittest.mock import patch
  from common_flow import native_environment
  with patch.dict(os.environ,{'PATCH_API_KEY':'fixture-secret','PATCH_MODEL':'fixture','CUDA_VISIBLE_DEVICES':'0'}):
   env=native_environment();self.assertNotIn('PATCH_API_KEY',env);self.assertNotIn('PATCH_MODEL',env);self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0')
