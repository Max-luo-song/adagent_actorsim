import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pseudo_lidar_adapter import TemporalRefinementGate
class TemporalDepthGateTests(unittest.TestCase):
 def setUp(self):
  self.g=TemporalRefinementGate();self.info=dict(scene_token='a',timestamp=0.,explicit_l2g_r_mat=np.eye(3),explicit_l2g_t=np.zeros(3));self.box=np.array([0.,8.,0.,2.,4.,1.5,0.,0.,0.])
 def seed(self):
  self.g.begin(self.info,'a');self.g.consider(dict(accepted=True,box=self.box.tolist()),self.box,self.info)
 def proposal(self,center=8.1):
  box=self.box.copy();box[1]=center
  return dict(accepted=False,box=self.box.tolist(),proposed_box=box.tolist(),before_surface_residual_m=.022,after_surface_residual_m=.004)
 def test_small_gain_needs_previous_frame_supported_geometry(self):
  self.g.begin(self.info,'a');self.assertFalse(self.g.consider(self.proposal(),self.box,self.info)['accepted'])
  self.seed();self.info['timestamp']=.1;self.g.begin(self.info,'a');self.assertTrue(self.g.consider(self.proposal(),self.box,self.info)['accepted'])
 def test_no_same_frame_confirmation_from_another_query(self):
  self.seed();self.assertFalse(self.g.consider(self.proposal(),self.box,self.info)['accepted'])
 def test_inconsistent_geometry_does_not_gain_admission(self):
  self.seed();self.info['timestamp']=.1;self.g.begin(self.info,'a');self.assertFalse(self.g.consider(self.proposal(10.),self.box,self.info)['accepted'])
 def test_scene_change_and_gap_invalidate_history(self):
  self.seed();self.info['timestamp']=1.;self.g.begin(self.info,'a');self.assertFalse(self.g.consider(self.proposal(),self.box,self.info)['accepted'])
  self.seed();self.info['timestamp']=1.1;self.g.begin(self.info,'b');self.assertFalse(self.g.consider(self.proposal(),self.box,self.info)['accepted'])
 def test_yaw_and_size_changes_cannot_enter_through_temporal_gate(self):
  self.seed();self.info['timestamp']=.1;self.g.begin(self.info,'a');r=self.proposal();r['proposed_box'][6]=1.;self.assertFalse(self.g.consider(r,self.box,self.info)['accepted'])
if __name__=='__main__':unittest.main()
