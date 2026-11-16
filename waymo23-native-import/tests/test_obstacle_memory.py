import sys,unittest,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planning_guard import ObstacleMemory,PlanningGuard
from runtime_configuration import load_vehicle_config
class Tests(unittest.TestCase):
 def setUp(self):
  self.m=ObstacleMemory();self.b=np.array([[0,8,0,2,4,1.5,0,0,0.]])
  self.info=dict(timestamp=0,scene_token='a',explicit_l2g_r_mat=np.eye(3),explicit_l2g_t=np.zeros(3))
 def update(self,score,state=None):
  obs=[] if state is None else [dict(candidate_index=0,state=state,raw_score=score,box=self.b[0].tolist(),label=0)]
  return self.m.update(dict(config={},obstacles=obs),self.b,np.array([score]),np.array([0]),self.info)
 def test_drop_and_loss(self):
  for t in (0.,.1,.2):
   self.info['timestamp']=t;self.update(.8,'network_retained')
  self.info['timestamp']=.3
  r=self.update(.3);self.assertEqual(r['obstacles'][0]['state'],'last_known_confirmed')
  self.info['timestamp']=.4;r=self.update(.01);self.assertEqual(len(r['obstacles']),1)
  g=PlanningGuard(r,load_vehicle_config(),np.array([[0,0,0,0,0]]),[0]);self.assertEqual(len(g.hard_boxes),1)
 def test_single_ghost(self):
  self.update(.24,'provisional_uncertain');self.info['timestamp']=.1
  self.assertFalse(self.update(.01)['obstacles'])
 def test_pose_and_reset(self):
  self.update(.8,'network_retained');self.info.update(timestamp=.1,explicit_l2g_t=np.array([0,1,0.]))
  r=self.update(.01);self.assertAlmostEqual(r['obstacles'][0]['box'][1],7)
  self.info.update(timestamp=.2,scene_token='b');self.assertFalse(self.update(.01)['obstacles'])
if __name__ == '__main__': unittest.main()
