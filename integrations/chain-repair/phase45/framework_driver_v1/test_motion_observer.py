from bootstrap import *
import copy, unittest
import numpy as np
from motion_observer import MotionObserver

class ObserverTests(unittest.TestCase):
 def sample(self,time,*,score=.2,supported=True,scene='a',position=None,velocity=(0.,15.)):
  box=[0.,10.+15*time,0.,2.,4.,1.5,0.,*velocity]
  if position is not None:box[:2]=position
  report=dict(config={},obstacles=[dict(candidate_index=0,raw_score=score,label=0,state='provisional',box=box,neural_box=box.copy())],global_evidence={'queries':[dict(index=0,protected_observation=supported)]})
  info=dict(timestamp=time,scene_token=scene,explicit_l2g_r_mat=np.eye(3),explicit_l2g_t=np.zeros(3))
  return report,info
 def test_depth_supported_weak_samples_qualify_motion_without_promoting_geometry(self):
  observer=MotionObserver();args=self.sample(0.);observer.observe(*args);args=self.sample(.1,score=.45);before=copy.deepcopy(args[0]);cert,audit=observer.observe(*args)
  self.assertIn(0,cert);self.assertEqual(args[0],before);self.assertFalse(audit['weak_classification_promoted_to_hard']);self.assertFalse(audit['geometry_anchors_updated'])
 def test_missing_support_static_and_contradictory_position_never_qualify(self):
  for mode in ['unsupported','static','opposite']:
   observer=MotionObserver();observer.observe(*self.sample(0.));kwargs={'supported':False} if mode=='unsupported' else {'position':(0.,10.)} if mode=='static' else {'position':(0.,8.5)}
   cert,_=observer.observe(*self.sample(.1,**kwargs));self.assertFalse(cert)
 def test_scene_clock_gap_and_static_snapshot_clear_history(self):
  for mode in ['scene','clock','gap','snapshot']:
   observer=MotionObserver();observer.observe(*self.sample(0.));args=self.sample(.1,scene='b') if mode=='scene' else self.sample(0.) if mode=='clock' else self.sample(1.) if mode=='gap' else self.sample(.1)
   if mode=='snapshot':args[1]['declared_static_epoch']='frozen'
   cert,_=observer.observe(*args);self.assertFalse(cert)
   if mode=='snapshot':self.assertFalse(observer.records)
 def test_same_frame_duplicate_queries_are_one_observation(self):
  observer=MotionObserver();report,info=self.sample(0.);duplicate=copy.deepcopy(report['obstacles'][0]);duplicate['candidate_index']=1;report['obstacles'].append(duplicate);report['global_evidence']['queries'].append(dict(index=1,protected_observation=True))
  cert,audit=observer.observe(report,info);self.assertFalse(cert);self.assertEqual(audit['depth_supported_queries'],1);self.assertEqual(len(observer.records),1)
 def test_ambiguous_association_cannot_create_motion_evidence(self):
  observer=MotionObserver();report,info=self.sample(0.);second=copy.deepcopy(report['obstacles'][0]);second['candidate_index']=1;second['box'][0]=.6;second['neural_box'][0]=.6;report['obstacles'].append(second);report['global_evidence']['queries'].append(dict(index=1,protected_observation=True));observer.observe(report,info)
  cert,audit=observer.observe(*self.sample(.1,position=(.3,11.5)));self.assertFalse(cert);self.assertGreater(audit['ambiguous_associations_rejected'],0)
 def test_world_motion_is_invariant_to_rotated_ego_pose(self):
  observer=MotionObserver();observer.observe(*self.sample(0.));report,info=self.sample(.1);rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
  for row in report['obstacles']:
   box=np.asarray(row['box']);box[:3]=rotation.T@box[:3];box[7:9]=(rotation.T@np.r_[box[7:9],0.])[:2];row['box']=box.tolist();row['neural_box']=box.tolist()
  info['explicit_l2g_r_mat']=rotation;cert,_=observer.observe(report,info);self.assertIn(0,cert);np.testing.assert_allclose(cert[0]['direction_xy'],[1.,0.],atol=1e-12)
if __name__=='__main__':unittest.main(verbosity=2)
