"""Frozen host tests of the candidate rollout using synthetic inputs only."""
import dataclasses,unittest
import numpy as np
import test_repair as fixture
from trajectory_quality import assess
from core import LOCAL,read

class CandidateMotionContract(unittest.TestCase):
 def probe(self,start_speed,target,curved):
  v=fixture.V;obj=fixture.JointReference(v,fixture.mapping(5.,-5.));m=fixture.Motion(v.dt,v.wheelbase_m,start_speed);plans=[]
  for i in range(6):
   info=fixture.info(m,i*v.dt);obj.observed({},info,{});raw=fixture.intent(m,target);obj._speed_reference_states=obj.states(raw);obj._risk_boxes=np.empty((0,9));obj._risk_tracks=[]
   guide=obj.guides(raw,obj._risk_boxes,m,target)[0]
   if curved:guide['x']=.2*np.sin((guide['y']-guide['y'][0])/10.)
   states,controls=obj.rollout(guide,m,target,5.)
   dense=states[:31].copy();dense=dense[:,[1,0,2,3,4]];pose=np.eye(4);c,s=np.cos(m.yaw),np.sin(m.yaw);pose[:2,:2]=[[c,s],[-s,c]];pose[:2,3]=[m.z,-m.x]
   plans.append({'timestamp':i*v.dt,'ego_to_world':pose.tolist(),'states':dense.tolist(),'action':{'acc_mps2':float(controls[0,0]),'steer_rate_radps':float(controls[0,1])}})
   m.advance({'acc':float(controls[0,0]),'steer_rate':float(controls[0,1])})
  result=assess(plans,dataclasses.asdict(v),read(LOCAL/'config.example.json')['host_governance']['motion']);self.assertTrue(result['passed'],result['reasons'])
 def test_acceleration_reference_is_smooth_across_frames(self):self.probe(2.5,4.,False)
 def test_curved_reference_is_smooth_across_frames(self):self.probe(2.5,3.,True)
 def test_low_speed_reference_has_no_position_teleport(self):self.probe(.1,.1,False)
