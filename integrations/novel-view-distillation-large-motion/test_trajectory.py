import json,unittest
import numpy as np
from trajectory import build

class TrajectoryTests(unittest.TestCase):
    def source(self):
        values=[]
        for i in range(61):
            pose=np.eye(4);pose[0,3]=2*i
            values.append(dict(time_s=i*.1,source_frame=i,recorded_ego_to_world=pose.tolist()))
        return dict(samples=values,gravity_up_source=[0.,0.,1.],start_frame=0)
    def test_visible_displacement_with_real_tangent_heading(self):
        task,audit=build(self.source());p=np.array([w['ego_to_world'] for w in task['waypoints']])
        np.testing.assert_allclose(p[-1,:3,3],[120,4,0]);self.assertGreater(audit['maximum_relative_heading_deg'],7)
        self.assertLess(audit['maximum_acceleration_mps2'],3)
        velocity=np.gradient(p[:,:3,3],.1,axis=0,edge_order=2);expected=velocity/np.linalg.norm(velocity,axis=1)[:,None]
        np.testing.assert_allclose(p[:,:3,0],expected,atol=1e-8)
        self.assertTrue(all(audit['checks'].values()));json.dumps(audit,allow_nan=False)
    def test_onset_and_recovery_have_no_pose_or_heading_jump(self):
        task,audit=build(self.source());p=np.array([w['ego_to_world'] for w in task['waypoints']])
        np.testing.assert_allclose(p[:6,1,3],0,atol=1e-10)
        np.testing.assert_allclose(p[35:,1,3],4,atol=1e-10)
        np.testing.assert_allclose(p[37:,:3,:3],np.repeat(np.eye(3)[None],24,axis=0),atol=1e-8)
        self.assertLess(np.max(abs(np.diff(np.array([r['relative_yaw_deg'] for r in audit['rows']])))),1.)
    def test_generation_is_deterministic_and_does_not_mutate_source(self):
        source=self.source();before=json.dumps(source);one=build(source);two=build(source)
        self.assertEqual(one,two);self.assertEqual(json.dumps(source),before)

if __name__=='__main__':unittest.main()
