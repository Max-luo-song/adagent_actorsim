"""Behavior regressions for candidate tiers, geometry and brake recovery."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'waymo172_closed_loop_v1')]
from planning_guard import PlanningGuard,RecoveryController,rectangle_separation,occupancy_clearance,sweep_allowance
from runtime_configuration import load_vehicle_config
from closed172_common import integrate_state,optimize

V=load_vehicle_config()


def braking(speed=2.,steer=0.,offset=(0.,0.)):
    states=[np.array([-offset[0],-offset[1],0.,speed,steer])]
    for i in range(30):
        s=states[-1];a=0 if i<2 else -min(2.,s[3]/.1)
        states.append(integrate_state(s,[a,0.]))
    return np.asarray(states),np.arange(31)*.1


def proposal(x=0.,y=10.,state='provisional_uncertain',score=.24,velocity=(0.,0.)):
    return dict(candidate_index=1,raw_score=score,state=state,observations=3 if state=='persistent_uncertain' else 1,
        duration_seconds=.2 if state=='persistent_uncertain' else 0.,path_clearance_m=-1.,
        box=[x,y,-1.,1.8,4.,1.5,0.,*velocity])


def make_guard(obstacle,**kw):
    states,times=braking(**kw)
    return PlanningGuard(dict(detection_threshold=.4,config={},obstacles=[obstacle]),V,states,times)


def feasible(urgent=False):
    return dict(selected=dict(controls=np.tile([.5,0.],(6,1)).tolist()),guard=dict(
        recovery=dict(confirmation_seconds=.2,confirmation_observations=3,max_gap_seconds=.75,speed_limit_mps=2.),
        evidence_tiers=[dict(reason='braking_envelope_conflict')] if urgent else []))


class GeometryTests(unittest.TestCase):
    def test_side_clearance_uses_vehicle_width_and_rotation(self):
        state=np.array([[0.,0.,0.,0.,0.]])
        self.assertGreater(occupancy_clearance(state,[[2.,0.]],V)[0],0)
        state[0,2]=np.pi/2
        self.assertLess(occupancy_clearance(state,[[2.,0.]],V)[0],0)

    def test_rectangle_geometry_is_rigid_rotation_invariant(self):
        states=np.array([[0.,0.,.1,2.,0.],[.5,2.,.4,2.,0.]])
        boxes=np.array([[3.,5.,-1.,1.8,4.,1.5,.3,.4,-.2]])
        original=rectangle_separation(states,boxes,[0.,1.],.8,1.5)
        a=.73;c,s=np.cos(a),np.sin(a);rotation=np.array([[c,s],[-s,c]])
        states2=states.copy();states2[:,:2]=states[:,:2]@rotation.T+[7.,-2.];states2[:,2]+=a
        boxes2=boxes.copy();boxes2[:,:2]=boxes[:,:2]@rotation.T+[7.,-2.];boxes2[:,7:9]=boxes[:,7:9]@rotation.T;boxes2[:,6]+=a
        np.testing.assert_allclose(rectangle_separation(states2,boxes2,[0.,1.],.8,1.5),original,atol=1e-12)

    def test_rotating_in_place_still_has_swept_allowance(self):
        states=np.array([[0.,0.,0.,0.,0.],[0.,0.,np.pi/2,0.,0.]])
        self.assertTrue((sweep_allowance(states,V)>0).all())


class EvidenceTests(unittest.TestCase):
    def test_far_single_frame_candidate_is_soft_without_changing_score(self):
        o=proposal();g=make_guard(o)
        self.assertFalse(g.rows[0]['hard_constraint']);self.assertEqual(len(g.soft_boxes),1)
        self.assertEqual(g.speed_limit,2.);self.assertEqual(o['raw_score'],.24)

    def test_confirmed_and_retained_obstacles_remain_hard(self):
        for o in (proposal(state='persistent_uncertain'),proposal(state='network_retained',score=.8)):
            self.assertTrue(make_guard(o).rows[0]['hard_constraint'])
        o=proposal(state='persistent_uncertain');o['observations']=1
        self.assertFalse(make_guard(o).rows[0]['hard_constraint'])

    def test_single_frame_near_hazard_is_not_ignored(self):
        g=make_guard(proposal(y=4.))
        self.assertTrue(g.rows[0]['hard_constraint'])
        self.assertEqual(g.rows[0]['reason'],'braking_envelope_conflict')

    def test_dynamic_future_conflict_is_not_current_overlap(self):
        g=make_guard(proposal(x=5.,y=0.,velocity=(-2.,0.)),speed=0.)
        self.assertTrue(g.rows[0]['hard_constraint'])
        states=np.zeros((2,5))
        gaps=g.clearance(states,np.array([0.,2.]))
        self.assertGreater(gaps[0],0);self.assertLess(gaps[1],0)

    def test_172_trigger_is_unconfirmed_but_not_imminent(self):
        o=proposal(x=2.9624252319335938,y=-.3020286560058594)
        o['box']=[2.9624252319335938,-.3020286560058594,.909846305847168,1.7462644577026367,4.1137776374816895,1.549096703529358,-3.061849594116211,.0004339513834565878,-.0010984382824972272]
        g=make_guard(o,speed=2.0970493680343683,steer=.3,offset=(0.,.985793))
        self.assertFalse(g.rows[0]['hard_constraint']);self.assertGreater(g.rows[0]['braking_envelope_clearance_m'],0)

    def test_raw_occupancy_collision_still_stops_before_solver(self):
        occupancy=np.zeros((1,5,200,200));occupancy[0,:,100,100]=1
        info=dict(ego_velo=1.,ego_steer=0.,model_lidar_frame=dict(virtual_to_ego=np.eye(4)))
        result=optimize(np.zeros((6,2)),occupancy,info)
        self.assertIsNone(result['selected']);self.assertEqual(result['reason'],'fixed_initial_occupancy_conflict')

    def test_low_confidence_actual_overlap_still_stops(self):
        # The overlapping centre stays OUTSIDE the ego footprint (y=2.5 > 1.5):
        # a physically possible real object is never escaped from.  Only a
        # centre inside the footprint (impossible without an ongoing collision)
        # activates the v28 non-worsening escape; see phase13/过程记录.md.
        occupancy=np.zeros((1,5,200,200));occupancy[0,:,190,190]=1
        info=dict(ego_velo=1.,ego_steer=0.,model_lidar_frame=dict(virtual_to_ego=np.eye(4)))
        result=optimize(np.zeros((6,2)),occupancy,info,reliability=dict(config={},obstacles=[proposal(y=2.5)]))
        self.assertIsNone(result['selected']);self.assertTrue(result['initial_conflict']['introduced_by_uncertainty_guard'])

    def test_margin_is_one_budget_not_sum(self):
        o=proposal(state='persistent_uncertain')
        g=make_guard(o);self.assertEqual(g.rows[0]['effective_margin_m'],.5)
        states,times=braking()
        bigger=replace(V,clearance_margin_m=.7)
        g=PlanningGuard(dict(config={'obstacle_margin_m':.5},obstacles=[o]),bigger,states,times)
        self.assertEqual(g.rows[0]['effective_margin_m'],.7)


class RecoveryTests(unittest.TestCase):
    def test_recovery_requires_continuous_feasible_evidence(self):
        c=RecoveryController();bad=dict(selected=None,reason='blocked')
        a,d=c.update(bad,1.,0.,.1,2.);self.assertEqual(a['acc'],-2.)
        for t in (.1,.2):
            a,d=c.update(feasible(),0.,t,.1,2.);self.assertEqual(a['acc'],0.)
        c.update(bad,0.,.3,.1,2.)
        for t in (.4,.5):
            a,d=c.update(feasible(),0.,t,.1,2.);self.assertEqual(a['acc'],0.)
        a,d=c.update(feasible(),0.,.6,.1,2.)
        self.assertEqual(d['state_after'],'recovering');self.assertEqual(a['acc'],.5)
        self.assertEqual(c.speed_cap({}),2.)
        c.update(feasible(),.1,.8,.1,2.);self.assertEqual(c.state,'normal')

    def test_caution_tracks_current_evidence_and_does_not_latch(self):
        c=RecoveryController();result=feasible();result['guard']['speed_limit_mps']=2.
        a,d=c.update(result,1.,0.,.1,2.)
        self.assertEqual(d['state_after'],'caution');self.assertEqual(a['acc'],.5)
        self.assertIsNone(c.speed_cap({}))
        a,d=c.update(feasible(),1.,.1,.1,2.)
        self.assertEqual(d['state_after'],'normal')

    def test_urgent_candidate_brakes_even_if_a_plan_is_feasible(self):
        c=RecoveryController();a,d=c.update(feasible(urgent=True),1.,0.,.1,2.)
        self.assertEqual(a['acc'],-2.);self.assertEqual(d['state_after'],'braking')

    def test_verified_alternative_is_not_overridden_by_stop_envelope(self):
        c=RecoveryController();result=feasible(urgent=True)
        result['selected'].update(feasible=True,minimum_constraint=-1e-8)
        result['selected']['controls']=np.tile([-.8,.03],(6,1)).tolist()
        action,decision=c.update(result,.5,0.,.1,2.)
        self.assertAlmostEqual(action['acc'],-.8)
        self.assertAlmostEqual(action['steer_rate'],.03)
        self.assertEqual(decision['reason'],'feasible_plan')

    def test_urgent_unverified_or_invalid_alternative_still_brakes(self):
        for values in ({'feasible':False,'minimum_constraint':0.},
                       {'feasible':True,'minimum_constraint':-.01},
                       {'feasible':True,'minimum_constraint':float('nan')}):
            result=feasible(urgent=True);result['selected'].update(values)
            action,decision=RecoveryController().update(result,.5,0.,.1,2.)
            self.assertEqual(action['acc'],-2.)
            self.assertEqual(decision['state_after'],'braking')

    def test_time_discontinuity_cannot_release_stop(self):
        c=RecoveryController();c.update(dict(selected=None),0.,0.,.1,2.)
        c.update(feasible(),0.,.1,.1,2.);c.update(feasible(),0.,.2,.1,2.)
        a,d=c.update(feasible(),0.,2.,.1,2.)
        self.assertEqual(a['acc'],0.);self.assertEqual(d['clear_observations'],1)

if __name__=='__main__':unittest.main()
