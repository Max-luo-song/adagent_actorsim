
# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand
Path = _HugsimPath

import copy
import json
pass  # Path is provided by the portable boundary above
import pickle
import sys
import unittest
import numpy as np
HERE=Path(__file__).resolve().parent
P28=HERE.parent
ROOT=P28.parents[5]
sys.path[:0]=[str(HERE),str(P28),str(ROOT/'waymo23-native-import')]
from policy import filter_report

class PolicyTests(unittest.TestCase):
    def test_prefix_failure_is_not_reported_as_physical_safety_failure(self):
        from postprocess import failure_flags
        before=_hugsim_expand(json.loads((P28/'runs/lineage92_closed_scene552_v1/run_report.json').read_text()))
        self.assertEqual(failure_flags(before),dict(execution_failed=True,safety_failure=False))
        self.assertEqual(failure_flags(dict(status='failed',error='Interpolated actor reference overlap after control')),
            dict(execution_failed=True,safety_failure=True))
    def test_archived_optimizer_is_reused_only_after_exact_input_and_guard_gate(self):
        from prefix import bind_optimizer,archived_packet,BASELINE
        _,info=archived_packet(0);folder=BASELINE/'uniad_worker/step_000'
        with np.load(folder/'perception.npz') as data:plan=data['trajectory'].copy();occ=data['occupancy'].copy()
        rel=_hugsim_expand(json.loads((folder/'guard_memory_report.json').read_text()));rel['lineage92_counterfactual']={}
        called=[];optimize=bind_optimizer(lambda *a,**kw:called.append(True))
        result=optimize(plan,occ,info,reliability=rel)
        self.assertTrue(result['prefix_replay_audit']['raw_inputs_exactly_verified']);self.assertFalse(called)
        broken=copy.deepcopy(rel);broken['speed_limit_mps']=999
        with self.assertRaises(RuntimeError):optimize(plan,occ,info,reliability=broken)
        with self.assertRaises(AssertionError):optimize(plan+1,occ,info,reliability=rel)
        _,live_info=archived_packet(39);optimize(plan,occ,live_info,reliability=rel)
        self.assertEqual(called,[True])

    def test_real_v2_guard_roundoff_and_material_changes(self):
        from prefix import verify_guard,bind_optimizer,archived_packet,BASELINE
        old=BASELINE/'uniad_worker/step_000'
        failed=P28/'runs/lineage92_closed_scene552_v2/uniad_worker/step_000'
        expected=_hugsim_expand(json.loads((old/'guard_memory_report.json').read_text()))
        actual=_hugsim_expand(json.loads((failed/'guard_memory_report.json').read_text()))
        actual.pop('lineage92_counterfactual',None)
        audit=verify_guard(actual,expected)
        self.assertGreater(audit['maximum_absolute_difference'],0)
        self.assertLess(audit['maximum_absolute_difference'],1e-7)
        _,info=archived_packet(0)
        with np.load(old/'perception.npz') as d:
            result=bind_optimizer(lambda *a,**k:self.fail('Unexpected live optimization'))(
                d['trajectory'],d['occupancy'],info,reliability=actual)
        self.assertEqual(result['prefix_replay_audit']['guard_numeric_verification'],audit)
        broken=copy.deepcopy(actual);broken['obstacles'][4]['box'][0]+=1e-5
        with self.assertRaises(RuntimeError):verify_guard(broken,expected)
        broken=copy.deepcopy(actual);broken['obstacles'][4]['candidate_index']+=1
        with self.assertRaises(RuntimeError):verify_guard(broken,expected)
        broken=copy.deepcopy(actual);broken['obstacles'][4]['raw_score']+=1e-5
        with self.assertRaises(RuntimeError):verify_guard(broken,expected)
        broken=copy.deepcopy(actual);broken['obstacles'][4]['box'][0]=float('nan')
        with self.assertRaises(RuntimeError):verify_guard(broken,expected)
        broken=copy.deepcopy(actual);broken['new_field']=False
        with self.assertRaises(RuntimeError):verify_guard(broken,expected)

    def test_real_v3_memory_roundoff_and_strict_memory_metadata(self):
        from prefix import verify_guard,BASELINE
        current=P28/'runs/lineage92_closed_scene552_v3/uniad_worker/step_001'
        expected=_hugsim_expand(json.loads((BASELINE/'uniad_worker/step_001/guard_memory_report.json').read_text()))
        actual=_hugsim_expand(json.loads((current/'guard_memory_report.json').read_text()));actual.pop('lineage92_counterfactual',None)
        audit=verify_guard(actual,expected)
        self.assertGreater(audit['maximum_absolute_difference'],0)
        self.assertLess(audit['maximum_absolute_difference'],1e-7)
        bad=copy.deepcopy(actual);bad['obstacle_memory']['constraints'][4]['box'][0]+=1e-5
        with self.assertRaises(RuntimeError):verify_guard(bad,expected)
        bad=copy.deepcopy(actual);bad['obstacle_memory']['tracks']+=1
        with self.assertRaises(RuntimeError):verify_guard(bad,expected)

    def test_derived_motion_tolerance_preserves_direct_geometry_and_decisions(self):
        from prefix import verify_guard
        expected=dict(obstacles=[dict(box=[1.0],geometry_motion_evidence=dict(fit_velocity_disagreement_m=0.5,validated=False))])
        actual=copy.deepcopy(expected);actual['obstacles'][0]['geometry_motion_evidence']['fit_velocity_disagreement_m']+=5e-7
        self.assertLess(verify_guard(actual,expected)['maximum_absolute_difference'],1e-6)
        bad=copy.deepcopy(actual);bad['obstacles'][0]['geometry_motion_evidence']['validated']=True
        with self.assertRaises(RuntimeError):verify_guard(bad,expected)
        bad=copy.deepcopy(expected);bad['obstacles'][0]['box'][0]+=5e-7
        with self.assertRaises(RuntimeError):verify_guard(bad,expected)
        bad=copy.deepcopy(expected);bad['obstacles'][0]['geometry_motion_evidence']['fit_velocity_disagreement_m']+=2e-6
        with self.assertRaises(RuntimeError):verify_guard(bad,expected)

    def test_archived_packet_requires_correct_feedback_state_then_switches_live(self):
        from prefix import bind_packet,archived_packet
        obs,info=archived_packet(0)
        rendered=dict(cameras={name:dict(rgb=np.zeros_like(rgb)) for name,rgb in obs['rgb'].items()})
        packet=bind_packet(lambda r:(dict(rgb={}),copy.deepcopy(info)))
        old_obs,old_info=packet(rendered)
        for name,rgb in obs['rgb'].items():np.testing.assert_array_equal(rendered['cameras'][name]['rgb'],rgb)
        self.assertEqual(old_info['scene_token'],info['scene_token'])
        bad=copy.deepcopy(info);bad['ego_velo']+=.1
        with self.assertRaises(RuntimeError):bind_packet(lambda r:({},bad))(rendered)
        live=copy.deepcopy(info);live.update(timestamp=12.,scene_token='new-runtime-token')
        live_obs=dict(rgb={'marker':123})
        after,metadata=bind_packet(lambda r:(live_obs,live))(dict())
        self.assertIs(after,live_obs);self.assertEqual(metadata['scene_token'],info['scene_token'])
    def test_actual_step39_removes_only_neural92_without_mutating_raw(self):
        run=P28/'runs/candidate_origin_20261102T044132Z/baseline'
        report=_hugsim_expand(json.loads((run/'scene_reliability.json').read_text()));before=copy.deepcopy(report)
        query=_hugsim_expand(json.loads((run/'neural_query_origin.json').read_text()))
        result=filter_report(report,query['selected_track_ids'])
        self.assertEqual(report,before)
        self.assertEqual([o['candidate_index'] for o in result['lineage92_counterfactual']['removed']],[33])
        self.assertEqual(result['obstacles'],[o for o in before['obstacles'] if o['candidate_index']!=33])

    def test_does_not_remove_nearby_objects_with_other_ids(self):
        r=dict(obstacles=[dict(candidate_index=0,raw_score=.9,box=[0,0],state='network_retained'),
            dict(candidate_index=1,raw_score=.9,box=[0,0],state='network_retained')],config=dict(caution_speed_mps=2))
        filtered=filter_report(r,[72,92])
        self.assertEqual([o['candidate_index'] for o in filtered['obstacles']],[0])

    def test_rejects_filtering_after_memory_and_keeps_other_caution(self):
        with self.assertRaises(ValueError):filter_report(dict(obstacles=[dict(candidate_index=-89)]),[92])
        r=dict(obstacles=[dict(candidate_index=0,raw_score=.4,box=[],state='network_retained'),
            dict(candidate_index=1,raw_score=.3,box=[],state='provisional_uncertain',path_clearance_m=-1)],
            config=dict(caution_speed_mps=2))
        result=filter_report(r,[92,93]);self.assertEqual(result['speed_limit_mps'],2)
        self.assertEqual(result['action'],'caution')

    def test_actual_memory_prefix_does_not_create_removed_obstacle(self):
        from planning_guard import ObstacleMemory
        from heading_fix_v1 import closed172_common as cc
        original=P28/'runs/heading_fix_scene552_v1';replay=P28/'runs/candidate_origin_20261102T044132Z'
        memory=ObstacleMemory();original_memory=ObstacleMemory()
        for step in range(40):
            folder=original/f'uniad_worker/step_{step:03d}'
            with (folder/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
            raw=_hugsim_expand(json.loads((folder/'scene_reliability.json').read_text()))
            qpath=replay/f'worker/step_{step:03d}/neural_query_origin.json' if step<39 else replay/'baseline/neural_query_origin.json'
            ids=_hugsim_expand(json.loads(qpath.read_text()))['selected_track_ids']
            filtered=filter_report(raw,ids)
            with np.load(folder/'perception.npz') as d:
                result=memory.update(filtered,d['geometry_candidate_boxes'],d['candidate_scores'],d['candidate_labels'],info,current_occupancy=d['occupancy'].reshape(5,200,200)[0])
                original_memory.update(raw,d['geometry_candidate_boxes'],d['candidate_scores'],d['candidate_labels'],info,current_occupancy=d['occupancy'].reshape(5,200,200)[0])
        baseline=_hugsim_expand(json.loads((folder/'guard_memory_report.json').read_text()))
        self.assertEqual(result['obstacles'],[o for o in baseline['obstacles'] if o['candidate_index']!=33])
        self.assertEqual(memory.next_id,original_memory.next_id-1)

    def test_return_measurement_requires_pass_and_sustained_heading(self):
        from postprocess import outcome
        from heading_fix_v1.closed172_common import load_vehicle_config
        run=P28/'runs/heading_fix_scene552_v1'
        report=_hugsim_expand(json.loads((run/'run_report.json').read_text()));asset=_hugsim_expand(json.loads((run/'asset_reference.json').read_text()))
        result=outcome(report,asset,load_vehicle_config(ROOT/'waymo23-native-import/configs/vehicle.json'))
        self.assertFalse(result['passed']);self.assertFalse(result['returned'])
        self.assertLess(result['last']['speed'],.001)

    def test_pass_and_return_positive_case_requires_half_second_hold(self):
        from postprocess import outcome,T
        from heading_fix_v1.closed172_common import load_vehicle_config,asset_reference_vertices
        run=P28/'runs/heading_fix_scene552_v1'
        original=_hugsim_expand(json.loads((run/'run_report.json').read_text()));asset=_hugsim_expand(json.loads((run/'asset_reference.json').read_text()))
        v=load_vehicle_config(ROOT/'waymo23-native-import/configs/vehicle.json')
        first=original['observations'][0];pose=T@np.array(first['closed_ego_to_world'])@T.T
        forward=pose[[0,2],2];forward=forward/np.linalg.norm(forward)
        far=float(np.max(asset_reference_vertices(asset)[:,[0,2]]@forward))
        pose[[0,2],3]+=forward*(far-float(pose[[0,2],3]@forward)+v.half_length_m+2)
        clear=T.T@pose@T
        obs=[dict(first)];trace=[]
        for i in range(7):
            trace.append(dict(step=i,state_before=[0,0,0,1,0],state_after=[0,0,0,1,0]))
            if i:obs.append(dict(first,index=i,timestamp=first['timestamp']+i*.1,closed_ego_to_world=clear.tolist(),closed_speed_mps=1.))
        self.assertFalse(outcome(dict(observations=obs[:2],trace=trace),asset,v)['returned'])
        result=outcome(dict(observations=obs,trace=trace),asset,v)
        self.assertTrue(result['passed']);self.assertTrue(result['returned'])

if __name__=='__main__':unittest.main()
