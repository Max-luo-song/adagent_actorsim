
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

import copy,json,tempfile,unittest
pass  # Path is provided by the portable boundary above
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from bbox_diagnostics import box_corners,target_corners,projected_segments,compare,planner_shapes
from proposal_geometry_audit import geometry_receipt,install
from stage_results import finish


class BoxTests(unittest.TestCase):
    def test_clockwise_bottom_center_corners(self):
        b=[2,10,-2,2,4,3,np.pi/2,0,0];p=box_corners(b)
        np.testing.assert_allclose(p[0],[-0.,11,-2],atol=1e-12)
        np.testing.assert_allclose(p[:4,:2].mean(0),b[:2])
        np.testing.assert_allclose(p[4:,2],1)
    def test_native_reference_transform_and_bottom(self):
        ref=dict(placements=[dict(asset_to_native=np.eye(4).tolist())],asset_wlh=[2,4,3],local_bottom_y=1,
            geometry_contract={'schema':'asset_geometry_x_longitudinal_y_down_z_lateral_wlh_v1'})
        info=dict(explicit_l2g_t=[1,2,3],explicit_l2g_r_mat=np.eye(3))
        p=target_corners(ref,info)
        np.testing.assert_allclose(p[:4].mean(0),[-1,-2,-4])
        np.testing.assert_allclose(p[4:].mean(0),[-1,-2,-1])
    def test_camera_projection_near_plane_and_image_clip(self):
        camera=dict(l2c=np.eye(4),intrinsic=dict(W=100,H=100,cx=50,cy=50,fovx=np.pi/2,fovy=np.pi/2))
        self.assertEqual(projected_segments(box_corners([0,0,-5,1,1,1,0]),camera),[])
        segments=projected_segments(box_corners([0,0,-.5,2,2,2,0]),camera)
        self.assertTrue(segments)
        for a,b in segments:
            self.assertTrue(np.isfinite([a,b]).all());self.assertGreaterEqual(np.min([a,b]),-1e-8);self.assertLessEqual(np.max([a,b]),99+1e-8)
    def test_matching_separates_low_score_and_retained_and_gate(self):
        target=box_corners([0,20,0,2,4,2,0])
        boxes=np.array([[0,20,0,2,4,2,0,0,0],[5,20,0,2,4,2,0,0,8]],float)
        r=compare(target,boxes,np.array([.06,.8]),np.array([0,0]))
        self.assertEqual(r['nearest_candidate']['candidate_index'],0)
        self.assertTrue(r['nearest_candidate']['within_proximity_gate'])
        self.assertFalse(r['nearest_retained_detection']['within_proximity_gate'])
        self.assertEqual(r['nearest_retained_detection']['static_target_speed_error_mps'],8)
        self.assertFalse(r['identity_confirmed'])
    def test_nonvehicle_and_empty_candidates_do_not_match(self):
        target=box_corners([0,20,0,2,4,2,0])
        r=compare(target,np.empty((0,9)),np.array([]),np.array([]));self.assertIsNone(r['nearest_candidate'])
        r=compare(target,np.array([[0,20,0,2,4,2,0,0,0]]),np.array([.9]),np.array([8]));self.assertIsNone(r['nearest_candidate'])
    def guard(self):
        return SimpleNamespace(rows=[dict(candidate_index=3,hard_constraint=True,geometry='heading_free_disk_motion_envelope',evidence_state='network_retained',raw_score=.8)],hard_boxes=np.array([[1,20,-1,2,4,2,0,0,3.]]),soft_boxes=np.empty((0,9)),hard_margins=np.array([.5]),soft_margins=np.array([]),hard_envelopes=np.array([True]),hard_oriented=np.array([False]),hard_radii=np.array([2.3]),hard_ages=np.array([.1]),hard_heading_ranges=np.array([0.]))
    def test_receipt_copies_actual_arrays_without_mutation(self):
        g=self.guard();before=copy.deepcopy(g.__dict__);r=geometry_receipt(g)
        r['rows'][0]['box'][0]=999
        for k,v in before.items():
            if isinstance(v,np.ndarray):np.testing.assert_array_equal(v,getattr(g,k))
            else:self.assertEqual(v,getattr(g,k))
        self.assertFalse(r['ground_truth_used'])
    def test_audit_wrapper_idempotent_and_preserves_output(self):
        class Guard:
            def audit(self):return {'original':True}
        install(Guard);first=Guard.audit;install(Guard);self.assertIs(Guard.audit,first)
        g=Guard();g.__dict__.update(self.guard().__dict__);self.assertTrue(g.audit()['original'])
    def test_soft_receipt_preserves_appended_demoted_proposal_order(self):
        g=self.guard();row=dict(candidate_index=9,hard_constraint=False,geometry='reported_rectangle',evidence_state='provisional_uncertain',raw_score=.06)
        demoted=dict(row,candidate_index=8,existence_promoted_by_risk=False)
        g.rows=[g.rows[0],demoted,row];g.soft_boxes=np.array([[9,20,-1,2,4,2,0,0,0],[8,20,-1,2,4,2,0,0,0]],float);g.soft_margins=np.array([.5,.5])
        rows=geometry_receipt(g)['rows'][1:]
        self.assertEqual([r['candidate_index'] for r in rows],[9,8]);self.assertEqual([r['box'][0] for r in rows],[9,8])
    def test_historical_fallback_explicit_and_receipt_preferred(self):
        rows,source=planner_shapes({'guard':{'evidence_tiers':[]}},{});self.assertIn('historical',source)
        receipt=geometry_receipt(self.guard());rows,source=planner_shapes({'guard':{'proposal_geometry_receipt':receipt}},{});self.assertEqual(source,'actual_guard_arrays');self.assertEqual(rows[0]['radius_m'],2.3)
    def test_expired_memory_is_distinct_risk_with_serialized_backup(self):
        g=self.guard();history=copy.copy(g)
        row=dict(g.rows[0],hard_constraint=False,expired_memory_risk_only=True,missing_seconds=2.)
        g.rows=[row];g.expired_rows=[row];g.expired_history=history
        for name in ('hard_boxes','hard_margins','hard_envelopes','hard_oriented','hard_radii','hard_ages','hard_heading_ranges'):
            setattr(g,name,getattr(g,name)[:0])
        r=geometry_receipt(g)['rows'];self.assertEqual(len(r),1);self.assertEqual(r[0]['tier'],'history_risk')
        self.assertTrue(r[0]['current_action_backup_retained']);self.assertEqual(r[0]['backup_geometry']['radius_m'],2.3)
    def test_required_diagnostic_failure_preserves_driving_status(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'comparison_frames').mkdir();(root/'comparison_frames/000.png').write_bytes(b'fake')
            (root/'run_report.json').write_text(json.dumps(dict(status='completed',integration_acceptance_passed=True,bbox_diagnostic_required=True,trace=[],observations=[])))
            (root/'integration_acceptance.json').write_text('{"passed":true}')
            with patch('bbox_diagnostics.analyze',side_effect=ValueError('bad convention')),patch('stage_results.subprocess.run',side_effect=RuntimeError('fake encoder')):
                r=finish(root,encoder='fake')
            self.assertEqual(r['bbox_diagnostic_status'],'failed');self.assertTrue(r['driving_acceptance_passed']);self.assertEqual(r['exit_code'],1)
    def test_new_validation_rejects_missing_actual_planner_receipts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'comparison_frames').mkdir();(root/'comparison_frames/000.png').write_bytes(b'fake')
            (root/'run_report.json').write_text(json.dumps(dict(status='completed',integration_acceptance_passed=True,bbox_diagnostic_required=True,proposal_geometry_receipt_enabled=True,trace=[{}],observations=[])))
            (root/'integration_acceptance.json').write_text('{"passed":true}')
            boxes=dict(passed=True,processed_frames=1,planner_receipt_frames=0,proximity_counts={})
            def encode(args,**kwargs):
                if '-c:v' in args:Path(args[-1]).write_bytes(b'fake video')
            with patch('bbox_diagnostics.analyze',return_value=boxes),patch('stage_results.subprocess.run',side_effect=encode):
                r=finish(root,encoder='fake')
            self.assertEqual(r['postprocess_status'],'completed');self.assertEqual(r['bbox_diagnostic_status'],'failed');self.assertFalse(r['bbox_diagnostic']['required_receipts_present']);self.assertEqual(r['exit_code'],1)


if __name__=='__main__':unittest.main()
