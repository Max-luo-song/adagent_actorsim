
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

import copy,json,sys,unittest
pass  # Path is provided by the portable boundary above
HERE=Path(__file__).resolve().parent;P28=HERE.parent;ROOT=P28.parents[5]
sys.path[:0]=[str(HERE),str(ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad')]
from admission import apply
class AdmissionTests(unittest.TestCase):
    def data(self):
        report=dict(config=dict(caution_speed_mps=2.),speed_limit_mps=1.,action='caution',obstacles=[
            dict(candidate_index=0,label=0,state='network_retained',raw_score=.41,box=[0.,0.,0.,2.,4.,2.,0.,0.,0.]),
            dict(candidate_index=1,label=8,state='network_retained',raw_score=.69624,box=[4.,8.,0.,.5,.5,1.7,0.,0.,0.])],
            observed_depth_obstacles_xy=[[1.,2.]],unknown_visibility_obstacles_xy=[[3.,4.]])
        info=dict(timestamp=1.,scene_token='any-scene')
        evidence=dict(timestamp=1.,scene_token='any-scene',queries=[dict(index=0,neural_id=123,untrusted_geometry=True,protected_observation=False,motion_innovation_m=.8,history_observations=3,history_seconds=.2),dict(index=1,neural_id=456,untrusted_geometry=False,protected_observation=True)])
        return report,evidence,info
    def test_only_external_untrusted_geometry_is_quarantined(self):
        report,evidence,info=self.data();before=copy.deepcopy(report);e=copy.deepcopy(evidence)
        out=apply(report,evidence,[123,456],info)
        self.assertEqual(report,before);self.assertEqual(evidence,e)
        self.assertEqual(out['obstacles'],[report['obstacles'][1]])
        self.assertEqual(out['obstacles'][0]['raw_score'],.69624)
        self.assertEqual(out['speed_limit_mps'],1.)
        self.assertEqual(out['observed_depth_obstacles_xy'],report['observed_depth_obstacles_xy'])
        self.assertEqual(out['unknown_visibility_obstacles_xy'],report['unknown_visibility_obstacles_xy'])
        self.assertFalse(out['proposal_admission']['free_space_asserted'])
    def test_positive_or_inconclusive_evidence_retains_original_obstacle(self):
        for protected,untrusted in [(True,True),(False,False)]:
            r,e,i=self.data();e['queries'][0].update(protected_observation=protected,untrusted_geometry=untrusted)
            self.assertEqual(apply(r,e,[123,456],i)['obstacles'],r['obstacles'])
    def test_stage_clock_and_id_mismatches_are_rejected(self):
        r,e,i=self.data();r['obstacles'][0]['candidate_index']=-1
        with self.assertRaises(ValueError):apply(r,e,[123,456],i)
        r,e,i=self.data();e['timestamp']=.9
        with self.assertRaises(ValueError):apply(r,e,[123,456],i)
        r,e,i=self.data()
        with self.assertRaises(ValueError):apply(r,e,[999,456],i)
    def test_geometry_roundoff_does_not_relax_score_protection(self):
        from screening import same_obstacle
        r,e,i=self.data();original=r['obstacles'][1];slight=copy.deepcopy(original);slight['box'][0]+=4e-9
        self.assertTrue(same_obstacle(original,slight))
        changed=copy.deepcopy(original);changed['raw_score']-=1e-6
        self.assertFalse(same_obstacle(original,changed))
        changed=copy.deepcopy(original);changed['box'][0]+=1e-5
        self.assertFalse(same_obstacle(original,changed))

    def test_real_failed_pedestrian_score_is_preserved(self):
        source=P28/'runs/proposal_evidence_inference_20261103T032120Z/baseline/step_025'
        replay=P28/'runs/proposal_admission_replay_20261103T043518Z/admission/step_025'
        baseline=_hugsim_expand(json.loads((source/'scene_reliability.json').read_text()));new=_hugsim_expand(json.loads((replay/'scene_reliability.json').read_text()))
        old=next(r for r in baseline['obstacles'] if r['candidate_index']==5);same=next(r for r in new['obstacles'] if r['candidate_index']==5)
        self.assertEqual(same,old);self.assertGreater(same['raw_score'],.69)
if __name__=='__main__':unittest.main()
