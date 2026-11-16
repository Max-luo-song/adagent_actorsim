from pathlib import Path
import sys,json,tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from asset_calibration import fit_support,RULE,CALIBRATION_SCHEMA,fingerprint,load,rebase_report
from asset_geometry import SCHEMA,declared_contract

class CalibrationTests(unittest.TestCase):
    def points(self):
        rng=np.random.default_rng(7)
        return rng.uniform([-2.,-1.5,-.8],[2.,0.,.8],size=(1000,3))
    def test_fit_is_scene_and_prediction_independent_and_handles_bad_input(self):
        x=self.points();a=fit_support(x,np.ones(len(x)))
        b=fit_support(x[::-1],np.ones(len(x)))
        self.assertEqual(a,b)
        self.assertGreater(a['wlh'][1],a['wlh'][0])
        for points,opacity in [(x,np.zeros(len(x))),(x,np.full(len(x),2.))]:
            with self.assertRaises(ValueError):fit_support(points,opacity)
    def test_source_hash_and_geometry_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            asset=Path(temp);(asset/'gs.pth').write_bytes(b'asset');(asset/'wlh.json').write_text('[1,4,2]')
            support=fit_support(self.points(),np.ones(1000))
            m=dict(schema=CALIBRATION_SCHEMA,geometry_schema=SCHEMA,asset=str(asset),rule=RULE,
                   source_files={n:fingerprint(asset/n) for n in ('gs.pth','wlh.json')},support=support,
                   geometry_contract=declared_contract(support['wlh'],support['bottom_y']),
                   independent_physical_calibration=False,n1_certified=False,prediction_data_used=False,scene_data_used=False)
            path=asset/'calibration.json';path.write_text(json.dumps(m));self.assertEqual(load(path,asset),m)
            m['support']['wlh'][0]*=2;path.write_text(json.dumps(m))
            with self.assertRaises(ValueError):load(path,asset)
            (asset/'gs.pth').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'source changed'):load(path,asset)
    def test_rebased_reference_preserves_raw_geometry_world_positions(self):
        support=fit_support(self.points()+[1,0,2],np.ones(1000))
        m=dict(asset='/tmp/asset',support=support,geometry_contract=declared_contract(support['wlh'],support['bottom_y']))
        pose=np.eye(4);pose[:3,3]=[3,4,5]
        r=dict(asset='/tmp/asset',local_recenter_xz=[0,0],placements=[dict(asset_to_source=pose.tolist(),asset_to_native=pose.tolist())])
        out=rebase_report(r,m);raw=np.array([1.5,-.5,2.5,1.])
        centered=raw.copy();centered[[0,2]]-=support['raw_recenter_xz']
        np.testing.assert_allclose(np.asarray(out['placements'][0]['asset_to_source'])@centered,pose@raw,atol=1e-12)
        self.assertFalse(out['evaluation_admitted']);self.assertFalse(out['n1_edit_certified'])

    def test_render_report_preserves_shared_named_input_profile(self):
        from model_profiles import bind_sequence_model_profile
        for scene in ('172','552','unseen'):
            sequence=dict(scene=scene)
            bind_sequence_model_profile(dict(model_input_profile='calibration_geometry_v1'),sequence)
            self.assertEqual(sequence['model_input_profile'],'calibration_geometry_v1')
        self.assertEqual(bind_sequence_model_profile({},{}),{})
        with self.assertRaises(ValueError):bind_sequence_model_profile(dict(model_input_profile='scene_offset'),{})
