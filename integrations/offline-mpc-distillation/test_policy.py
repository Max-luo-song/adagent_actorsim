
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
pass  # Path is provided by the portable boundary above
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from mpc_policy import Student,Provider,SCHEMA,control_bounds,family_distances,vehicle_hash,with_fallback


class PolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.vehicle=dict(min_acc_mps2=-2.,max_acc_mps2=2.,max_steer_rate_degrees=15.)

    def artifact(self,folder,threshold=.5):
        model=Student(4)
        with torch.no_grad():
            for parameter in model.parameters():parameter.zero_()
        state=dict(schema=SCHEMA,dimension=4,model=model.state_dict(),mean=[0.]*4,std=[1.]*4,
            training_descriptors=[[0.]*4],distance_threshold=threshold,trained_modes=[0],
            vehicle_identity=vehicle_hash(self.vehicle),feature_schema='fixture')
        path=Path(folder)/'policy.pt';torch.save(state,path);return path

    def test_reloaded_proposal_requires_full_current_check(self):
        with tempfile.TemporaryDirectory() as folder:
            provider=Provider(self.artifact(folder),encode=lambda p:np.zeros(4))
            seen=[]
            def evaluate(controls):
                seen.append(controls)
                return dict(controls=controls.tolist(),objective=3.,minimum_constraint=.2)
            problem=SimpleNamespace(vehicle=SimpleNamespace(**self.vehicle),evaluate=evaluate)
            decision=provider(problem)
            self.assertIsNotNone(decision['selected']);self.assertEqual(len(seen),1)
            self.assertTrue(decision['audit']['full_current_constraints_checked'])
            self.assertFalse(decision['audit']['objective_optimality_guaranteed'])

    def test_obstacle_failure_falls_back_without_substitute(self):
        with tempfile.TemporaryDirectory() as folder:
            provider=Provider(self.artifact(folder),encode=lambda p:np.zeros(4))
            problem=SimpleNamespace(vehicle=SimpleNamespace(**self.vehicle),evaluate=lambda u:None)
            def optimize(reference,occupancy,info,**kwargs):
                decision=kwargs['_proposal_provider'](problem)
                self.assertIsNone(decision['selected'])
                return dict(selected='original_full_mpc',audit=decision['audit'])
            result=with_fallback(optimize,provider)(None,None,None)
            self.assertEqual(result['selected'],'original_full_mpc')
            self.assertEqual(result['audit']['reason'],'no_current_feasible_quality_proposal')

    def test_out_of_distribution_and_vehicle_fail_before_admission(self):
        with tempfile.TemporaryDirectory() as folder:
            problem=SimpleNamespace(vehicle=SimpleNamespace(**self.vehicle),evaluate=lambda u:self.fail('Must reject before admission'))
            provider=Provider(self.artifact(folder),encode=lambda p:np.full(4,10.))
            self.assertEqual(provider(problem)['audit']['reason'],'outside_training_support')
            problem.vehicle.min_acc_mps2=-3.
            self.assertEqual(provider(problem)['audit']['reason'],'vehicle_mismatch')

    def test_family_gate_excludes_augmented_siblings(self):
        features=np.array([[0.,0.],[.01,.01],[2.,2.]])
        distances=family_distances(features,[('episode',0),('episode',0),('episode',1)])
        self.assertGreater(distances[0],1.9)
        with self.assertRaises(ValueError):family_distances(features,[('e',0)]*3)

    def test_checkpoint_hash_and_physical_bounds(self):
        lo,hi=control_bounds(self.vehicle)
        self.assertEqual(lo.shape,(6,2));self.assertAlmostEqual(hi[0,1],np.pi/12)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):Provider(self.artifact(folder),expected_sha256='wrong')


if __name__=='__main__':unittest.main()
