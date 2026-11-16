
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

import copy,pickle,sys,unittest
pass  # Path is provided by the portable boundary above
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from global_repair_v1.driver import ContractObstacleMemory,validate_guard_input
from global_repair_v1.velocity import apply

class DriverContractTests(unittest.TestCase):
    def test_driver_memory_has_stable_pickle_identity(self):
        memory=ContractObstacleMemory()
        restored=pickle.loads(pickle.dumps(memory,protocol=4))
        self.assertIs(type(restored),ContractObstacleMemory)
        self.assertEqual(vars(memory),vars(restored))

    def test_guard_rejects_missing_stale_or_pre_memory_receipt(self):
        info=dict(scene_token='episode',timestamp=1.)
        receipt=dict(stage='after_driver_memory',episode='episode',timestamp=1.)
        validate_guard_input(dict(global_evidence_receipt=receipt),info)
        for report in ({},dict(global_evidence_receipt=dict(receipt,stage='before_driver_memory')),
                       dict(global_evidence_receipt=dict(receipt,timestamp=.9))):
            with self.assertRaises(ValueError):validate_guard_input(report,info)

    def test_velocity_override_preserves_original_neural_evidence(self):
        box=[0.,0.,0.,4.,2.,2.,0.,3.,0.]
        obstacle=dict(box=list(box),neural_box=list(box),candidate_index=0)
        report=dict(obstacles=[obstacle],observed_depth_obstacles_xy=[[20.,20.]],
                    observed_depth_geometry=dict(voxel_resolution_m=.5))
        result=apply(report,{})
        self.assertEqual(result['obstacles'][0]['neural_box'],box)
        self.assertEqual(result['obstacles'][0]['guard_motion_box'][7:9],[0.,0.])
        np.testing.assert_array_equal(result['obstacles'][0]['box'][:7],box[:7])

if __name__=='__main__':unittest.main()
