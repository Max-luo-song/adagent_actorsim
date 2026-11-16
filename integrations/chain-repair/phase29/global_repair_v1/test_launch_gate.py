
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

import sys,unittest
pass  # Path is provided by the portable boundary above
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from global_repair_v1.launch import validate_inference

class LaunchGateTests(unittest.TestCase):
    def report(self):
        return dict(complete=True,raw_neural_arrays_identical=True,normal_object_screen_passed=True,
            implementation_manifest_sha256='current',screening=dict(frames=81,
            visible_reference_observations=1203,visible_reference_changed_or_removed=0))
    def test_missing_or_failed_checks_do_not_start_closed_loop(self):
        with self.assertRaises(ValueError):validate_inference({},'current')
        for key,value in [('failed',True),('complete',False),('raw_neural_arrays_identical',False),
                          ('normal_object_screen_passed',False),('implementation_manifest_sha256','old')]:
            report=self.report();report[key]=value
            with self.assertRaises(ValueError):validate_inference(report,'current')
    def test_correct_completed_inference_can_start_diagnostic_not_certify_global_success(self):
        validate_inference(self.report(),'current')
        report=self.report();report['screening']['frames']=1
        with self.assertRaises(ValueError):validate_inference(report,'current')
if __name__=='__main__':unittest.main()
