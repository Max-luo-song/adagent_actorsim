
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

import json
pass  # Path is provided by the portable boundary above
import pickle
import tempfile
import unittest
import numpy as np
from postprocess_recover_v2 import recovered_metadata


class RecoveryTests(unittest.TestCase):
    def test_complete_metadata_is_linked_without_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            step=Path(tmp)/'source';step.mkdir();out=Path(tmp)/'derived.json'
            source=step/'sensor_metadata.json';source.write_text('{"original":true}')
            self.assertIsNone(recovered_metadata(step,out))
            self.assertTrue(out.is_symlink());self.assertEqual(source.read_text(),'{"original":true}')

    def test_terminal_metadata_uses_saved_mount_and_leaves_original_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            step=Path(tmp)/'step_069';step.mkdir();out=Path(tmp)/'derived.json'
            mount=np.eye(4);mount[1,3]=.985793
            packet=step/'observation.pkl'
            packet.write_bytes(pickle.dumps(({},dict(model_lidar_frame=dict(virtual_to_ego=mount)))))
            before=packet.read_bytes();meta=recovered_metadata(step,out)
            self.assertEqual(meta['step'],'step_069')
            np.testing.assert_array_equal(_hugsim_expand(json.loads(out.read_text()))['model_lidar_frame']['virtual_to_ego'],mount)
            self.assertEqual(packet.read_bytes(),before)
            self.assertFalse((step/'sensor_metadata.json').exists())

    def test_no_saved_input_fails_instead_of_inventing_or_omitting_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                recovered_metadata(Path(tmp),Path(tmp)/'derived.json')


if __name__=='__main__':unittest.main(verbosity=2)
