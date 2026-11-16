
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

import hashlib
import json
pass  # Path is provided by the portable boundary above
import pickle
import tempfile
import unittest

import numpy as np
from input_receipts import InputReceiptRecorder


class ReceiptTests(unittest.TestCase):
    def test_consumed_packet_deletion_retains_original_rgb_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            def writer(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(pickle.dumps(value))
            recorder = InputReceiptRecorder(writer, out)
            image = np.full((4, 6, 3), 17, dtype=np.uint8)
            original_hash = hashlib.sha256(image.tobytes()).hexdigest()
            path = out / 'closed_loop/uniad_worker/step_001/observation.pkl'
            recorder(path, (dict(rgb={'CAM_FRONT': image}), dict(timestamp=8.1)))
            image[:] = 0
            path.unlink()
            saved = _hugsim_expand(json.loads((out / 'weather_input_receipts.json').read_text()))
            self.assertEqual(saved['step_001']['rgb_sha256']['CAM_FRONT'], original_hash)
            self.assertEqual(saved['step_001']['timestamp'], 8.1)
            recorder(out / 'sensor_metadata.pkl', {})
            self.assertEqual(len(recorder.receipts), 1)

    def test_failed_packet_write_creates_no_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            def failed(*args):
                raise OSError('simulated write failure')
            recorder = InputReceiptRecorder(failed, out)
            with self.assertRaises(OSError):
                recorder(out / 'closed_loop/uniad_worker/step_001/observation.pkl', None)
            self.assertFalse(recorder.receipts)
            self.assertFalse((out / 'weather_input_receipts.json').exists())


if __name__ == '__main__':
    unittest.main()
