"""Retain RGB evidence after the harness deletes consumed model packets."""

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


class InputReceiptRecorder:
    def __init__(self, writer, output):
        self.writer = writer
        self.output = Path(output)
        self.receipts = {}

    def __call__(self, path, value):
        self.writer(path, value)
        path = Path(path)
        if path.name != 'observation.pkl' or not path.is_relative_to(
                self.output / 'closed_loop/uniad_worker'):
            return
        obs, info = value
        self.receipts[path.parent.name] = dict(timestamp=float(info['timestamp']),
            rgb_sha256={name: hashlib.sha256(image.tobytes()).hexdigest()
                        for name, image in obs['rgb'].items()})
        (self.output / 'weather_input_receipts.json').write_text(json.dumps(self.receipts, indent=2))
