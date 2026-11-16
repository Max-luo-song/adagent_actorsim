
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

import struct,tempfile,unittest
pass  # Path is provided by the portable boundary above
import numpy as np
from raw_metadata import fields,transform,records
class RawTests(unittest.TestCase):
 def test_unpacked_transform(self):
  matrix=np.eye(4);encoded=b''.join(b'\x09'+struct.pack('<d',v) for v in matrix.ravel());np.testing.assert_array_equal(transform(encoded),matrix)
 def test_packed_transform(self):
  encoded=b'\x0a\x80\x01'+np.eye(4).astype('<f8').tobytes();np.testing.assert_array_equal(transform(encoded),np.eye(4))
 def test_truncated_fields_rejected(self):
  for value in [b'\x10\x80',b'\x1a\x08short',b'\x09short',b'\x00']:
   with self.assertRaises(ValueError):fields(value)
 def test_tfrecord_bounds(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'record';p.write_bytes(struct.pack('<Q',3)+bytes(4)+b'abc'+bytes(4));self.assertEqual(list(records(p)),[(0,b'abc')]);p.write_bytes(p.read_bytes()[:-1]);self.assertRaises(ValueError,list,records(p))
if __name__=='__main__':unittest.main(verbosity=2)
