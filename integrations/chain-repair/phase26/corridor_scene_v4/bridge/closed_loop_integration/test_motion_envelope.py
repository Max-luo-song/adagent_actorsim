
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

import json,tempfile,unittest
pass  # Path is provided by the portable boundary above
from run_edited_closed_loop import read_task,validation_scope,HERE,PROFILES
class EnvelopeProfileTests(unittest.TestCase):
 def test_explicit_short_profile_selects_isolated_source(self):
  task=read_task(HERE/'examples/source_insert551_motion_envelope_v2.json');scope=validation_scope(task)
  self.assertEqual(scope['name'],'motion_envelope_3s_v2');self.assertEqual(scope['steps'],30)
  self.assertEqual(PROFILES[task['driver']][0].name,'motion_envelope_v2')
 def test_long_profile_is_explicit_and_bounded(self):
  task=read_task(HERE/'examples/diagnostic551_motion_envelope_v2_10s.json');scope=validation_scope(task)
  self.assertEqual(scope['steps'],100);self.assertEqual(scope['max_wall_seconds'],14400)
 def test_cannot_expand_to_eighteen_seconds(self):
  task=_hugsim_expand(json.loads((HERE/'examples/source_insert551_motion_envelope_v2.json').read_text()));task['validation']='avoidance_18s_v1'
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'task.json';p.write_text(json.dumps(task))
   with self.assertRaises(ValueError):read_task(p)
 def test_old_memory_profile_keeps_old_bundle(self):
  self.assertEqual(PROFILES['framework_memory551_geometry_v1'][0].name,'memory_lifecycle_v1')
if __name__=='__main__':unittest.main()
