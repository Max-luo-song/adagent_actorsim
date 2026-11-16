
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
from run_edited_closed_loop import read_task,validation_scope,HERE


class RuntimeRoadTests(unittest.TestCase):
 def test_short_new_profile_has_separate_scope(self):
  t=read_task(HERE/'examples/source_insert551_runtime_road.json');s=validation_scope(t)
  self.assertEqual(s['steps'],30);self.assertEqual(s['name'],'road_runtime_3s_v1');self.assertEqual(s['max_wall_seconds'],7200)
 def test_new_100_step_profile_is_explicit(self):
  t=read_task(HERE/'examples/diagnostic551_runtime_road_10s.json');self.assertEqual(validation_scope(t)['steps'],100)
 def test_old_diagnostic_profile_remains_available(self):
  t=read_task(HERE/'examples/diagnostic551_10s.json');self.assertEqual(t['driver'],'phase35_event_mpc551_geometry_v1')
 def test_new_profile_cannot_silently_expand_to_18_seconds(self):
  t=_hugsim_expand(json.loads((HERE/'examples/source_insert551_runtime_road.json').read_text()));t['validation']='avoidance_18s_v1'
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'task.json';p.write_text(json.dumps(t))
   with self.assertRaises(ValueError):read_task(p)
 def test_memory_lifecycle_short_and_long_profiles_are_explicit(self):
  short=read_task(HERE/'examples/source_insert551_memory_lifecycle.json');long=read_task(HERE/'examples/diagnostic551_memory_lifecycle_10s.json')
  self.assertEqual(validation_scope(short)['name'],'memory_lifecycle_3s_v1');self.assertEqual(validation_scope(short)['steps'],30);self.assertEqual(validation_scope(long)['steps'],100)


if __name__=='__main__':unittest.main()
