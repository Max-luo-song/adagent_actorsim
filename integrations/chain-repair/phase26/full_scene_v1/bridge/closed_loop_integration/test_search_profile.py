
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
from run_edited_closed_loop import HERE,PROFILES,read_task,validation_scope
class SearchProfileTests(unittest.TestCase):
 def test_explicit_new_profile_selects_migrated_search_bundle(self):
  task=read_task(HERE/'examples/source_insert551_motion_search_v3.json');scope=validation_scope(task)
  self.assertEqual(PROFILES[task['driver']][0].name,'motion_search_v3');self.assertEqual(scope['name'],'motion_search_3s_v3');self.assertEqual(scope['steps'],30)
 def test_hundred_steps_require_explicit_diagnostic_profile(self):
  task=read_task(HERE/'examples/diagnostic551_motion_search_v3_10s.json');scope=validation_scope(task)
  self.assertEqual(scope['steps'],100);self.assertEqual(scope['max_wall_seconds'],14400)
 def test_old_white_car_repair_bundle_remains_available(self):
  self.assertEqual(PROFILES['framework_memory_envelope551_geometry_v2'][0].name,'motion_envelope_v2')
 def test_migration_cannot_expand_to_eighteen_seconds(self):
  task=_hugsim_expand(json.loads((HERE/'examples/source_insert551_motion_search_v3.json').read_text()));task['validation']='avoidance_18s_v1'
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'task.json';f.write_text(json.dumps(task))
   with self.assertRaises(ValueError):read_task(f)
if __name__=='__main__':unittest.main()
