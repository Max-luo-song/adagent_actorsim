
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

import json,sys,tempfile,unittest
pass  # Path is provided by the portable boundary above
LOCAL=Path(__file__).resolve().parents[1];sys.path.insert(0,str(LOCAL))
from bindings import ROOT,OLD,load

class SceneClockTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  common=load('common',OLD/'common.py');common.configure()
  from scene_api import SceneSpec
  cls.SceneSpec=SceneSpec
  cls.original=ROOT/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26/cross_scene_validation_v1/cases'
 def manifest(self,scene):return _hugsim_expand(json.loads((self.original/str(scene)/'off/scene_manifest.json').read_text()))
 def parse(self,value):
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'scene.json';path.write_text(json.dumps(value));return self.SceneSpec.read(path)
 def test_actual_v1_clock_string_fails_and_registered_value_passes_both_scenes(self):
  for scene in [552,172]:
   with self.subTest(scene=scene):
    old=self.manifest(scene)
    with self.assertRaisesRegex(ValueError,'Clock disagrees'):self.parse(old)
    fixed=dict(old,fps_authority='source_metadata_assumption');spec=self.parse(fixed)
    self.assertEqual(spec.source['scene'],scene);self.assertEqual(spec.frames,198);self.assertEqual(spec.fps,10.)
 def test_fix_does_not_bypass_wrong_frequency(self):
  value=self.manifest(552);value.update(fps_authority='source_metadata_assumption',fps=9.)
  with self.assertRaisesRegex(ValueError,'Clock disagrees'):self.parse(value)
 def test_runtime_preflight_rejects_mismatched_source_and_controls(self):
  from pipeline import validate_scene_manifest
  for field,value in [('scene',172),('controls',121)]:
   with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
    manifest=self.manifest(552);manifest['fps_authority']='source_metadata_assumption';Path(d,'scene_manifest.json').write_text(json.dumps(manifest))
    plan=_hugsim_expand(json.loads((self.original/'552/off/plan.json').read_text()));plan[field]=value
    with self.assertRaisesRegex(ValueError,'mismatch'):validate_scene_manifest(Path(d),plan)
 def test_current_generator_uses_interface_enum_not_prose(self):
  import ast
  tree=ast.parse((LOCAL/'pipeline.py').read_text());found=[]
  for node in ast.walk(tree):
   if isinstance(node,ast.Dict):
    for key,value in zip(node.keys,node.values):
     if isinstance(key,ast.Constant) and key.value=='fps_authority':found.append(ast.literal_eval(value))
  self.assertEqual(found,['source_metadata_assumption'])
if __name__=='__main__':unittest.main()
