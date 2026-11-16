
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

import copy,json,tempfile,unittest
pass  # Path is provided by the portable boundary above
import pipeline as p
class AdmissionTests(unittest.TestCase):
 def candidate(self,score,status='completed'):
  return {'metrics':{'original551':{'status':'completed'},'cutin551':{'status':status,'score':{'hdscore':score}}}}
 def test_highest_completed_pair_not_latest(self):
  best,_=p.select_best({'best':self.candidate(.3),'new':self.candidate(.29),'incomplete':self.candidate(.9,'terminated')})
  self.assertEqual(best,'best')
 def test_invalid_score_not_ranked(self):
  with self.assertRaises(ValueError):p.select_best({'bad':self.candidate(float('nan'))})
 def test_missing_pair_not_ranked(self):
  with self.assertRaises(ValueError):p.select_best({'new':{'metrics':{'cutin551':{'status':'completed','score':{'hdscore':.9}}}}})
 def test_wrong_scene_raw_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   b=Path(d);(b/'raw_metadata').mkdir();(b/'raw_metadata/raw_metadata.json').write_text(json.dumps({'scene_index':552,'frames':[]}))
   with self.assertRaises(ValueError):p.verify_raw(b,{'scene':172,'frames':198})
 def test_missing_raw_not_treated_as_verified(self):
  with tempfile.TemporaryDirectory() as d:self.assertFalse(p.verify_raw(Path(d),{'scene':172,'frames':198}))
 def test_prepared_asset_tamper_rejected(self):
  scene=621;source=p.read(p.REPORTS[scene])
  with tempfile.TemporaryDirectory() as d:
   b=Path(d);(b/'geometry').write_bytes(b'before');m={'scene':scene,'frames':source['frames'],'raw_data':str(p.RAW_BASE/str(scene)),
    'source_checkpoint':source['checkpoint'],'frames_data':[{'frame':i} for i in range(source['frames'])],'artifacts':{'geometry':p.sha(b/'geometry')}}
   (b/'source_manifest.json').write_text(json.dumps(m));p.verify_bundle(scene,b,checkpoint_hash=False)
   (b/'geometry').write_bytes(b'after')
   with self.assertRaises(ValueError):p.verify_bundle(scene,b,checkpoint_hash=False)
if __name__=='__main__':unittest.main()
