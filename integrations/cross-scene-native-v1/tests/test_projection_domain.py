
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
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scene_contracts import ASSETS,read
from boundaries import camera_projection
class ProjectionDomainTests(unittest.TestCase):
 def case(self,scene,camera,ray,wh):
  source=read(ASSETS/str(scene)/'source_manifest.json');cal={'opencv_to_waymo':np.eye(3),'camera_waymo_to_ego':np.eye(4),'intrinsics_9':source['cameras'][str(camera)]['intrinsics_9']};return camera_projection(np.array([[ray[0],ray[1],1.]]),np.eye(4),cal,wh)
 def test_actual172_ground_ray_on_folded_radial_branch_is_not_visible(self):
  _,_,valid=self.case(172,4,(1.3287672866809945,.18676001657453126),(1920,886));self.assertFalse(valid[0])
 def test_actual552_ground_ray_with_negative_radial_scale_is_not_visible(self):
  _,_,valid=self.case(552,2,(-.031841211307750174,1.417369513628504),(1920,1280));self.assertFalse(valid[0])
 def test_central_camera_ray_remains_visible(self):
  _,_,valid=self.case(172,4,(0.,.05),(1920,886));self.assertTrue(valid[0])
 def test_point_semantics_wrapper_filters_domain_without_changing_pixels_or_depth(self):
  from projection_domain import wrap_point_projector
  intrinsic=read(ASSETS/'172/source_manifest.json')['cameras']['4']['intrinsics_9'];pixels=np.array([[17,19],[23,25]]);depth=np.ones(2)
  def legacy(*args):return pixels.copy(),depth.copy(),np.ones(2,bool)
  wrapped=wrap_point_projector(legacy,np.eye(3));p,z,valid=wrapped(np.array([[1.3287672866809945,.18676001657453126,1],[0,.05,1]]),np.eye(4),intrinsic,(1920,886));np.testing.assert_array_equal(p,pixels);np.testing.assert_array_equal(z,depth);np.testing.assert_array_equal(valid,[False,True])
 def test_native_view_cannot_fall_back_to_legacy_point_labels(self):
  import tempfile
  from unittest.mock import patch
  import scoring_assets
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);legacy=root/'172/point_semantics';legacy.mkdir(parents=True);(legacy/'point_semantics_manifest.json').write_text('{}')
   with patch.object(scoring_assets,'ASSETS',root):
    with self.assertRaisesRegex(ValueError,'Corrected principal-branch'):scoring_assets.ensure_view(172)
 def test_nonpositive_tangential_jacobian_is_rejected(self):
  from projection_domain import valid_ray
  self.assertFalse(valid_ray(np.array([[0.,-.1,1.]]),[100,100,50,50,0,0,2,0,0])[0])
if __name__=='__main__':unittest.main()
