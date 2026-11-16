"""Exercise the production render writer and final validator on CPU fixtures."""

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

import tempfile,types,unittest
pass  # Path is provided by the portable boundary above
from unittest.mock import patch
import numpy as np
from PIL import Image
import workflow
from contracts import read,write,pose
from guards import validate_rows

class RenderWriter(unittest.TestCase):
    def test_production_writer_persists_all_camera_ids_and_finalizes(self):
        import torch
        import pipeline
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);prior=out/'prior.png';Image.new('RGB',(8,6)).save(prior)
            records=[dict(source_frame=f,camera_id=c,variant=v,image=workflow.asset(prior)) for f in [0,4] for c in [0,1,2] for v in ['original','swapped']]
            write(out/'preview.json',dict(records=records))
            write(out/'compiled_trajectory.json',dict(samples=[dict(source_frame=f,timestamp_s=f/10,ego_to_world=np.eye(4).tolist()) for f in [0,4]]))
            q=np.zeros((5,4,4));q[:,:,0]=1
            np.savez(out/'tracks.npz',RigidNodes__instances_size=np.ones((4,3)),RigidNodes__instances_trans=np.full((5,4,3),100.),RigidNodes__instances_quats=q)
            legacy=types.SimpleNamespace(pose=pose,verify_asset=workflow.verify_asset,image=pipeline.image,save_image=pipeline.save_image)
            legacy.validate_rows=lambda rows:validate_rows(rows,'program',legacy)
            def enrich(path,job):
                manifest=read(path/'render/manifest.json')
                for row in manifest['rows']:self.assertEqual(row['camera_id'],row['appearance_conditioning']['camera_id'])
                manifest['appearance_conditioning_sources']=workflow.asset(path/'render/conditioning_sources.json')
                write(path/'render/manifest.json',manifest)
            module=types.SimpleNamespace(legacy=legacy,SOURCE_MAPS=[dict(camera_image_ids={str(c):list(range(5)) for c in [0,1,2]})],enrich_render_manifest=enrich)
            class Renderer:
                def __init__(self,*args):self.backend=types.SimpleNamespace(cameras_data={c:types.SimpleNamespace(cam_to_ego=np.eye(4)) for c in [0,1,2]})
                def original(self,f,c):return np.zeros((6,8,3),np.uint8),prior,np.eye(4)
                def select_row(self,row):pass
                def frame(self,ego,f,c):return dict(cam_infos=dict(camera_to_world=torch.eye(4),intrinsics=torch.eye(3)),_appearance_conditioning=dict(source_frame=f,camera_id=c))
                def rgb(self,data,layers=False):return dict(rgb=torch.zeros((6,8,3)))
                def valid_mask(self,result,data):return torch.ones((6,8),dtype=torch.bool)
                def close(self):pass
            job=dict(frame_stride=3,task=dict(scene_manifest='fixture',renderer_bundle='fixture'),inputs=dict(checkpoint=dict(sha256='weights')))
            write(out/'job.json',job)
            contract=dict(program=dict(sha256='program'),effective_actor_tracks=dict(path=str(out/'tracks.npz')),source_preview=dict(path=str(out/'preview.json')))
            prepared=(out,job,{},module,contract,dict(car_trans=np.zeros((5,3))),Renderer)
            with patch.object(workflow,'configure',return_value=prepared):result=workflow.render(out)
            self.assertEqual(result['rendered_rows'],12)
            manifest=read(out/'render/manifest.json');self.assertTrue(manifest['completed'])
            self.assertEqual({r['camera_id'] for r in manifest['rows']},{0,1,2})
            for path in (out/'render/rows').glob('*.json'):
                row=read(path);self.assertEqual(row['camera_id'],int(row['id'].split('_')[1]))

if __name__=='__main__':unittest.main()
