
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

import contextlib,copy,sys,unittest,types
pass  # Path is provided by the portable boundary above
import numpy as np
from guards import outside_box_mask,validate_rows,DeferredSceneState

class Guards(unittest.TestCase):
    def test_projection_clips_straddling_box_and_leaves_background(self):
        k=np.array([[50,0,50],[0,50,50],[0,0,1]])
        mask=outside_box_mask((100,100),k,np.eye(4),np.array([0,0,5]),np.eye(3),[2,2,2],padding=0)
        self.assertFalse(mask[50,50]);self.assertTrue(mask[0,0])
        self.assertTrue(outside_box_mask((100,100),k,np.eye(4),np.array([0,0,-5]),np.eye(3),[2,2,2]).all())
        mask=outside_box_mask((100,100),k,np.eye(4),np.array([0,0,.1]),np.eye(3),[2,2,2]);self.assertFalse(mask.all())
    def test_edit_lives_through_backward_and_restores_on_close(self):
        import torch
        state=torch.tensor([2.]);feature=torch.nn.Parameter(torch.tensor([3.]))
        class Owner:
            @contextlib.contextmanager
            def at(self,frame,edited):
                state.fill_(10 if edited else 2)
                try:yield
                finally:state.fill_(2)
        context=DeferredSceneState(Owner());context.select(0,True)
        loss=(state*feature).sum();loss.backward();self.assertEqual(feature.grad.item(),10)
        context.select(1,False);self.assertEqual(state.item(),2);context.close();self.assertEqual(state.item(),2)
    def rows(self):
        return [dict(id=f'{frame}_{kind}',kind=kind,split=split,source_frame=frame,camera_id=0,
            edit_revision=int(kind=='novel'),scene_variant='position_swapped' if kind=='novel' else 'original',
            edit_program_sha256='program',reference_sanitized=kind=='novel',dynamic_loss_masked=True,
            ego_to_world=np.eye(4).tolist(),camera_to_world=np.eye(4).tolist(),K=np.eye(3).tolist(),
            scene=551,checkpoint_sha256='weights',**{k:dict(path='fixture') for k in ['before','reference','mask','reference_mask']})
            for frame,split in [(0,'train'),(4,'eval')] for kind in ['novel','original']]
    def helper(self):return types.SimpleNamespace(pose=lambda p:None,verify_asset=lambda a:None,image=lambda p:np.zeros((4,4,3)))
    def test_explicit_two_revisions_accepted_and_wrong_program_rejected(self):
        rows=self.rows();validate_rows(rows,'program',self.helper());rows[-1]['edit_program_sha256']='stale'
        with self.assertRaises(ValueError):validate_rows(rows,'program',self.helper())
    def test_cannot_silently_label_edited_rows_as_unedited(self):
        for field,value in [('edit_revision',0),('reference_sanitized',False),('dynamic_loss_masked',False),('scene_variant','original')]:
            rows=self.rows();rows[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_rows(rows,'program',self.helper())

if __name__=='__main__':unittest.main()
