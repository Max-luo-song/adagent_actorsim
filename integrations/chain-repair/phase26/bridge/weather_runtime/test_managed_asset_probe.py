
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

pass  # Path is provided by the portable boundary above
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from managed_asset_probe import probe_managed_asset
from runtime_contract import CAMERAS


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);(self.root/'ego_pose').mkdir()
        for frame in (0,80):np.savetxt(self.root/f'ego_pose/{frame:03d}.txt',np.eye(4))
        backend=SimpleNamespace(spec=SimpleNamespace(source=dict(inputs=str(self.root),start_timestep=0)),
            managed_assets=SimpleNamespace(assets={}),_last=None)
        backend.clear_observation_history=lambda:setattr(backend,'_last',None)
        self.session=SimpleNamespace(backend=backend,objects={},revision=0)
        def edit(command):
            self.assertIsNone(backend._last)
            self.assertEqual(command.get('expected_revision',self.session.revision),self.session.revision)
            op=command['operation']
            if op=='insert':
                backend.managed_assets.assets['car']=SimpleNamespace(pose=np.asarray(command['asset_to_world']),local_means=torch.zeros(3,3))
                self.session.objects['car']={}
            elif op=='set_pose':backend.managed_assets.assets['car'].pose=np.asarray(command['asset_to_world'])
            elif op=='remove':backend.managed_assets.assets.clear();self.session.objects.clear()
            self.session.revision+=1
            return dict(edit_revision=self.session.revision)
        def reset():
            backend.managed_assets.assets.clear();self.session.objects.clear();self.session.revision=0
            backend.clear_observation_history()
        self.session.edit=edit;self.session.reset=reset
        self.command=dict(schema='scene_edit_v1',mode='diagnostic_preview',operation='insert',
            object_id='managed:car',asset_to_world=np.eye(4).tolist(),expected_revision=0)

    def render(self,backend,pose,time):
        rgb=np.zeros((4,6,3),dtype=np.uint8);mask=np.zeros((4,6),dtype=np.uint8)
        if backend.managed_assets.assets:
            asset=backend.managed_assets.assets['car'];x=2+int(asset.pose[0,3]>0)
            rgb[1,x]=100;mask[1,x]=255
        backend._last=(time,pose)
        return dict(cameras={c:dict(rgb=rgb.copy(),weather_dynamic_mask=mask.copy()) for c in CAMERAS})

    def test_insert_move_remove_and_clean_revision_for_real_episode(self):
        r=probe_managed_asset(self.session,self.command,self.root/'probe',self.render)
        self.assertTrue(r['passed'])
        self.assertEqual(len(r['receipts']),3)
        self.assertFalse(self.session.objects)
        self.assertFalse(self.session.backend.managed_assets.assets)
        self.assertEqual(self.session.revision,0)

    def test_failed_render_does_not_leave_asset_or_edit_revision(self):
        calls=[]
        def render(*args):
            calls.append(True)
            if len(calls)==2:raise RuntimeError('raster failed')
            return self.render(*args)
        with self.assertRaisesRegex(RuntimeError,'raster failed'):
            probe_managed_asset(self.session,self.command,self.root/'probe',render)
        self.assertFalse(self.session.backend.managed_assets.assets)
        self.assertEqual(self.session.revision,0)
        self.assertTrue((self.root/'probe/report.json').exists())


if __name__=='__main__':unittest.main()
