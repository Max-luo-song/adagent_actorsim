
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

import ast,sys,tempfile,types,unittest
pass  # Path is provided by the portable boundary above
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
LOCAL=Path(__file__).resolve().parents[1];ROOT=LOCAL.parents[1]
sys.path[:0]=[str(LOCAL),str(ROOT/'waymo23-native-import')]
from counterfactual import wrap_forward,write

class ProbeTests(unittest.TestCase):
    def test_each_case_clears_full_cache_and_selects_only_declared_gate(self):
        # Execute the existing native reset implementation, with its synchronization
        # function extracted verbatim to avoid loading the GPU inference package.
        tree=ast.parse((ROOT/'waymo23-native-import/run_reliable_sequence.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='synchronize_temporal_state')
        scope={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'existing_synchronize','exec'),scope)
        module=types.ModuleType('run_reliable_sequence');module.synchronize_temporal_state=scope[fn.name]
        class Reliability:
            def __init__(self):self.reset()
            def reset(self):self.tracks={};self.next_id=0;self.scene_token=None;self.previous_time=None
        class TrackBase:
            def clear(self):self.max_obj_id=0
        class Core:
            def __init__(self):
                self.scene_reliability=Reliability();self.track_base=TrackBase();self.track_base.clear()
                self.prev_frame_info={};self.switch=SimpleNamespace(can_bus_mlp=object(),use_can_bus=False);self.calls=[]
                self.dirty()
            def dirty(self):
                self.prev_bev=object();self.test_track_instances=[1];self.l2g_r_mat=object();self.l2g_t=object();self.timestamp=0.
                self._adapter_scene_token='previous';self._adapter_timestamp=0.;self.prev_frame_info['prev_bev']=object()
                self.scene_reliability.tracks={1:1};self.scene_reliability.next_id=1;self.track_base.max_obj_id=1
            def named_modules(self):return [('encoder',self.switch)]
            def forward_test(self,img_metas):
                from resident_session_state import snapshot
                self.calls.append((self.switch.use_can_bus,float(img_metas[0][0]['can_bus'][13]),snapshot(self)))
                self.dirty();return 'unchanged-output'
        with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,run_reliable_sequence=module):
            root=Path(tmp);core=Core();wrap_forward(core,lambda:root)
            for enabled,speed in [(False,4.),(False,16.),(True,4.),(True,16.)]:
                write(root/'case.json',dict(use_can_bus=enabled,declared_speed_mps=speed));bus=np.zeros(18);bus[13]=speed
                result=core.forward_test(img_metas=[[dict(can_bus=bus)]])
                self.assertEqual(result,'unchanged-output');flag,seen,state=core.calls[-1]
                self.assertEqual((flag,seen),(enabled,speed));self.assertFalse(state['bev_present']);self.assertFalse(state['tracking_present'])
                self.assertFalse(state['tracking_pose_present']);self.assertEqual(state['reliability_tracks'],0)
            write(root/'case.json',dict(use_can_bus=True,declared_speed_mps=4.))
            with self.assertRaisesRegex(ValueError,'speed mismatch'):core.forward_test(img_metas=[[dict(can_bus=bus)]])

if __name__=='__main__':unittest.main()
