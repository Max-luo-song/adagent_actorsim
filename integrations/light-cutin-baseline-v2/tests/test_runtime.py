
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

import importlib.util,json,pickle,sys,tempfile,unittest
pass  # Path is provided by the portable boundary above
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

LOCAL=Path(__file__).resolve().parents[1];OLD=LOCAL.parent/'light-cutin-diagnostic'
sys.path[:0]=[str(LOCAL),str(OLD)]
from common import configure,read,write,DEFAULT,CR
configure()

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

runtime=load('runtime',LOCAL/'runtime.py')
scoring=load('scoring',LOCAL/'scoring.py')

class OffsetRuntimeTests(unittest.TestCase):
    def test_actual_motion_feedback_keeps_render_collision_and_model_clocks_consistent(self):
        from runtime_configuration import load_vehicle_config
        from uniad_bridge import ORDER
        vehicle=load_vehicle_config();renders=[];checks=[];inputs=[]
        class Session:
            def __init__(self,**kwargs):
                self.vehicles=SimpleNamespace(activation_frame=0)
                self.backend=SimpleNamespace(trainer=SimpleNamespace(models={'Background':SimpleNamespace(get_opacity=torch.ones(1))}))
            def load_scene(self,path):pass
            def render(self,pose,timestamp):
                renders.append((pose.copy(),timestamp))
                return dict(ego_to_world=pose.copy(),timestamp_s=timestamp,source_frame=int(round(timestamp*10)),
                    cameras={k:dict(rgb=np.zeros((2,2,3),dtype=np.uint8)) for k in ORDER})
            def close(self):pass
        class Evaluation:
            def __init__(self,*args):pass
            def check(self,pose,frame):
                checks.append(frame)
                return dict(collision=False,native_route_deviation=False,native_route_completed=False)
        def convert(packet,mount,**kwargs):
            inputs.append((packet['timestamp_s'],packet['source_frame'],kwargs['speed_mps']))
            return {},dict(timestamp=packet['timestamp_s'],model_lidar_frame=dict(virtual_to_ego=np.eye(4)))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pose=np.eye(4);pose[0,3]=20.
            write(root/'plan.json',dict(seed=123,common_scene_token='test',initial_context=dict(ego_to_world=pose.tolist(),source_frame=10)))
            write(root/'scenario/scenario.json',dict(config=dict(intervention_start_s=1.2,initial_ego_speed_mps=4.)))
            write(root/'baseline_freeze.json',read(DEFAULT/'baseline_freeze.json'))
            def respond(path,process,seconds):
                i=int(Path(path).stem.split('_')[-1]);folder=root/'A'/f'uniad_worker/step_{i:03d}'
                np.savez(folder/'perception.npz',trajectory=np.c_[np.zeros(6),np.arange(1,7)*2.])
                np.savez(folder/'motion_head_raw.npz',all_traj_preds=np.zeros((1,1,1,1,1,2)))
                write(folder/'temporal_state.json',dict(history_bev_present=i>0,metadata_cache_is_tracking_cache=True))
                return dict(timestamp=i*.1,perception=str(folder/'perception.npz'))
            ready=dict(planning_bev='current',branch_restored=None,model_profile='calibration_geometry_v1',original_collision_postprocessor=False,cameras=3,thresholds=[.4,.35])
            def boot(folder,out,seed):
                with (folder/'step_000/observation.pkl').open('rb') as f:obs,info=pickle.load(f)
                self.assertEqual(info['timestamp'],0.)
                return SimpleNamespace(),None,ready
            with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.empty_cache'),\
                patch('native_car_edit.make_types',return_value=(object,Session)),\
                patch('scene_api.SceneSpec.read',return_value=SimpleNamespace(vehicle=vehicle,fps=10.)),\
                patch.object(runtime,'actor_arrays',return_value={'RigidNodes__instances_size':np.ones((15,3))}),\
                patch.object(runtime,'Evaluation',Evaluation),patch('uniad_bridge.to_uniad',side_effect=convert),\
                patch.object(runtime,'start_worker',side_effect=boot),patch.object(runtime,'stop_worker'),\
                patch.object(runtime,'wait_json',side_effect=respond):
                report=runtime.run_arm(root,'A',2)
            self.assertEqual(report['status'],'completed',report.get('error'))
            np.testing.assert_allclose([t for p,t in renders],[1.,1.1,1.2])
            np.testing.assert_allclose([v[0] for v in inputs],[0.,.1,.2])
            self.assertEqual([v[1] for v in inputs],[10,11,12])
            self.assertEqual(checks[0],10);self.assertAlmostEqual(checks[-1],12.)
            self.assertAlmostEqual(renders[0][0][0,3],20.)
            self.assertGreater(renders[-1][0][0,3],20.)
            for obs in report['observations']:
                self.assertAlmostEqual(obs['pose'][0][3],20.+obs['motion_state'][1])
            self.assertEqual(report['controls_completed'],2)

    def test_native_scoring_offsets_actor_time_but_never_uses_neural_plan_as_execution(self):
        previous=load('previous_contract_tests',OLD/'test_contracts.py')
        original_read=scoring.read;original_boxes=scoring.Evaluation.boxes;frames=[]
        def extended_read(path):
            value=original_read(path)
            if Path(path).name=='plan.json':value['initial_context']=dict(source_frame=10)
            return value
        def boxes(owner,frame):
            frames.append(frame);return original_boxes(owner,frame)
        with patch.object(scoring,'read',side_effect=extended_read),patch.object(scoring.Evaluation,'boxes',boxes):
            previous.DiagnosticEvidenceTests().test_native_execution_stream_uses_recorded_future_not_neural_plan()
        self.assertEqual(min(frames),10);self.assertEqual(max(frames),40)

    def test_B_and_patch_are_refused_before_creating_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(ValueError):runtime.run_arm(root,'B',2)
            with self.assertRaises(ValueError):runtime.run_arm(root,'A',2,candidate={'fixture':True})
            self.assertFalse((root/'B').exists());self.assertFalse((root/'A').exists())

if __name__=='__main__':unittest.main()
