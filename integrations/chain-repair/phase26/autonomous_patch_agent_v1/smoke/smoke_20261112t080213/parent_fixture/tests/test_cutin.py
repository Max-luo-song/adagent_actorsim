"""CPU tests with explicit mocked neural inference; no GPU outcome is claimed."""

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

import contextlib,importlib.util,pickle,shutil,sys,tempfile,unittest
pass  # Path is provided by the portable boundary above
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
TEST_DIR=Path(__file__).resolve().parents[1];V4=TEST_DIR.parent/'light-map551-v4';OLD=TEST_DIR.parent/'light-cutin-diagnostic'
sys.path[:0]=[str(TEST_DIR),str(V4),str(OLD)]
from common import *
configure()
from runtime_configuration import load_vehicle_config
from uniad_bridge import ORDER
from cutin_scenario import CONFIG,polygon
from pair_diagnostics import verify_actor_sync
spec=importlib.util.spec_from_file_location('runtime',TEST_DIR/'runtime.py');runtime=importlib.util.module_from_spec(spec);sys.modules['runtime']=runtime;spec.loader.exec_module(runtime)
A=P26/'light_map551_v4';B=P26/'light_map551_cutin_v5'

class CutinTests(unittest.TestCase):
    def test_entrypoint_binds_fixed_source_offset_scoring_release(self):
        spec=importlib.util.spec_from_file_location('test_v5_entrypoint',TEST_DIR/'pipeline.py')
        entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
        selected=entry.bindings()
        self.assertEqual(selected['scoring'],str(V4/'scoring.py'))
        self.assertEqual(selected['runtime'],str(TEST_DIR/'runtime.py'))

    def test_common_boundary_bypasses_edit_render_context(self):
        import intervention
        calls=[]
        @contextlib.contextmanager
        def edited(frame):calls.append(frame);yield
        owner=SimpleNamespace(activation_frame=15,render_context=edited)
        session=SimpleNamespace(vehicles=owner)
        with patch.object(intervention,'original_install',return_value=dict(activation_frame=15)):
            result=intervention.install_trajectory(session,'mocked',1.5)
        for frame in [10,15,16]:
            with owner.render_context(frame):pass
        self.assertEqual(calls,[16]);self.assertTrue(result['common_boundary_uses_exact_original_renderer'])

    def test_script_is_same_native_car_safe_continuous_and_relevant(self):
        r=read(B/'scenario/scenario.json');self.assertTrue(r['passed']);self.assertEqual(r['config'],CONFIG)
        self.assertGreater(r['minimum_other_actor_gap_m'],.2)
        with np.load(B/'scenario/target_track.npz') as z:
            self.assertTrue(np.array_equal(z['edited_trans'][:16],z['original_trans'][:16]))
            self.assertTrue(np.array_equal(z['edited_quats'][:16],z['original_quats'][:16]))
            p=z['rendered_trans'];q=z['rendered_quats']
            self.assertLess(np.max(np.linalg.norm(np.diff(p[10:],axis=0)/.1,axis=1)),35)
            self.assertLess(np.max(np.linalg.norm(np.diff(p[10:],n=2,axis=0)/.01,axis=1)),8)
            self.assertLess(np.linalg.norm(p[64:]-p[-1],axis=1).max(),1e-7)
        # Counterfactual only: replay the KNOWN A path offline to ensure B's intervention
        # actually reaches that corridor within the fixed window. This is never online input.
        report=read(A/'A/run_report.json');v=load_vehicle_config();overlap=[]
        with np.load(A/'A/actor_tracks.npz') as z:size=z['RigidNodes__instances_size'][3]
        for o in report['observations']:
            i=o['index'];pose=np.asarray(o['pose']);rot=__import__('scipy.spatial.transform',fromlist=['Rotation']).Rotation.from_matrix(pose[:3,:3]).as_quat()[[3,0,1,2]]
            if polygon(pose[:3,3],rot,[2*v.half_length_m,2*v.half_width_m,v.body_height_m]).intersects(polygon(p[i+10],q[i+10],size)):overlap.append(i)
        self.assertTrue(overlap,'Cut-in never affects baseline corridor in fixed episode')
        print('OFFLINE COUNTERFACTUAL ONLY: scripted target overlaps recorded A ego at indices',overlap[0], '..',overlap[-1])

    def test_real_tracker_motion_common_history_and_absolute_actor_clock(self):
        self.assertEqual(Path(runtime.__file__).resolve(),TEST_DIR/'runtime.py')
        v=load_vehicle_config();seen=[];starts=[]
        with np.load(A/'A/actor_tracks.npz') as z:tracks={k:z[k].copy() for k in z.files}
        with np.load(B/'scenario/target_track.npz') as z:
            tracks['RigidNodes__instances_trans'][:,3]=z['rendered_trans'];tracks['RigidNodes__instances_quats'][:,3]=z['rendered_quats']
        class Session:
            def __init__(self,**kwargs):
                state={k:torch.as_tensor(tracks['RigidNodes__'+k]) for k in ['instances_trans','instances_quats','instances_size','instances_fv']}
                self.vehicles=SimpleNamespace(activation_frame=0,programs={})
                self.backend=SimpleNamespace(env=SimpleNamespace(models={'RigidNodes':state}),trainer=SimpleNamespace(models={
                    'Background':SimpleNamespace(get_opacity=torch.ones(1)),'RigidNodes':SimpleNamespace(**state)}))
            def load_scene(self,path):pass
            def render(self,pose,timestamp):
                index=round(timestamp*10)-10;cameras={}
                for k in ORDER:
                    with Image.open(A/f'A/steps/{index:03d}/{k}.png') as im:rgb=np.asarray(im).copy()
                    if index>5:rgb[0,0,0]^=1
                    cameras[k]=dict(rgb=rgb,depth=np.zeros(rgb.shape[:2]),K=np.eye(3),camera_to_ego=np.eye(4),camera_to_world=pose)
                return dict(schema='unified_observation_v1',camera_order=ORDER,training=False,formal_evaluation_admitted=False,
                    timestamp_s=timestamp,source_frame=round(timestamp*10),ego_to_world=pose,edit_trust=dict(edited=True),cameras=cameras)
            def close(self):pass
        class Evaluation:
            def __init__(self,*args):pass
            def check(self,pose,frame):return dict(collision=False,native_route_deviation=False,native_route_completed=False)
        def install(session,path,start):
            starts.append(start);session.vehicles.activation_frame=15;session.vehicles.programs={3:dict(mock=True)}
            return dict(activation_frame=15)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for n in ['scene_manifest.json','baseline_freeze.json','map/map.json','scenario/scenario.json','scenario/target_track.npz']:
                dst=root/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(B/n,dst)
            plan=read(B/'plan.json');plan['native_collision_policy']=dict(enabled_after_common_index=-1,implementation='unchanged UniAD collision_optimization',external_obstacle_mpc=False);write(root/'plan.json',plan)
            def response(path,process,seconds):
                index=int(Path(path).stem.split('_')[-1]);dst=Path(path).parent/f'step_{index:03d}'
                with (dst/'observation.pkl').open('rb') as f:obs,info=pickle.load(f)
                seen.append((info['timestamp'],info['source_frame']))
                for n in ['perception.npz','motion_head_raw.npz','temporal_state.json']:
                    shutil.copyfile(A/f'A/uniad_worker/step_{index:03d}'/n,dst/n)
                write(dst/'native_collision_audit.json',dict(enabled=True,native_code_modified=False))
                return dict(timestamp=index*.1,perception=str(dst/'perception.npz'))
            ready=dict(planning_bev='current',branch_restored=None,model_profile='calibration_geometry_v1',original_collision_postprocessor=False,cameras=3,thresholds=[.4,.35],native_collision_postprocessor_policy=plan['native_collision_policy'])
            with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.empty_cache'),\
                patch('native_car_edit.make_types',return_value=(object,Session)),\
                patch('scene_api.SceneSpec.read',return_value=SimpleNamespace(vehicle=v,fps=10.)),\
                patch.object(runtime,'actor_arrays',return_value=tracks),patch.object(runtime,'Evaluation',Evaluation),\
                patch.object(runtime,'install_trajectory',side_effect=install),\
                patch.object(runtime,'start_worker',return_value=(SimpleNamespace(),None,ready)),patch.object(runtime,'stop_worker'),\
                patch.object(runtime,'wait_json',side_effect=response):
                original_tracks={k:v.copy() for k,v in tracks.items()}
                with np.load(A/'A/actor_tracks.npz') as original:
                    tracks.clear();tracks.update({k:original[k].copy() for k in original.files})
                baseline=runtime.run_arm(root,'A',7)
                self.assertEqual(baseline['status'],'completed',baseline.get('error'))
                tracks.clear();tracks.update(original_tracks);seen.clear()
                report=runtime.run_arm(root,'B',7,root/'A')
                (root/'B').rename(root/'B_verified_prefix_cpu')
                standalone=runtime.run_arm(root,'B',7)
                self.assertEqual(standalone['status'],'completed',standalone.get('error'))
                self.assertEqual(standalone['controls_completed'],7)
                self.assertFalse(list((root/'B').glob('steps/*/prefix_gate.json')))
                (root/'B').rename(root/'B_standalone_cpu')
                (root/'B_verified_prefix_cpu').rename(root/'B')
                seen[:]=seen[:8]
            self.assertEqual(report['status'],'completed',report.get('error'));self.assertEqual(report['controls_completed'],7)
            self.assertEqual(starts,[1.5,1.5]);self.assertEqual(len(seen),8);self.assertAlmostEqual(seen[-1][0],.7);self.assertEqual(seen[-1][1],17)
            self.assertTrue(report['actor_sync']['passed'])
            for i in range(6):
                gate=read(root/f'B/steps/{i:03d}/prefix_gate.json');self.assertTrue(all(gate['pixel_identical'].values()))
                self.assertTrue(gate['full_model_history_rebuilt_by_fresh_prefix_inference']);self.assertTrue(gate['motion_max_absolute_delta'])
            self.assertFalse((root/'B/steps/006/prefix_gate.json').exists())
            # An accidental extra-actor edit must fail the real synchronization checker.
            tracks['RigidNodes__instances_trans'][100,2,0]+=.1
            with self.assertRaisesRegex(ValueError,'synchronization'):
                fake=Session();fake.vehicles.programs={3:{}};verify_actor_sync(root,tracks,fake,dict(activation_frame=15))
        print('CPU TEST ONLY: actual fixed tracker/native Motion, mocked neural inference/rendering, 6 verified prefix observations; no B GPU result')

    def test_prefix_rejects_changed_motion_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'steps/000';folder.mkdir(parents=True);out=root/'uniad_worker/step_000';out.mkdir(parents=True)
            p=A/'A/uniad_worker/step_000'
            with np.load(p/'motion_head_raw.npz') as z:motion={k:z[k].copy() for k in z.files}
            motion['all_traj_preds'].flat[0]+=.01;np.savez(out/'motion_head_raw.npz',**motion)
            with np.load(p/'perception.npz') as z:perception={k:z[k].copy() for k in z.files}
            a=read(A/'A/steps/000/receipt.json')
            with self.assertRaisesRegex(ValueError,'motion differs'):
                runtime.prefix_gate(folder,A/'A',0,a['ego_to_world'],a['image_sha256'],perception)

if __name__=='__main__':unittest.main()
