
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

import copy,importlib.util,pickle,sys,tempfile,unittest
pass  # Path is provided by the portable boundary above
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
TEST_DIR=Path(__file__).resolve().parents[1];OLD=TEST_DIR.parent/'light-cutin-diagnostic'
sys.path[:0]=[str(TEST_DIR),str(OLD)]
from common import *
configure()
from map_reference import MapRoadReference,StaticGeometry,previous
from runtime_configuration import load_vehicle_config
from live_env import Motion
from tracker import Tracker
from uniad_bridge import ORDER

def load(name,file):
    s=importlib.util.spec_from_file_location(name,file);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
runtime=load('runtime',TEST_DIR/'runtime.py');scoring=load('scoring',TEST_DIR/'scoring.py')
MAP=P26/'light_map551_v4/map/map.json'

class MapEpisodeTests(unittest.TestCase):
    def test_scored_planning_positions_match_fixed_tracker_mount_contract(self):
        import native_scores
        v=load_vehicle_config();mount=np.eye(4);mount[:3,3]=[.2,.985793,1.84019]
        raw=np.c_[np.zeros(6),np.arange(1,7)*2.]-mount[:2,3]
        info=dict(model_lidar_frame=dict(virtual_to_ego=mount),explicit_l2g_r_mat=np.eye(3),explicit_l2g_t=mount[:3,3]+[0,20.,0],timestamp=0.)
        frame=scoring.native_plan_from_lidar(native_scores,raw,info,v)
        self.assertAlmostEqual(frame['planned_traj']['traj'][0][0],22.)
        self.assertAlmostEqual(frame['planned_traj']['traj'][0][1],0.)
        self.assertAlmostEqual(frame['planned_traj']['traj'][-1][0],30.)
    def test_whole_body_rejects_outside_even_when_center_is_inside(self):
        v=load_vehicle_config();g=StaticGeometry(read(MAP),v);pose=np.eye(4);pose[0,3]=40;pose[1,3]=-1.4
        self.assertTrue(g.poly.contains(__import__('shapely.geometry',fromlist=['Point']).Point(pose[:2,3])))
        self.assertLess(g.pose_clearance(pose),0)
        pose[1,3]=0;self.assertGreater(g.pose_clearance(pose),0)
    def test_actual_lost_line_step58_can_use_map_without_any_detector(self):
        root=P26/'light_road551_v3/A';a=read(root/'run_report.json');v=load_vehicle_config();policy=MapRoadReference(v,read(MAP))
        with patch.object(previous,'detect',side_effect=AssertionError('Online image-line detector forbidden')):
            for i in (0,58):
                with (root/f'uniad_worker/step_{i:03d}/sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
                with np.load(root/f'uniad_worker/step_{i:03d}/perception.npz') as z:raw=z['trajectory'].copy()
                state=a['observations'][i]['motion_state'];motion=Motion(v.dt,v.wheelbase_m,state[3]);motion.x,motion.z,motion.yaw,motion.speed,motion.steer=state
                ref,audit=policy.process(raw,{}, {},info,motion)
                action,_=Tracker(v).control(ref,info,motion);chosen,gate=policy.admit(motion,action)
                self.assertEqual(audit['per_frame_line_detector_calls'],0);self.assertIsNotNone(chosen,gate)
                self.assertTrue(gate['certificate']['passed'])
    def test_map_exterior_has_cost_and_no_safe_action_is_fabricated(self):
        v=load_vehicle_config();g=StaticGeometry(read(MAP),v);mount=np.eye(4);mount[1,3]=.985793;mount[2,3]=1.84019
        g.set_pose(dict(model_lidar_frame=dict(virtual_to_ego=mount),explicit_l2g_r_mat=np.eye(3),explicit_l2g_t=np.array([0.,40.,1.84019])))
        inside=np.array([[0.,5.,0.,4.,0.]]);outside=np.array([[5.,5.,0.,4.,0.]])
        self.assertGreater(g.distance_cost(outside,1),g.distance_cost(inside,1));self.assertLess(g.clearance(outside)[0],0)
    def test_real_motion_full188_clock_history_and_native_score_actor_end(self):
        self.assertEqual(Path(runtime.__file__).resolve(),TEST_DIR/'runtime.py')
        v=load_vehicle_config();renders=[];checks=[];model_times=[]
        with np.load(P26/'light_baseline551_v2/A/actor_tracks.npz') as z:tracks={k:z[k].copy() for k in z.files}
        class Session:
            def __init__(self,**kwargs):
                self.vehicles=SimpleNamespace(activation_frame=0)
                self.backend=SimpleNamespace(trainer=SimpleNamespace(models={'Background':SimpleNamespace(get_opacity=torch.ones(1))}))
            def load_scene(self,path):pass
            def render(self,pose,timestamp):
                renders.append(timestamp)
                return dict(schema='unified_observation_v1',camera_order=ORDER,training=False,formal_evaluation_admitted=False,
                    timestamp_s=timestamp,source_frame=int(round(timestamp*10)),ego_to_world=pose,
                    edit_trust=dict(edited=False),cameras={k:dict(rgb=np.zeros((2,2,3),np.uint8),depth=np.zeros((2,2)),K=np.eye(3),camera_to_ego=np.eye(4),camera_to_world=pose) for k in ORDER})
            def close(self):pass
        class Evaluation:
            def __init__(self,*args):pass
            def check(self,pose,frame):
                checks.append(frame);return dict(collision=False,native_route_deviation=False,native_route_completed=False)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pose=np.eye(4);pose[0,3]=20.
            write(root/'plan.json',dict(seed=123,common_scene_token='mocked_full_episode',source_end_frame=198,
                initial_context=dict(ego_to_world=pose.tolist(),source_frame=10),assets=dict(uniad_weights=dict(path='mocked',sha256='mocked'))))
            write(root/'scenario/scenario.json',dict(config=dict(initial_ego_speed_mps=4.,intervention_start_s=1.2)))
            write(root/'map/map.json',read(MAP));write(root/'baseline_freeze.json',read(DEFAULT/'baseline_freeze.json'))
            write(root/'scene_manifest.json',read(P26/'light_baseline551_v2/scene_manifest.json'))
            def response(path,process,seconds):
                i=int(Path(path).stem.split('_')[-1]);step=root/'A'/f'uniad_worker/step_{i:03d}'
                with (step/'observation.pkl').open('rb') as f:obs,info=pickle.load(f)
                model_times.append((info['timestamp'],info['source_frame']))
                raw=np.c_[[.004,.054,.059,.097,.164,.454],np.arange(1,7)*2-.985793]
                np.savez(step/'perception.npz',trajectory=raw);np.savez(step/'motion_head_raw.npz',mocked=np.zeros(1))
                write(step/'temporal_state.json',dict(history_bev_present=i>0,metadata_cache_is_tracking_cache=True))
                return dict(timestamp=i*.1,perception=str(step/'perception.npz'))
            ready=dict(planning_bev='current',branch_restored=None,model_profile='calibration_geometry_v1',original_collision_postprocessor=False,cameras=3,thresholds=[.4,.35])
            with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.empty_cache'),\
                patch('native_car_edit.make_types',return_value=(object,Session)),\
                patch('scene_api.SceneSpec.read',return_value=SimpleNamespace(vehicle=v,fps=10.)),\
                patch.object(runtime,'actor_arrays',return_value=tracks),patch.object(runtime,'Evaluation',Evaluation),\
                patch.object(runtime,'start_worker',return_value=(SimpleNamespace(),None,ready)),\
                patch.object(runtime,'stop_worker'),patch.object(runtime,'wait_json',side_effect=response),\
                patch.object(previous,'detect',side_effect=AssertionError('No online detector')):
                report=runtime.run_arm(root,'A',188)
            self.assertEqual(report['status'],'completed',report.get('error'));self.assertEqual(report['controls_completed'],188)
            self.assertEqual(len(model_times),189);self.assertAlmostEqual(renders[-1],19.8);self.assertEqual(model_times[-1][1],198);self.assertAlmostEqual(model_times[-1][0],18.8)
            self.assertLessEqual(max(checks),198);self.assertEqual(report['termination_reason'],'source_scene_time_window_complete')
            self.assertTrue(all(t['road_gate']['certificate']['passed'] for t in report['trace']))
            self.assertLess(max(abs(o['motion_state'][0]) for o in report['observations']),.5)
            import native_scores
            captured={};actor_frames=[];old_boxes=scoring.Evaluation.boxes
            def record_boxes(owner,frame):actor_frames.append(frame);return old_boxes(owner,frame)
            def native_stub(out,vehicle):
                data,manifest=native_scores.export(out,vehicle);captured.update(data=data,manifest=manifest)
                return dict(status='unit_test_only_no_GPU',scores={})
            with patch.object(native_scores,'score',side_effect=native_stub),patch.object(scoring.Evaluation,'boxes',record_boxes):
                scores=scoring.score_arm(root,'A')
            self.assertEqual(max(actor_frames),198);self.assertEqual(min(actor_frames),10)
            self.assertEqual(scores['matched_keyframe_count'],164)
            executed=captured['data']['executed_path'][0]['frames'][0]['planned_traj']['traj']
            self.assertAlmostEqual(executed[-1][0],report['observations'][25]['pose'][0][3],places=4)
            print('UNIT-TEST ONLY: 188 native Motion controls, map gate, fresh mocked model per step, source end198 and164 scored keyframes; no real GPU inference')

if __name__=='__main__':unittest.main()
