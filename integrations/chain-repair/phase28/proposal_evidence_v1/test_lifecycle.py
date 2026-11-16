
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

import copy,sys,unittest
pass  # Path is provided by the portable boundary above
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[6]
sys.path[:0]=[str(HERE),str(ROOT/'waymo23-native-import'),str(ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad')]
from lifecycle import QueryLifecycle,retire_queries
CFG=dict(max_gap_seconds=.75,confirmation_seconds=.2,confirmation_observations=3,obstacle_margin_m=.5)
class Instances:
    def __init__(self,ids,values):self.obj_idxes=np.array(ids);self.values=np.array(values)
    def __getitem__(self,mask):return Instances(self.obj_idxes[mask],self.values[mask])

def fixture(visible=False,surface=False,occluded=False):
    # Camera Z is model-local forward Y, image is 100x100.
    l2c=np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1.]])
    intr=dict(fovx=float(2*np.arctan(100/60)),fovy=float(2*np.arctan(100/60)),cx=50.,cy=50.,W=100,H=100)
    # Reuse the actual intrinsic parser's schema.
    info=dict(timestamp=0.,scene_token='test',explicit_l2g_r_mat=np.eye(3),explicit_l2g_t=np.zeros(3),depth_encoding='expected_camera_z_v1',cam_params={'cam':dict(l2c=l2c,intrinsic=intr)})
    box=np.array([0.,10.,-1.,2.,4.,2.,0.,0.,0.]) if visible else np.array([30.,1.,-1.,2.,4.,2.,0.,0.,0.])
    depth=np.full((100,100),10.,dtype=np.float32);valid=np.ones((100,100),bool)
    obs=dict(depth={'cam':depth},surface_depth={'cam':depth},surface_valid={'cam':valid})
    return obs,info,box

class Tests(unittest.TestCase):
    def replay(self,visible=False,velocity=0.,missing=False):
        obs,info,box=fixture(visible);gate=QueryLifecycle(CFG);results=[]
        if missing:obs={}
        for i in range(4):
            b=box.copy();b[0]+=i*.4;b[7]=velocity;info=dict(info,timestamp=i*.1)
            results.append(gate.observe(b[None],[731],[0],obs,info))
        return results,gate,obs,info,box
    def test_inconsistent_unobserved_history_schedules_next_cache_retirement(self):
        result,*_=self.replay();self.assertEqual(result[0]['retire_next_forward'],[]);self.assertEqual(result[-1]['retire_next_forward'],[731])
        self.assertFalse(result[-1]['current_detections_filtered']);self.assertFalse(result[-1]['free_space_asserted'])
    def test_motion_explained_by_model_velocity_is_retained(self):
        result,*_=self.replay(velocity=4.);self.assertEqual(result[-1]['retire_next_forward'],[])
    def test_visible_object_and_missing_depth_are_retained(self):
        for visible,missing in [(True,False),(False,True)]:
            result,*_=self.replay(visible=visible,missing=missing);self.assertEqual(result[-1]['retire_next_forward'],[])
    def test_previously_observed_object_is_protected_after_leaving_view(self):
        result,gate,obs,info,box=self.replay(visible=True)
        for i in range(4,10):
            box[0]+=8;info=dict(info,timestamp=i*.1);r=gate.observe(box[None],[731],[0],obs,info)
            self.assertEqual(r['retire_next_forward'],[])
    def test_scene_and_nonmonotonic_time_reset_history(self):
        _,gate,obs,info,box=self.replay();info=dict(info,timestamp=0.,scene_token='next');r=gate.observe(box[None],[731],[0],obs,info)
        self.assertTrue(r['reset']);self.assertEqual(r['retire_next_forward'],[])
    def test_duplicate_ids_do_not_manufacture_temporal_evidence(self):
        obs,info,box=fixture();g=QueryLifecycle(CFG)
        with self.assertRaises(ValueError):g.observe(np.stack([box,box]),[12,12],[0,0],obs,info)
    def test_partial_view_surface_or_occluder_protects_unknown_center(self):
        for observed_depth in (6.9,1.):
            obs,info,box=fixture();box[:2]=[9.,5.]
            obs['depth']['cam'][:]=observed_depth
            gate=QueryLifecycle(CFG)
            for i in range(4):
                b=box.copy();b[0]+=i*.3
                result=gate.observe(b[None],[731],[0],obs,dict(info,timestamp=i*.1))
                self.assertEqual(result['retire_next_forward'],[])
    def test_real_instances_preserve_fixed_fresh_pool_and_aligned_tensors(self):
        import importlib.util,torch
        path=ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad/dense_heads/track_head_plugin/track_instance.py'
        spec=importlib.util.spec_from_file_location('actual_track_instances',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        inst=module.Instances((1,1));inst.obj_idxes=torch.tensor([-1]*901+[22,33,44]);inst.query=torch.arange(904*2).reshape(904,2).float();inst.ref_pts=torch.zeros(904,3)
        out=retire_queries(inst,[22,44]);self.assertEqual(len(out),902)
        self.assertEqual(out.obj_idxes[:901].tolist(),[-1]*901);self.assertEqual(out.obj_idxes[-1].item(),33)
        torch.testing.assert_close(out.query[-1],inst.query[902]);self.assertEqual(len(inst),904)
        # The original forward marks row900 as the ego after fresh/active concat.
        reordered=module.Instances.cat([out[out.obj_idxes<0],out[out.obj_idxes>=0]])
        self.assertEqual(reordered.obj_idxes[900].item(),-1)

    def test_worker_defers_retirement_and_never_filters_current_reports(self):
        import importlib.util,json,pickle,tempfile,torch,types
        from unittest.mock import patch
        spec=importlib.util.spec_from_file_location('actual_instances_for_worker',ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad/dense_heads/track_head_plugin/track_instance.py')
        im=importlib.util.module_from_spec(spec);spec.loader.exec_module(im)
        for mode in ('baseline','repair'):
            with tempfile.TemporaryDirectory() as temp:
                job=Path(temp);(job/'lifecycle_mode.json').write_text(json.dumps(dict(mode=mode)))
                obs,info,box=fixture();seen=[]
                class Reliability:
                    def update(self,*args,**kwargs):return dict(config=CFG,obstacles=[dict(candidate_index=0,state='network_retained')])
                core=types.SimpleNamespace(scene_reliability=Reliability(),track_base=types.SimpleNamespace(max_obj_id=1000))
                def original_forward(track_instances):seen.append(len(track_instances));return {}
                core._forward_single_frame_inference=original_forward
                def runner(model,namespace):
                    for n in range(4):
                        d=job/('step_%03d'%n);d.mkdir();b=box.copy();b[0]+=n*.4
                        with (d/'observation.pkl').open('wb') as out:pickle.dump((obs,dict(info,timestamp=n*.1)),out)
                        gravity=b.copy();gravity[2]+=gravity[5]/2
                        (d/'neural_query_origin.json').write_text(json.dumps(dict(decoded_gravity_boxes=[gravity.tolist()],selected_track_ids=[731])))
                        inst=im.Instances((1,1));inst.obj_idxes=torch.tensor([-1]*901+[731]);inst.query=torch.zeros(902,2)
                        core._forward_single_frame_inference(track_instances=inst)
                        returned=core.scene_reliability.update(b[None],torch.tensor([.9]),torch.tensor([0]))
                        self.assertEqual(returned['obstacles'],[dict(candidate_index=0,state='network_retained')])
                        self.assertEqual(core.track_base.max_obj_id,1000)
                fake=types.SimpleNamespace(original_serve=runner,worker=types.SimpleNamespace(JOB=job,atomic_json=lambda path,data:path.write_text(json.dumps(data))))
                spec=importlib.util.spec_from_file_location('isolated_lifecycle_worker',HERE/'worker.py');module=importlib.util.module_from_spec(spec)
                with patch.dict(sys.modules,{'origin_worker':fake}):spec.loader.exec_module(module)
                module.serve(core,None)
                self.assertEqual(seen,[902,902,902,901] if mode=='repair' else [902]*4)

    def test_cache_filter_preserves_fresh_ego_fields_and_source(self):
        ids=[-1,-1,-2,2,4,7];inst=Instances(ids,list(range(6)))
        out=retire_queries(inst,[2,7]);np.testing.assert_array_equal(out.obj_idxes,[-1,-1,-2,4]);np.testing.assert_array_equal(out.values,[0,1,2,4]);np.testing.assert_array_equal(inst.obj_idxes,ids)
        one=retire_queries(inst,[4]);np.testing.assert_array_equal(one.obj_idxes,[-1,-1,-2,2,7])
        self.assertIs(retire_queries(inst,[]),inst)
        with self.assertRaises(ValueError):retire_queries(inst,[-2])
if __name__=='__main__':unittest.main()
