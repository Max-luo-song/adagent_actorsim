
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

import copy,hashlib,json,pickle,sys,types,unittest
pass  # Path is provided by the portable boundary above
import numpy as np
HERE=Path(__file__).resolve().parent;CR=HERE.parents[1];ROOT=CR.parents[4]
sys.path[:0]=[str(HERE.parent),str(CR/'phase28/proposal_admission_v1'),str(CR/'phase27'),str(CR/'phase28/heading_fix_v1'),str(ROOT/'waymo23-native-import'),str(ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad')]
from global_repair_v1.pipeline import EvidencePipeline,validate_memory_input
from global_repair_v1.snapshot import extend_apply,KEY,COUNTER
from global_repair_v1.driver import memory_class
from test_evidence import fixture,CFG

class PipelineTests(unittest.TestCase):
    def run_frame(self,pipe,index,identity=731,extra=False,token='test'):
        obs,info,box=fixture();box[0]+=.4*index;info=dict(info,timestamp=index*.1,scene_token=token)
        boxes=box[None];ids=[identity];labels=np.array([0]);scores=np.array([.41])
        if extra:boxes=np.vstack([boxes,box]);ids.append(-1);labels=np.array([0,0]);scores=np.array([.41,.35])
        report=dict(config=CFG,raw_scores_modified=False,obstacles=[dict(candidate_index=i,raw_score=float(scores[i]),box=b.tolist(),label=0,state='network_retained') for i,b in enumerate(boxes)])
        result=pipe.process(report,boxes,boxes,scores,labels,ids,obs,info)
        return result,boxes,scores,labels,info
    def test_snapshot_restores_identical_next_decision(self):
        p=EvidencePipeline(CFG)
        for i in range(3):self.run_frame(p,i)
        state=p.capture();expected=self.run_frame(p,3)[0]
        restored=EvidencePipeline(CFG);restored.restore(state)
        self.assertEqual(self.run_frame(restored,3)[0],expected)
    def test_corruption_or_incompatible_snapshot_does_not_mutate_live_state(self):
        p=EvidencePipeline(CFG);self.run_frame(p,0);before=p.capture()
        bad=dict(before,sha256='bad')
        with self.assertRaises(ValueError):p.restore(bad)
        self.assertEqual(p.capture(),before)
        other=EvidencePipeline(dict(CFG,caution_speed_mps=1.))
        with self.assertRaises(ValueError):other.restore(before)
    def test_clock_rejection_and_scene_reset(self):
        p=EvidencePipeline(CFG);self.run_frame(p,0);before=p.capture()
        with self.assertRaises(ValueError):self.run_frame(p,0)
        self.assertEqual(before,p.capture())
        r=self.run_frame(p,0,token='other')[0];self.assertEqual(r['global_evidence_receipt']['frame_count'],1)
        self.assertEqual(r['global_evidence']['queries'][0]['history_observations'],1)
    def test_fresh_duplicate_inherits_shape_status_not_temporal_confirmation(self):
        p=EvidencePipeline(CFG)
        for i in range(3):self.run_frame(p,i)
        r=self.run_frame(p,3,extra=True)[0]
        self.assertEqual(r['obstacles'],[])
        fresh=next(q for q in r['global_evidence']['queries'] if q['neural_id']==-1)
        self.assertTrue(fresh['untrusted_geometry']);self.assertEqual(fresh['history_observations'],0)
        self.assertNotIn(-1,p.observer.records)
    def test_receipt_rejects_wrong_arrays_clock_and_double_memory(self):
        p=EvidencePipeline(CFG);r,b,s,l,info=self.run_frame(p,0)
        validate_memory_input(r,b,s,l,info)
        with self.assertRaises(ValueError):validate_memory_input(r,b,s+.01,l,info)
        with self.assertRaises(ValueError):validate_memory_input(r,b,s,l,dict(info,timestamp=.1))
        class Memory:
            def update(self,report,*args,**kwargs):return copy.deepcopy(report)
        m=memory_class(Memory)();out=m.update(r,b,s,l,info)
        with self.assertRaises(ValueError):m.update(out,b,s,l,info)
    def test_incomplete_or_counter_reusing_model_restore_is_rejected_first(self):
        p=EvidencePipeline(CFG);self.run_frame(p,0,identity=3);called=[];restore=extend_apply(lambda core,payload:called.append(True),p)
        with self.assertRaises(ValueError):restore(None,{})
        payload={KEY:p.capture(),COUNTER:dict(max_obj_id=4),'adapter':dict(scene_token='test',timestamp=0.), 'test_track_instances':dict(fields=dict(obj_idxes=dict(data=np.array([-1,-2,4]))))}
        with self.assertRaises(ValueError):restore(None,payload)
        self.assertEqual(called,[])
        payload[COUNTER]['max_obj_id']=5;restore(None,payload);self.assertEqual(called,[True])

    def test_invalid_frame_does_not_advance_evidence_state(self):
        p=EvidencePipeline(CFG);self.run_frame(p,0);before=p.capture()
        obs,info,box=fixture();info=dict(info,timestamp=.1,explicit_l2g_r_mat=np.zeros((2,2)))
        report=dict(config=CFG,obstacles=[])
        with self.assertRaises(ValueError):p.process(report,box[None],box[None],[.4],np.array([0]),np.array([731]),obs,info)
        self.assertEqual(p.capture(),before)
        with self.assertRaises(ValueError):p.process(report,box[None],box[None],[[.4]],np.array([0]),np.array([731]),obs,info)
        self.assertEqual(p.capture(),before)

    def test_real_model_snapshot_preserves_bev_alias_counter_rng_and_evidence(self):
        import importlib.util,random,torch
        import branch_state
        from origin_state import extend_capture as counter_capture,extend_apply as counter_apply
        from global_repair_v1.snapshot import extend_capture
        path=CR/'phase27/test_branch_state.py';spec=importlib.util.spec_from_file_location('actual_branch_state_fixture',path);fixture_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture_module)
        core=fixture_module._fake_core();core.track_base=types.SimpleNamespace(max_obj_id=1000,score_thresh=.4,filter_score_thresh=.35,miss_tolerance=5)
        p=EvidencePipeline(CFG)
        for i in range(3):self.run_frame(p,i)
        core._adapter_scene_token='test';core._adapter_timestamp=.2;core.timestamp=.2;core.scene_token='test'
        capture=extend_capture(counter_capture(branch_state.capture_worker_state),p)
        restore=extend_apply(counter_apply(branch_state.apply_worker_state),p)
        payload,meta=capture(core,index=3,previous_time=.2)
        expected_rng=(random.random(),np.random.rand(),torch.rand(2));expected=self.run_frame(p,3)[0]
        core.prev_bev=None;core.prev_frame_info['prev_bev']=None;core.track_base.max_obj_id=99;p.reset()
        restore(core,payload)
        actual_rng=(random.random(),np.random.rand(),torch.rand(2))
        self.assertIs(core.prev_frame_info['prev_bev'],core.prev_bev);self.assertEqual(core.track_base.max_obj_id,1000)
        self.assertEqual(actual_rng[:2],expected_rng[:2]);torch.testing.assert_close(actual_rng[2],expected_rng[2])
        self.assertEqual(self.run_frame(p,3)[0],expected)

if __name__=='__main__':unittest.main()
