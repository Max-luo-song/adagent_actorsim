"""Real origin wrapper/signature/decoder map; external-only admission invariants."""

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

import ast,copy,importlib.util,json,pickle,sys,tempfile,types,unittest
pass  # Path is provided by the portable boundary above
from unittest.mock import patch
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;P28=HERE.parents[1]/'phase28';ROOT=P28.parents[5]
sys.path[:0]=[str(HERE.parent),str(P28/'proposal_admission_v1'),str(P28/'heading_fix_v1'),str(ROOT/'waymo23-native-import'),str(ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad')]
from test_evidence import CFG,fixture

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class WorkerChainTests(unittest.TestCase):
    def test_model_queries_scores_and_context_unchanged_in_both_modes(self):
        for mode in ('baseline','global'):
            origin=load('actual_origin_admission',P28/'heading_fix_v1/origin_worker.py')
            Instances=load('actual_instances_admission',ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad/dense_heads/track_head_plugin/track_instance.py').Instances
            source=ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad/detectors/uniad_track.py';tree=ast.parse(source.read_text())
            method=copy.deepcopy(next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_forward_single_frame_inference'))
            method.decorator_list=[];method.body=ast.parse('return self._cpu_forward(img,img_metas,track_instances,prev_bev,l2g_r1,l2g_t1,l2g_r2,l2g_t2,time_delta)').body
            scope={};exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(source),'exec'),scope)
            def decode_base(self,instances,results,img_metas):return self.bbox_coder.decode(dict(obj_idxes=instances[instances.obj_idxes!=-2].obj_idxes))
            Core=type('CpuCore',(),{'_forward_single_frame_inference':scope['_forward_single_frame_inference'],'_det_instances2results':decode_base})
            core=Core();core.track_base=types.SimpleNamespace(max_obj_id=1000);core.prev_bev=object();bev=core.prev_bev
            image=object();metadata=object();rotation=np.eye(3);translation=np.zeros(3);seen=[];returns=[]
            class Reliability:
                def update(self,boxes,scores,labels):
                    return dict(config=CFG,obstacles=[dict(candidate_index=i,label=int(labels[i]),state='network_retained',raw_score=float(scores[i]),box=boxes[i].tolist()) for i in range(2)],speed_limit_mps=None,action='unchanged')
            core.scene_reliability=Reliability()
            def bbox_decode(predictions):
                selection=torch.nonzero(predictions['obj_idxes']>=0).flatten();gravity=core.boxes.copy();gravity[:,2]+=gravity[:,5]/2
                return [dict(bbox_index=selection,mask=torch.tensor([True,True]),bboxes=torch.tensor(gravity))]
            core.bbox_coder=types.SimpleNamespace(decode=bbox_decode)
            def cpu_forward(img,img_metas,track_instances,prev_bev,l2g_r1,l2g_t1,l2g_r2,l2g_t2,time_delta):
                self.assertIs(img,image);self.assertIs(img_metas,metadata);self.assertIs(prev_bev,bev)
                self.assertIs(l2g_r1,rotation);self.assertIs(l2g_r2,rotation);self.assertIs(l2g_t1,translation);self.assertIs(l2g_t2,translation);self.assertEqual(time_delta,.1)
                seen.append(track_instances.obj_idxes.clone())
                full=Instances.cat([track_instances[track_instances.obj_idxes<0],track_instances[track_instances.obj_idxes>=0]])
                full.obj_idxes[900]=-2;core._det_instances2results(full,{},None);return {}
            core._cpu_forward=cpu_forward
            with tempfile.TemporaryDirectory() as temp:
                job=Path(temp);origin.worker.JOB=job;(job/'global_mode.json').write_text(json.dumps(dict(mode=mode)))
                obs,info,box=fixture();human=box.copy();human[:2]=[0.,10.]
                def runner(model,namespace):
                    for i in range(4):
                        d=job/('step_%03d'%i);d.mkdir();b=box.copy();b[0]+=i*.4;core.boxes=np.stack([b,human])
                        with (d/'observation.pkl').open('wb') as f:pickle.dump((obs,dict(info,timestamp=i*.1)),f)
                        incoming=Instances((1,1));incoming.obj_idxes=torch.tensor([-1]*901+[731,888]);incoming.query=torch.zeros(903,2)
                        core.test_track_instances=incoming;ids=incoming.obj_idxes.clone();query=incoming.query.clone()
                        if i%2:core._forward_single_frame_inference(img=image,img_metas=metadata,track_instances=incoming,prev_bev=bev,l2g_r1=rotation,l2g_t1=translation,l2g_r2=rotation,l2g_t2=translation,time_delta=.1)
                        else:core._forward_single_frame_inference(image,metadata,incoming,bev,rotation,translation,rotation,translation,.1)
                        scores=torch.tensor([.41,.69624]);labels=torch.tensor([0,8]);score_copy=scores.clone();box_copy=core.boxes.copy()
                        returned=core.scene_reliability.update(core.boxes,scores,labels);returns.append(returned)
                        torch.testing.assert_close(scores,score_copy);np.testing.assert_array_equal(core.boxes,box_copy)
                        self.assertIs(core.test_track_instances,incoming);self.assertIs(core.prev_bev,bev);self.assertEqual(core.track_base.max_obj_id,1000)
                        torch.testing.assert_close(incoming.obj_idxes,ids);torch.testing.assert_close(incoming.query,query)
                        q=_hugsim_expand(json.loads((d/'neural_query_origin.json').read_text()));self.assertEqual(q['selected_track_ids'],[731,888]);self.assertFalse(q['decode_input_has_special_ego'])
                        expected=[0,1] if mode=='baseline' or i<2 else [1]
                        self.assertEqual([r['candidate_index'] for r in returned['obstacles']],expected)
                        self.assertEqual(returned['global_evidence_receipt']['stage'],'before_driver_memory')
                        self.assertEqual(returned['global_evidence_receipt']['frame_count'],i+1)
                        visible=next(r for r in returned['obstacles'] if r['candidate_index']==1)
                        self.assertAlmostEqual(visible['raw_score'],.69624,places=6)
                from dataclasses import make_dataclass
                Config=make_dataclass('FixtureConfig',[(k,type(v)) for k,v in CFG.items()])
                with patch.dict(sys.modules,{'origin_worker':origin}):worker=load('global_test_worker',HERE/'worker.py')
                worker.original_serve=runner;original=origin.original_serve;origin.original_serve=worker.serve
                try:
                    with patch('runtime_configuration.configured_reliability',return_value=Config(**CFG)):origin.serve(core,None)
                finally:origin.original_serve=original
                self.assertEqual([len(v) for v in seen],[903]*4)
                for ids in seen:self.assertEqual(ids.tolist(),[-1]*901+[731,888])

if __name__=='__main__':unittest.main()
