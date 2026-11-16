"""Actual origin wrapper + actual UniAD signature/Instances, CPU-only decoder.

The NN tensors are stubbed; binding, reorder, ego exclusion, origin decode map,
query lifecycle and positional/keyword call shapes are the real wrapper chain.
"""

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

import ast,copy,importlib.util,json,os,pickle,sys,tempfile,types,unittest
pass  # Path is provided by the portable boundary above
from unittest.mock import patch
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;P28=HERE.parent;ROOT=P28.parents[5]
sys.path[:0]=[str(HERE),str(P28/'heading_fix_v1'),str(ROOT/'waymo23-native-import'),str(ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad')]
from test_lifecycle import CFG,fixture

def module_at(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class ForwardChainTests(unittest.TestCase):
    def exercise(self,folder,mode,expected_failure=False):
        origin=module_at('real_origin_for_chain',P28/'heading_fix_v1/origin_worker.py')
        instances_module=module_at('real_instances_for_chain',ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad/dense_heads/track_head_plugin/track_instance.py')
        Instances=instances_module.Instances
        source=ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad/detectors/uniad_track.py'
        tree=ast.parse(source.read_text())
        method=copy.deepcopy(next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_forward_single_frame_inference'))
        method.decorator_list=[]
        method.body=ast.parse('return self._cpu_forward(img,img_metas,track_instances,prev_bev,l2g_r1,l2g_t1,l2g_r2,l2g_t2,time_delta)').body
        scope={};exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(source),'exec'),scope)
        seen=[];queries=[];bev=object();img=object();metas=object();rotation=np.eye(3);translation=np.zeros(3)
        def decode_base(self,instances,results,img_metas):
            filtered=instances[instances.obj_idxes!=-2]
            return self.bbox_coder.decode(dict(obj_idxes=filtered.obj_idxes))
        Core=type('MinimalCore',(),{'_forward_single_frame_inference':scope['_forward_single_frame_inference'],'_det_instances2results':decode_base})
        core=Core();core.track_base=types.SimpleNamespace(max_obj_id=1000)
        class Reliability:
            def update(self,*args,**kwargs):return dict(config=CFG,obstacles=[dict(candidate_index=0,state='network_retained')])
        core.scene_reliability=Reliability()
        def bbox_decode(predictions):
            ids=predictions['obj_idxes'];where=torch.nonzero(ids>=0).flatten()
            assert len(where)==1
            gravity=core._model_box.copy();gravity[2]+=gravity[5]/2
            return [dict(bbox_index=where,mask=torch.tensor([True]),bboxes=torch.tensor(gravity[None]))]
        core.bbox_coder=types.SimpleNamespace(decode=bbox_decode)
        def cpu_forward(img_arg,metas_arg,track_instances,prev_bev,l2g_r1,l2g_t1,l2g_r2,l2g_t2,time_delta):
            self.assertIs(img_arg,img);self.assertIs(metas_arg,metas);self.assertIs(prev_bev,bev)
            self.assertIs(l2g_r1,rotation);self.assertIs(l2g_r2,rotation);self.assertIs(l2g_t1,translation);self.assertIs(l2g_t2,translation)
            self.assertEqual(time_delta,.1);seen.append(len(track_instances))
            reordered=Instances.cat([track_instances[track_instances.obj_idxes<0],track_instances[track_instances.obj_idxes>=0]])
            self.assertEqual(reordered.obj_idxes[900].item(),-1)
            reordered.obj_idxes[900]=-2
            if not bool((reordered.obj_idxes>=0).any()):reordered.obj_idxes[0]=999
            core._det_instances2results(reordered,{},None)
            return {}
        core._cpu_forward=cpu_forward
        with tempfile.TemporaryDirectory() as temp:
            job=Path(temp);origin.worker.JOB=job;(job/'lifecycle_mode.json').write_text(json.dumps(dict(mode=mode)))
            obs,info,box=fixture()
            def runner(model,namespace):
                for index in range(4):
                    d=job/('step_%03d'%index);d.mkdir();b=box.copy();b[0]+=index*.4;core._model_box=b
                    with (d/'observation.pkl').open('wb') as out:pickle.dump((obs,dict(info,timestamp=index*.1)),out)
                    incoming=Instances((1,1));incoming.obj_idxes=torch.tensor([-1]*901+[731]);incoming.query=torch.zeros(902,2)
                    if index%2==0:
                        core._forward_single_frame_inference(img,metas,incoming,bev,rotation,translation,rotation,translation,.1)
                    else:
                        core._forward_single_frame_inference(img=img,img_metas=metas,track_instances=incoming,prev_bev=bev,l2g_r1=rotation,l2g_t1=translation,l2g_r2=rotation,l2g_t2=translation,time_delta=.1)
                    q=_hugsim_expand(json.loads((d/'neural_query_origin.json').read_text()));queries.append(q)
                    self.assertFalse(q['decode_input_has_special_ego'])
                    self.assertEqual(q['selected_track_ids'],[731] if len(incoming)==seen[-1] else [999])
                    returned=core.scene_reliability.update(b[None],torch.tensor([.9]),torch.tensor([0]))
                    self.assertEqual(returned['obstacles'],[dict(candidate_index=0,state='network_retained')])
                    self.assertEqual(core.track_base.max_obj_id,1000)
            with patch.dict(sys.modules,{'origin_worker':origin}):
                worker=module_at('tested_lifecycle_worker',folder/'worker.py')
            worker.original_serve=runner
            if expected_failure:
                original=origin.original_serve;origin.original_serve=worker.serve
                try:
                    with self.assertRaisesRegex(KeyError,'track_instances'):origin.serve(core,None)
                finally:origin.original_serve=original
                self.assertEqual(seen,[])
            else:
                original=origin.original_serve;worker.entrypoint(core,None)
                self.assertIs(origin.original_serve,original)
                self.assertEqual(seen,[902,902,902,901] if mode=='repair' else [902]*4)
                self.assertEqual(queries[-1]['selected_source_kind'],['fresh_or_inactive_query'] if mode=='repair' else ['historical_active_track'])
                self.assertEqual(queries[-1]['selected_forward_input_indices'],[0] if mode=='repair' else [901])
    def test_v2_failure_reproduced_with_real_origin_wrapper(self):
        self.exercise(P28/'proposal_evidence_v2','baseline',expected_failure=True)
    def test_v3_real_wrapper_baseline_and_repair_positional_and_keyword_calls(self):
        for mode in ('baseline','repair'):
            with self.subTest(mode=mode):self.exercise(HERE,mode)

if __name__=='__main__':unittest.main()
