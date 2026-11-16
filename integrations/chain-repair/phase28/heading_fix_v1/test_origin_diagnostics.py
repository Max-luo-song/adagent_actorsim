"""Meaningful CPU checks of snapshot isolation and real decoder provenance."""

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

import ast
import json
import os
import subprocess
pass  # Path is provided by the portable boundary above
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parents[5]
sys.path[:0]=[str(HERE),str(HERE.parent.parent/'phase27'),str(ROOT/'waymo23-native-import/tests')]
import branch_state
import origin_worker
import run_origin_inference
from origin_state import extend_capture,extend_apply,KEY
from test_decoded_box_origin import load_methods

def core_state():
    return SimpleNamespace(prev_bev=None,prev_frame_info=dict(scene_token='s',prev_pos=0,prev_angle=0,prev_bev=None),
        test_track_instances=None,timestamp=1.,scene_token='s',l2g_t=None,l2g_r_mat=None,
        scene_reliability=SimpleNamespace(scene_token='s',previous_time=1.,tracks=[],next_id=0),
        track_base=SimpleNamespace(max_obj_id=7,score_thresh=.4,filter_score_thresh=.35,miss_tolerance=5))

class OriginTests(unittest.TestCase):
    def test_worker_entrypoint_activates_runtime_before_numerical_imports(self):
        code='''import runpy, sys, types
from pathlib import Path
path=Path(sys.argv[1])
sys.path.insert(0,str(path.parent))
def activate():
    assert 'numpy' not in sys.modules and 'torch' not in sys.modules
    print('EARLY_RUNTIME_BOOTSTRAP_OK')
sys.modules['runtime_bundle']=types.SimpleNamespace(activate_configured=activate)
sys.argv=[str(path),'--help']
runpy.run_path(str(path),run_name='__main__')
'''
        result=subprocess.run([sys.executable,'-c',code,str(HERE/'origin_worker.py')],capture_output=True,text=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('EARLY_RUNTIME_BOOTSTRAP_OK',result.stdout)
        self.assertIn('--model-profile',result.stdout)
    def test_bootstrap_packet_uses_original_frame_without_copy_or_write(self):
        with tempfile.TemporaryDirectory(prefix='origin-bootstrap-') as directory:
            target=run_origin_inference.prepare_bootstrap_packet(directory)
            self.assertTrue(target.is_file())
            self.assertTrue(target.is_symlink())
            self.assertEqual(target.resolve(),Path(run_origin_inference.original_request(0)['packet']).resolve())
            with self.assertRaises(FileExistsError):run_origin_inference.prepare_bootstrap_packet(directory)

    def test_real_five_uniad_initializer_accepts_prepared_bootstrap(self):
        import five_uniad
        with tempfile.TemporaryDirectory(prefix='origin-initializer-') as directory:
            packet=run_origin_inference.prepare_bootstrap_packet(directory)
            callback=[]
            # Execute the actual infer() preflight and AST-rewrite path. Only
            # the model construction body is a CPU stub; no GPU claim.
            stub=ast.parse('def main():\n    model = object()\n    cameras = []\nif __name__ == "__main__":\n    main()\n')
            real_parse=ast.parse
            source_text=(ROOT/'code/UniAD_SIM/tools/closeloop/e2e.py').read_text()
            def parse_initializer(text,*args,**kwargs):
                return stub if text==source_text else real_parse(text,*args,**kwargs)
            cwd=Path.cwd();argv=sys.argv[:]
            try:
                with patch.dict(os.environ,{'BASE':str(ROOT),'FIVE_JOB':str(packet.parent)}), \
                     patch.object(five_uniad.ast,'parse',side_effect=parse_initializer), \
                     patch.object(five_uniad,'infer_five',side_effect=lambda *args:callback.append(True)), \
                     patch.dict(sys.modules,{'runtime_bundle':SimpleNamespace(checkpoint_path=lambda:None)}):
                    five_uniad.infer()
            finally:
                os.chdir(cwd);sys.argv=argv
            self.assertEqual(callback,[True])
            self.assertTrue((packet.parent/'prediction').is_dir())
    def test_replay_uses_original_seed_and_exact_request_fields(self):
        request=run_origin_inference.original_request(0)
        self.assertEqual(request['diagnostic_seed'],20261021)
        self.assertEqual(request['index'],0)
        self.assertNotIn('diagnostic_seed',run_origin_inference.original_request(39))
    def test_actual_legacy_snapshot_does_not_restore_tracker_counter(self):
        core=core_state();payload,_=branch_state.capture_worker_state(core,index=39,previous_time=1.)
        self.assertNotIn(KEY,payload)
        core.track_base.max_obj_id=93
        branch_state.apply_worker_state(core,payload)
        self.assertEqual(core.track_base.max_obj_id,93)

    def test_new_snapshot_restores_counter_and_rejects_incomplete_old_snapshot(self):
        core=core_state();capture=extend_capture(branch_state.capture_worker_state);apply=extend_apply(branch_state.apply_worker_state)
        payload,_=capture(core,index=39,previous_time=1.)
        core.track_base.max_obj_id=93;apply(core,payload)
        self.assertEqual(core.track_base.max_obj_id,7)
        del payload[KEY]
        with self.assertRaises(ValueError):apply(core,payload)

    def test_counter_omission_can_duplicate_real_tracker_ids(self):
        source=ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad/dense_heads/track_head_plugin/tracker.py'
        node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef))
        scope=dict(Instances=object)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),scope)
        tracker=scope['RuntimeTrackerBase'](.4,.35,5)
        class Tracks:
            obj_idxes=torch.tensor([0,-1]);scores=torch.tensor([.8,.8]);disappear_time=torch.zeros(2)
            def __len__(self):return 2
        tracks=Tracks();tracker.update(tracks)
        self.assertEqual(tracks.obj_idxes.tolist(),[0,0])

    def test_all_ablation_modes_preserve_real_export_and_query_mapping(self):
        methods=load_methods()
        class Instances:
            def __init__(self,ids):
                self.obj_idxes=torch.tensor(ids);self.pred_logits=torch.ones(len(ids),10)
                self.pred_boxes=torch.zeros(len(ids),10);self.scores=torch.ones(len(ids))*.6
            def __getitem__(self,key):return Instances(self.obj_idxes[key].tolist())
        class Core:
            def __init__(self):
                self.bbox_coder=SimpleNamespace(decode=self.coder);self.track_base=core_state().track_base;self.inputs=[]
            def _generate_empty_tracks(self):return Instances([-1,-1,-2])
            def _forward_single_frame_inference(self,img,track_instances,prev_bev=None):
                self.inputs.append((track_instances.obj_idxes.tolist(),prev_bev))
                ids=track_instances.obj_idxes
                return Instances(torch.cat([ids[ids<0],ids[ids>=0]]).tolist())
            def _det_instances2results(self,instances,results,img_metas):return methods['_det_instances2results'](self,instances,results,img_metas)
            def coder(self,predictions,**kwargs):
                n=len(predictions['obj_idxes']);boxes=torch.zeros(n,9);boxes[:,3:6]=1
                return [dict(bboxes=boxes,labels=torch.zeros(n,dtype=torch.long),scores=torch.ones(n)*.6,
                    track_scores=torch.ones(n)*.6,obj_idxes=predictions['obj_idxes'],bbox_index=torch.arange(n),mask=torch.ones(n,dtype=torch.bool))]
        core=Core();old_serve=origin_worker.original_serve;old_job=origin_worker.worker.JOB
        with tempfile.TemporaryDirectory(prefix='query-origin-') as folder:
            job=Path(folder);(job/'step_039').mkdir();origin_worker.worker.JOB=job
            def serve(model,namespace):
                for mode in ('baseline','drop_bev','drop_tracks','drop_both'):
                    (job/'origin_mode.json').write_text(json.dumps(dict(mode=mode)))
                    instances=model._forward_single_frame_inference('img',Instances([-1,7,-2]),'bev')
                    result=model._det_instances2results(instances,[{}],[])[0]
                    self.assertEqual(len(result['scores_3d_det']),2)
                    audit=_hugsim_expand(json.loads((job/'step_039/neural_query_origin.json').read_text()))
                    self.assertFalse(audit['decode_input_has_special_ego'])
                    self.assertEqual(audit['prev_bev_present'],mode not in ('drop_bev','drop_both'))
                    expected=['fresh_or_inactive_query']*2 if mode in ('drop_tracks','drop_both') else ['fresh_or_inactive_query','historical_active_track']
                    self.assertEqual(audit['selected_source_kind'],expected)
                    self.assertNotIn(-2,audit['selected_preupdate_track_ids'])
            origin_worker.original_serve=serve
            try:origin_worker.serve(core,{})
            finally:origin_worker.original_serve=old_serve;origin_worker.worker.JOB=old_job
        self.assertIs(core._forward_single_frame_inference.__func__,Core._forward_single_frame_inference)
        self.assertIs(core._det_instances2results.__func__,Core._det_instances2results)

if __name__=='__main__':unittest.main()
