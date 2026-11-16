
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

import copy
pass  # Path is provided by the portable boundary above
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from contracts import appearance_parameters, masked_l1, merge_checkpoint, check_checkpoint_invariants, split_frame, pose
from pipeline import asset, filter_target, validate_rows, write, target_rows, distill, evaluate, read


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._features_dc = torch.nn.Parameter(torch.zeros(1,3))
        self._features_rest = torch.nn.Parameter(torch.zeros(1,3))
        self._means = torch.nn.Parameter(torch.ones(1,3))


class Tests(unittest.TestCase):
    def test_time_split_no_camera_leak(self):
        train = {i for i in range(61) if split_frame(i)=='train'}
        holdout = {i for i in range(61) if split_frame(i)=='eval'}
        self.assertTrue(train and holdout)
        self.assertFalse(train & holdout)
        self.assertEqual(holdout, set(range(4,61,5)))

    def test_reject_invalid_pose(self):
        x=np.eye(4);x[0,0]=2
        with self.assertRaises(ValueError):pose(x)
        with self.assertRaises(ValueError):pose(np.zeros((4,4)))

    def test_teacher_large_change_filtered(self):
        a=np.zeros((10,10,3),dtype='uint8');b=a.copy();b[:5]=255
        mask, quality=filter_target(a,b,np.ones((10,10),bool))
        self.assertFalse(mask[:5].any());self.assertTrue(mask[5:].all())
        self.assertTrue(quality['accepted']);self.assertFalse(quality['semantics_certified'])

    def test_teacher_cannot_resize(self):
        with self.assertRaises(ValueError):filter_target(np.zeros((8,8,3)),np.zeros((16,16,3)),np.ones((8,8),bool))

    def test_empty_mask_rejected(self):
        mask,q=filter_target(np.zeros((10,10,3)),np.ones((10,10,3))*255,np.ones((10,10),bool))
        self.assertFalse(mask.any());self.assertFalse(q['accepted'])
        with self.assertRaises(ValueError):masked_l1(torch.zeros(2,2,3),torch.ones(2,2,3),torch.zeros(2,2,dtype=torch.bool))

    def test_masked_gradient(self):
        p=torch.zeros(2,2,3,requires_grad=True);target=torch.ones_like(p);mask=torch.tensor([[1,0],[0,0]],dtype=torch.bool)
        loss=masked_l1(p,target,mask);loss.backward()
        self.assertGreater(p.grad[0,0].abs().sum(),0)
        self.assertEqual(p.grad[1].abs().sum(),0)
        self.assertEqual(p.grad[0,1].abs().sum(),0)

    def test_appearance_only_optimizer_with_unregistered_models(self):
        trainer=torch.nn.Module();trainer.models={n:Model() for n in ['Background','RoadNodes','RigidNodes','CamPose']}
        selected=appearance_parameters(trainer)
        self.assertEqual(len(selected),4)
        optimizer=torch.optim.Adam(list(selected.values()),lr=.1)
        initial={n:copy.deepcopy(m.state_dict()) for n,m in trainer.models.items()}
        for _ in range(20):
            optimizer.zero_grad();loss=sum(((p-1)**2).mean() for p in selected.values());loss.backward();optimizer.step()
        self.assertLess(float(loss),.5)
        for n,m in trainer.models.items():
            self.assertTrue(torch.equal(m._means,initial[n]['_means']))
            if n in ['RigidNodes','CamPose']:
                self.assertFalse(m._features_dc.requires_grad)
                self.assertTrue(torch.equal(m._features_dc,initial[n]['_features_dc']))

    def checkpoint(self):
        return dict(models={n:Model().state_dict() for n in ['Background','RigidNodes']},step=40000,optimizer={'stale':1},misc=[torch.tensor([2]),'metadata'])

    def test_checkpoint_replacement_preserves_all_other_state(self):
        original=self.checkpoint();candidate=merge_checkpoint(copy.deepcopy(original),{('Background','_features_dc'):torch.ones(1,3)})
        changed=check_checkpoint_invariants(original,candidate)
        self.assertEqual(changed,['models.Background._features_dc'])
        self.assertEqual(candidate['step'],40000)
        self.assertNotIn('optimizer',candidate)

    def test_checkpoint_forbidden_geometry_or_actor_update(self):
        for n,f in [('Background','_means'),('RigidNodes','_features_dc')]:
            original=self.checkpoint();candidate=copy.deepcopy(original);candidate['models'][n][f]+=1
            with self.assertRaises(ValueError):check_checkpoint_invariants(original,candidate)
            with self.assertRaises(ValueError):merge_checkpoint(original,{(n,f):torch.ones(1,3)})

    def test_noop_checkpoint_rejected(self):
        a=self.checkpoint()
        with self.assertRaises(ValueError):check_checkpoint_invariants(a,copy.deepcopy(a))

    def rows(self,d):
        p=Path(d)/'rgb.png';Image.new('RGB',(8,8)).save(p)
        rows=[]
        for f,s in [(0,'train'),(4,'eval')]:
            rows.append(dict(id=str(f),source_frame=f,camera_id=0,kind='novel',split=s,scene=551,checkpoint_sha256='abc',edit_revision=0,
                ego_to_world=np.eye(4).tolist(),camera_to_world=np.eye(4).tolist(),K=np.eye(3).tolist(),
                before=asset(p),reference=asset(p),mask=asset(p)))
        return rows

    def test_dataset_hash_and_revision_rejections(self):
        with tempfile.TemporaryDirectory() as d:
            rows=self.rows(d);validate_rows(rows)
            edited=copy.deepcopy(rows);edited[0]['edit_revision']=1
            with self.assertRaises(ValueError):validate_rows(edited)
            Path(rows[0]['before']['path']).write_bytes(b'changed')
            with self.assertRaises(ValueError):validate_rows(rows)

    def test_dataset_no_duplicate_or_camera_split_leak(self):
        with tempfile.TemporaryDirectory() as d:
            rows=self.rows(d)
            with self.assertRaises(ValueError):validate_rows(rows+[rows[0]])
            extra=copy.deepcopy(rows[0]);extra.update(camera_id=1,split='eval')
            with self.assertRaises(ValueError):validate_rows(rows+[extra])

    def test_dataset_mixed_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for field,value in [('scene',552),('checkpoint_sha256','def')]:
                rows=self.rows(d);rows[1][field]=value
                with self.assertRaises(ValueError):validate_rows(rows)

    def test_identity_teacher_cannot_be_admitted(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'pseudo_gt').mkdir()
            write(out/'job.json',{});write(out/'compiled_trajectory.json',{})
            write(out/'render.json',dict(completed=True,job=asset(out/'job.json'),trajectory=asset(out/'compiled_trajectory.json')))
            write(out/'pseudo_gt/manifest.json',dict(completed=True,identity_fallback=True,render_manifest=asset(out/'render.json'),rows=[]))
            with self.assertRaisesRegex(ValueError,'Actual DiFix'):target_rows(out)

    def test_training_checkpoint_reload_and_heldout_evaluation(self):
        # Exercise real stage orchestration/autograd/save/reload. Only rasterization is a CPU fake.
        class FakeRenderer:
            def __init__(self, scene, bundle):
                self.trainer=torch.nn.Module();self.trainer.models={'Background':Model()}
                if Path(scene).parent.name=='distill':
                    state=torch.load(Path(scene).parent/'checkpoint_distilled.pth',weights_only=False)
                    self.trainer.models['Background'].load_state_dict(state['models']['Background'])
            def frame(self,*args):return {}
            def rgb(self,data):
                p=self.trainer.models['Background']._features_dc
                return {'rgb':p.sigmoid().expand(8,8,3)}
            def close(self):pass
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);checkpoint=out/'base.pth'
            base=dict(models={'Background':Model().state_dict()},step=40000,extra='frozen')
            torch.save(base,checkpoint)
            write(out/'config.yaml',{});write(out/'source.json',{});write(out/'scene.json',{'vehicle_config':str(out/'vehicle.json')})
            (out/'pseudo_gt').mkdir();write(out/'pseudo_gt/manifest.json',{'fixture':True})
            write(out/'compiled_trajectory.json',{'fps':10})
            before=out/'before.png';target=out/'target.png';mask=out/'mask.png'
            Image.new('RGB',(8,8),(128,128,128)).save(before)
            Image.new('RGB',(8,8),(160,160,160)).save(target)
            Image.new('L',(8,8),255).save(mask)
            rows=[dict(id=f'{kind}_{split}',source_frame=frame,camera_id=0,kind=kind,split=split,accepted=True,
                ego_to_world=np.eye(4).tolist(),target=asset(target),valid_mask=asset(mask),before=asset(before),target_kind='real_original_gt' if kind=='original' else 'difix_pseudo_gt')
                for kind in ['novel','original'] for frame,split in [(0,'train'),(4,'eval')]]
            job=dict(task={'renderer_bundle':'fixture'},learning_rate=.02,epochs=3,
                inputs={k:asset(p) for k,p in [('checkpoint',checkpoint),('config',out/'config.yaml'),('source_report',out/'source.json'),('scene_manifest',out/'scene.json')]})
            with patch('pipeline.job_at',return_value=(out,job)),patch('pipeline.target_rows',return_value=rows),patch('framework.Renderer',FakeRenderer):
                result=distill(out)
                self.assertEqual(result['steps'],9)
                receipt=read(out/'distill/receipt.json')
                self.assertEqual(set(receipt['trained_sample_ids']),{'novel_train','original_train'})
                with patch('pipeline.make_comparison'):
                    evaluate(out)
                metrics=read(out/'evaluation/report.json')
                self.assertTrue(metrics['novel_teacher_similarity_improved'])
                self.assertFalse(metrics['formal_evaluation_admitted'])
                saved=torch.load(out/'distill/checkpoint_distilled.pth',weights_only=False)
                self.assertTrue(torch.equal(saved['models']['Background']['_means'],base['models']['Background']['_means']))
                self.assertEqual(saved['step'],40000)


if __name__=='__main__':unittest.main()
