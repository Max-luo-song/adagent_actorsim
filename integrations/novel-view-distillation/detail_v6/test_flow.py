"""Exercise cache/probe/train/save/reload/evaluate with a differentiable CPU renderer."""

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

import contextlib
import copy
import io
pass  # Path is provided by the portable boundary above
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import experiment_v6 as e


class Node(torch.nn.Module):
    def __init__(self,background):
        super().__init__();self.use_pbr=background
        n=128 if background else 16
        for key,width in [('_features_dc',3),('_features_rest',3),('_scales',3),('_opacities',1)]:
            self.register_parameter(key,torch.nn.Parameter(torch.zeros(n,width)))
        self.register_parameter('_means',torch.nn.Parameter(torch.tensor([[0.,0.,5.]]*n)))
        self.register_parameter('_quats',torch.nn.Parameter(torch.tensor([[1.,0.,0.,0.]]*n)))


class ToyRenderer:
    state=None
    def __init__(self,scene):
        source=e.read(e.read(scene)['source_report'])
        state=torch.load(source['checkpoint'],map_location='cpu',weights_only=False)
        self.trainer=torch.nn.Module();self.trainer.models=dict(Background=Node(True),RoadNodes=Node(False))
        for key,node in self.trainer.models.items():node.load_state_dict(state['models'][key])
    def frame(self,*args):return {}
    def rgb(self,data):
        b=self.trainer.models['Background'];r=self.trainer.models['RoadNodes']
        x=torch.linspace(0,1,16);grid=(x[:,None]+x[None,:])/2
        color=(b._features_dc.mean(0)+r._features_dc.mean(0)+r._features_rest.mean(0))*.1
        scale=b._scales.mean();opacity=b._opacities.mean()
        rgb=.3+.3*grid[...,None]+color[None,None,:]+scale*.05*grid[...,None]+opacity*.05
        return dict(rgb=rgb.expand(16,16,3),depth=(5+scale*.1).expand(16,16,1),
                    opacity=(.95+opacity*.01).expand(16,16,1))
    def close(self):pass


class FlowTest(unittest.TestCase):
    def test_both_profiles_and_checkpoint_reload(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            out=Path(tmp);warm=out/'warm';warm.mkdir()
            nodes=dict(Background=Node(True),RoadNodes=Node(False))
            state=dict(models={k:n.state_dict() for k,n in nodes.items()},step=100)
            torch.save(state,warm/'model.pth');(warm/'config.yaml').write_text('fixture: true\n')
            e.write(warm/'source.json',dict(checkpoint=str(warm/'model.pth'),source_config=str(warm/'config.yaml')))
            e.write(warm/'scene.json',dict(source_report=str(warm/'source.json'),vehicle_config='/unchanged'))
            e.write(warm/'trajectory.json',dict(fps=10))
            a=e.v1.asset
            plan=dict(assets=dict(warm_checkpoint=a(warm/'model.pth'),warm_scene=a(warm/'scene.json'),
                source_config=a(warm/'config.yaml'),**{'compiled_trajectory.json':a(warm/'trajectory.json')}))
            e.write(out/'plan.json',dict(fixture=True))
            job=dict(task=dict(renderer_bundle='fixture'))
            image=(np.linspace(.2,.8,256).reshape(16,16,1)*np.ones((1,1,3))*255).astype('uint8')
            target=e.v1.save_image(warm/'target.png',image)
            mask=e.v1.save_image(warm/'mask.png',np.full((16,16),255,dtype='uint8'))
            rows=[]
            for frame,split in [(15,'train'),(45,'train'),(9,'eval'),(29,'eval')]:
                for kind in ['novel','original']:
                    rows.append(dict(id=f'{frame}_{kind}',source_frame=frame,camera_id=0,kind=kind,split=split,
                        accepted=True,target=target,valid_mask=mask,ego_to_world=np.eye(4).tolist(),camera_to_world=np.eye(4).tolist(),
                        K=[[4.,0,8],[0,4,8],[0,0,1]]))
            baseline=ToyRenderer(warm/'scene.json');base_scores=[]
            for row in rows:
                if row['split']=='eval':
                    with torch.no_grad():_,pred,t,m=e.render_row(baseline,row);score=e.metrics(pred,t,m)
                    base_scores.append(dict(kind=row['kind'],score=score))
            avg=e.averages(base_scores,'score')
            e.write(warm/'evaluation.json',dict(averages={k:dict(before=v,after=v) for k,v in avg.items()}))
            plan['assets']['structure/evaluation/report.json']=a(warm/'evaluation.json')
            with patch.object(e,'load',return_value=(out,plan,job,rows)),patch.object(e,'renderer',side_effect=lambda plan,job,scene=None:ToyRenderer(scene or plan['assets']['warm_scene']['path'])),patch.object(e.v1,'make_comparison'):
                e.cache(out);probe=e.probe(out)
                self.assertEqual(probe['admitted_profiles'],['edge_color','footprint'])
                for profile in probe['admitted_profiles']:
                    receipt=e.train(out,profile)
                    self.assertEqual(receipt['steps'],24)
                    self.assertFalse(set(receipt['trained_sample_ids'])&{r['id'] for r in rows if r['split']=='eval'})
                    expected=3 if profile=='edge_color' else 5
                    self.assertEqual(len(receipt['changed_weights']),expected)
                    result=e.evaluate(out,profile)
                    self.assertTrue(result['completed'])
                    self.assertEqual(result['footprint_changed'],profile=='footprint')
                result=e.compare(out)
                self.assertEqual(len(result['profiles']),2)
                self.assertFalse(result['automatic_promotion'])


if __name__=='__main__':unittest.main()
