"""Exercise real output methods with the installed CPU box constructor; no CUDA imports."""
import ast,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
BASE=Path(__file__).resolve().parents[2]
def load_methods():
 folder=BASE/'code/mmdetection3d-uniad-legacy/mmdet3d/core/bbox/structures'
 tree=ast.parse((folder/'base_box3d.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BaseInstance3DBoxes');init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__');scope=dict(torch=torch,np=np);exec(compile(ast.fix_missing_locations(ast.Module(body=[init],type_ignores=[])),'installed_box_constructor','exec'),scope)
 class Box:
  __init__=scope['__init__']
  def to(self,*args,**kwargs):return self
 path=BASE/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad/detectors/uniad_track.py';tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='UniADTrack');functions=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_track_instances2results','_det_instances2results')];scope=dict(LiDARInstance3DBoxes=Box);exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),str(path),'exec'),scope);return scope
class OriginTests(unittest.TestCase):
 def test_tracking_export_keeps_raw_gravity_and_origin_version(self):
  path=BASE/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad/detectors/uniad_track.py'
  tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='UniADTrack')
  method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='simple_test_track')
  assignment=next(n for n in method.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='get_keys' for t in n.targets))
  keys=ast.literal_eval(assignment.value)
  self.assertIn('boxes_3d_gravity',keys)
  self.assertIn('decoded_box_origin',keys)
 def test_retained_and_candidate_decoders_convert_gravity_once(self):
  methods=load_methods()
  for method in ('_track_instances2results','_det_instances2results'):
   data=torch.tensor([[3.,8.,-.4,1.8,4.5,1.6,.37,.2,-.1],[-2.,4.,1.,2.,3.,2.5,-.7,0.,0.]])
   original=data.clone();scores=torch.tensor([.8,.6]);labels=torch.tensor([0,1]);decoded=dict(bboxes=data,labels=labels,scores=scores,bbox_index=torch.arange(2),track_scores=scores,obj_idxes=torch.arange(2),mask=torch.ones(2,dtype=torch.bool))
   core=SimpleNamespace(bbox_coder=SimpleNamespace(decode=lambda *args,**kwargs:[decoded]))
   class Instances:
    pred_logits=torch.ones((2,10));pred_boxes=torch.ones((2,10));scores=torch.ones(2);obj_idxes=torch.arange(2)
    def __getitem__(self,key):return self
   if method=='_track_instances2results':r=methods[method](core,Instances(),[]);key='boxes_3d'
   else:r=methods[method](core,Instances(),[{}],[])[0];key='boxes_3d_det'
   expected=original.clone();expected[:,2]-=original[:,5]/2
   torch.testing.assert_close(r[key].tensor,expected)
   torch.testing.assert_close(data,original)
   self.assertEqual(r['decoded_box_origin'],'gravity_to_bottom_v1')
   torch.testing.assert_close(r['boxes_3d_gravity' if method=='_track_instances2results' else 'boxes_3d_det_gravity'],original)
   torch.testing.assert_close(r['scores_3d' if method=='_track_instances2results' else 'scores_3d_det'],scores)
if __name__=='__main__':unittest.main()
