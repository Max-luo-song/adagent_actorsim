import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple
import unittest
import tempfile

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
import yaml

from scene_math import (pbr_color,sh_eval,deformation,lighting,quat_matrix,
    matrix_quat,skinning_transforms,rigid_geometry,smpl_geometry)
from export_snapshot import camera_metadata,WAYMO_TO_HUGSIM,OPENCV_TO_WAYMO,export

INPUTS=Path(os.environ.get('WAYMO_INPUTS',str(Path(__file__).resolve().parent.parent/'waymo23-inputs')))


def definitions(path,names,namespace):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


class MathTests(unittest.TestCase):
    def test_quaternion_roundtrip_and_covariance(self):
        rng=np.random.default_rng(3)
        r=Rotation.random(100,random_state=rng).as_matrix()
        recovered=quat_matrix(matrix_quat(torch.tensor(r))).numpy()
        np.testing.assert_allclose(recovered,r,atol=1e-10)

    def test_rigid_visibility_and_pose(self):
        state=dict(points_ids=torch.tensor([[0],[1]]),_means=torch.tensor([[1.,0,0],[2.,0,0]]),
            _quats=torch.tensor([[1.,0,0,0]]*2),instances_fv=torch.tensor([[True,False]]),
            instances_quats=torch.tensor([[[1.,0,0,0],[1.,0,0,0]]]),
            instances_trans=torch.tensor([[[3.,0,0],[4.,0,0]]]))
        xyz,q,v=rigid_geometry(state,0)
        torch.testing.assert_close(xyz[:,0],torch.tensor([4.,6.]))
        self.assertEqual(v.tolist(),[True,False])

    def test_skinning_identity_has_no_translation(self):
        parents=torch.tensor([-1,0,1])
        joints=torch.tensor([[[1.,2,3],[1.,3,3],[2.,4,3]]])
        r=torch.eye(3).expand(1,3,3,3)
        a=skinning_transforms(r,joints,parents)
        torch.testing.assert_close(a,torch.eye(4).expand(1,3,4,4))

    def test_skinning_root_rotation_around_joint(self):
        r=torch.tensor(Rotation.from_euler('z',90,degrees=True).as_matrix(),dtype=torch.float64)[None,None]
        j=torch.tensor([[[1.,2,3]]],dtype=torch.float64)
        a=skinning_transforms(r,j,torch.tensor([-1]))
        expected=j[0,0]-r[0,0]@j[0,0]
        torch.testing.assert_close(a[0,0,:3,3],expected)

    def test_smpl_all_invisible(self):
        state=dict(instances_fv=torch.tensor([[False]]),_means=torch.ones(4,3),_quats=torch.tensor([[1.,0,0,0]]*4))
        xyz,q,visible=smpl_geometry(state,0)
        self.assertEqual(xyz.shape,(4,3));self.assertFalse(visible.any())

    def test_coordinate_projection_is_invariant(self):
        camera=np.eye(4);camera[:3,3]=[2,1,.5]
        points=np.array([[4.,2,3],[5,1,2]])
        target_camera=WAYMO_TO_HUGSIM@camera
        target_points=points@WAYMO_TO_HUGSIM[:3,:3].T
        a=(points-camera[:3,3])@camera[:3,:3]
        b=(target_points-target_camera[:3,3])@target_camera[:3,:3]
        np.testing.assert_allclose(a,b)
        self.assertAlmostEqual(np.linalg.det(WAYMO_TO_HUGSIM[:3,:3]),1.)


@unittest.skipUnless((INPUTS/'backup/models/modules.py').exists(),'training source archive required')
class SourceComparisonTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        namespace=dict(torch=torch,nn=nn,F=F,Tensor=Tensor,Optional=Optional,Tuple=Tuple)
        self.ns=definitions(INPUTS/'backup/models/modules.py',
            ['XYZ_Encoder','SinusoidalEncoder','MLP','GlobalLightingModule',
             'get_embedder','Embedder','ConditionalDeformNetwork','VoxelDeformer'],namespace)

    def test_global_lighting_matches_archived_network(self):
        cls=self.ns['GlobalLightingModule']
        model=cls('GlobalLighting',n=18,sh_degree=2,hidden_dim=64,num_layers=4,device='cpu')
        with torch.no_grad():
            for frame in (0,2,5):
                expected=model(torch.tensor([[frame/17]]),torch.tensor([frame*3])).squeeze(0)
                actual=lighting(model.state_dict(),frame,6,frame*3)
                torch.testing.assert_close(actual,expected)

    def test_deformation_matches_archived_network(self):
        cfg=dict(D=8,W=32,embed_dim=4,x_multires=3,t_multires=2,deform_quat=True,deform_scale=False)
        model=self.ns['ConditionalDeformNetwork'](**cfg)
        n=13;ids=torch.arange(n)%2
        means=torch.randn(n,3);sizes=torch.ones(2,3)*2;embedding=torch.randn(2,4)
        state=dict(_means=means,points_ids=ids[:,None],instances_size=sizes,instances_embedding=embedding)
        state.update({'deform_network.'+k:v for k,v in model.state_dict().items()})
        with torch.no_grad():
            actual,dq=deformation(state,cfg,frame=2,frames=6,chunk=5)
            expected,eq,_=model(means/sizes[ids,2,None]*2,torch.full((n,1),.4),embedding[ids])
            torch.testing.assert_close(actual,expected);torch.testing.assert_close(dq,eq)

    def test_pbr_matches_archived_method(self):
        tree=ast.parse((INPUTS/'backup/models/gaussians/vanilla.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VanillaGaussians')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='compute_pbr_color')
        ns=dict(torch=torch,F=F,dataclass_camera=object,
                spherical_harmonics=lambda degree,dirs,c:sh_eval(c,dirs))
        exec(compile(ast.Module(body=[method],type_ignores=[]),'archived_pbr','exec'),ns)
        n=31
        s=dict(_means=torch.randn(n,3),_features_dc=torch.randn(n,3),_normals=torch.randn(n,3),
            _roughness=torch.randn(n,1),_asg_amplitude=torch.randn(n,4,3),_asg_dir=torch.randn(n,4,3))
        obj=SimpleNamespace(**s,num_points=n,num_asg_lobes=4)
        cam=SimpleNamespace(camtoworlds=torch.eye(4));cam.camtoworlds[:3,3]=torch.tensor([3.,2,1])
        for light in (None,torch.randn(9,3)):
            expected=ns['compute_pbr_color'](obj,cam,light)
            actual=pbr_color(s,cam.camtoworlds[:3,3],light)
            torch.testing.assert_close(actual,expected)

    def test_smpl_voxel_sampling_matches_archived_deformer(self):
        cls=self.ns['VoxelDeformer'];obj=cls.__new__(cls);nn.Module.__init__(obj)
        obj.ratio_dim=-1
        obj.register_buffer('ratio',torch.tensor(4.))
        obj.register_buffer('offset',torch.randn(2,1,3))
        obj.register_buffer('scale',torch.ones(2,1,1)*2)
        obj.register_buffer('lbs_voxel_base',torch.randn(2,24,2,3,3))
        obj.voxel_w_correction=nn.Parameter(torch.randn_like(obj.lbs_voxel_base)*.01)
        x=torch.randn(2,5,3)
        expected=obj(x)
        grid=(x-obj.offset)/obj.scale;grid[...,-1]*=obj.ratio
        actual=F.grid_sample(obj.lbs_voxel_base+obj.voxel_w_correction,grid[:,:,None,None],
            align_corners=True,mode='bilinear',padding_mode='border').squeeze(3,4).permute(0,2,1)
        torch.testing.assert_close(actual,expected)

    def test_smpl_full_geometry_matches_archived_method(self):
        tree=ast.parse((INPUTS/'backup/models/nodes/smpl.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SMPLNodes')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='transform_means_and_quats')
        ns=dict(torch=torch,Tensor=Tensor,Tuple=Tuple,quat_mult=lambda a,b:__import__('scene_math').quat_mul(a,b),
            matrix_to_quaternion=lambda r:torch.tensor(Rotation.from_matrix(r.detach().numpy().reshape(-1,3,3)).as_quat()[:,[3,0,1,2]],dtype=r.dtype).reshape(r.shape[:-2]+(4,)))
        exec(compile(ast.Module(body=[method],type_ignores=[]),'archived_smpl','exec'),ns)
        b,n=2,6890
        means=torch.randn(b*n,3)*.1;q=torch.tensor([[1.,0,0,0]]).expand(b*n,4).clone()
        angles=torch.tensor(Rotation.from_euler('z',np.array([[.3],[-.7]])).as_quat()[:,[3,0,1,2]],dtype=torch.float32)
        rootq=torch.tensor([1.,0,0,0]).expand(1,b,1,4).clone();rootq[0,:,0]=angles
        identity=torch.eye(4).expand(b,24,4,4).clone()
        weights=torch.zeros(b,n,24);weights[:,:,0]=1
        state=dict(_means=means,_quats=q,points_ids=torch.arange(b).repeat_interleave(n)[:,None],
            instances_fv=torch.ones(1,b,dtype=torch.bool),instances_quats=rootq,
            instances_trans=torch.randn(1,b,3),smpl_qauts=torch.tensor([1.,0,0,0]).expand(1,b,23,4).clone(),
            **{'template.J_canonical':torch.zeros(b,24,3),'template.A0_inv':identity,
               'template._template_layer.parents':torch.tensor([-1]+[0]*23),'template.W':weights})
        def template(masked_theta,instances_mask,xyz_canonical):
            a=torch.eye(4).expand(b,24,4,4).clone()
            # At zero canonical joints, all child transforms follow the root.
            r=torch.tensor(Rotation.from_quat(masked_theta[:,0].numpy()[:,[1,2,3,0]]).as_matrix(),dtype=torch.float32)
            a[:,:,:3,:3]=r[:,None]
            return weights[instances_mask],a
        obj=SimpleNamespace(num_instances=b,smpl_points_num=n,point_ids=state['points_ids'],
            instances_fv=state['instances_fv'],instances_quats=rootq,instances_trans=state['instances_trans'],
            smpl_qauts=state['smpl_qauts'],cur_frame=0,in_test_set=False,use_voxel_deformer=False,
            quat_act=lambda x:F.normalize(x,dim=-1),template=template,device='cpu')
        expected,eq=ns['transform_means_and_quats'](obj,means,q)
        actual,aq,v=smpl_geometry(state,0)
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(quat_matrix(aq),quat_matrix(eq))

    def test_end_to_end_snapshot_has_all_points_and_native_schema(self):
        def node(points):
            n=len(points)
            return dict(_means=torch.tensor(points,dtype=torch.float32),_scales=torch.zeros(n,3),
                _quats=torch.tensor([[1.,0,0,0]]*n),_opacities=torch.ones(n,1),
                _features_dc=torch.zeros(n,3),_features_rest=torch.zeros(n,3,3))
        bg=node([[0,0,0],[1,0,0]])
        bg.update(_roughness=torch.zeros(2,1),_normals=torch.tensor([[0.,0,1]]*2),
            _asg_amplitude=torch.zeros(2,4,3),_asg_dir=torch.ones(2,4,3))
        rigid=node([[0,0,0]])
        rigid.update(points_ids=torch.zeros(1,1,dtype=torch.long),
            instances_trans=torch.zeros(2,1,3),instances_quats=torch.tensor([[[1.,0,0,0]]]*2),
            instances_fv=torch.ones(2,1,dtype=torch.bool))
        checkpoint=dict(step=40000,models=dict(Background=bg,RoadNodes=node([[0,0,-1]]),RigidNodes=rigid))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);torch.save(checkpoint,path/'source.pth')
            report=export(path/'source.pth',INPUTS,path/'native',refine=False)
            values,_=torch.load(path/'native/scene.pth',weights_only=False)
            self.assertEqual(len(values),11);self.assertEqual(len(values[1]),4)
            self.assertEqual(values[5].shape,(4,3));self.assertEqual(values[-1][5].shape,(0,2))
            self.assertEqual(len(report['nodes']),3)
            report2=export(path/'source.pth',INPUTS,path/'edited',refine=False,remove_rigid=[0])
            self.assertEqual(next(n['visible_points'] for n in report2['nodes'] if n['node']=='RigidNodes'),0)
            with self.assertRaisesRegex(ValueError,'output exists'):
                export(path/'source.pth',INPUTS,path/'native',refine=False)

    def test_real_camera_normalization_and_resolution(self):
        cfg=yaml.safe_load((INPUTS/'config.yaml').read_text())
        c=camera_metadata(INPUTS,cfg,{},0,0,.5,False)
        expected=np.loadtxt(INPUTS/'extrinsics/0.txt')@OPENCV_TO_WAYMO
        np.testing.assert_allclose(c['source_c2w'],expected,atol=1e-10)
        self.assertEqual((c['width'],c['height']),(480,320))
        self.assertAlmostEqual(c['K'][0,0],2056.2823692090583/4)


if __name__=='__main__':unittest.main()
