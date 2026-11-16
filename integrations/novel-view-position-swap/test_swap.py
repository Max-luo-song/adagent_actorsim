import unittest,types
import numpy as np
from geometry import design

class SwapGeometry(unittest.TestCase):
    def source(self):
        poses=np.repeat(np.eye(4)[None],61,axis=0);poses[:,0,3]=np.arange(61)*2
        samples=[dict(source_frame=i,recorded_ego_to_world=p.tolist()) for i,p in enumerate(poses)]
        trans=np.zeros((61,4,3));trans[:,3]=np.c_[poses[:,0,3]-5+np.arange(61)*.2,np.full(61,4.4),np.full(61,.76)]
        quat=np.zeros((61,4,4));quat[:,:,0]=1
        sizes=np.tile([5.,2.,1.52],(4,1))
        return poses,samples,dict(RigidNodes__instances_trans=trans,RigidNodes__instances_quats=quat,RigidNodes__instances_size=sizes)
    def test_swap_ground_height_and_trajectories(self):
        old,samples,actors=self.source();ego,car,audit=design(samples,actors,[0,0,1])
        np.testing.assert_allclose(ego[:,1,3],4.4);np.testing.assert_allclose(ego[:,2,3],0)
        np.testing.assert_allclose(car[:,:2],old[:,:2,3]);np.testing.assert_allclose(car[:,2],.76)
        self.assertTrue(all(audit['checks'].values()));self.assertGreater(audit['minimum_ego_target_footprint_gap_m'],2.)
    def test_intersection_is_rejected(self):
        old,samples,actors=self.source();actors['RigidNodes__instances_trans'][:,3,0]=old[:,0,3];actors['RigidNodes__instances_trans'][:,3,1]=0
        with self.assertRaisesRegex(ValueError,'Swap geometry check failed'):design(samples,actors,[0,0,1])
    def test_heldout_translation_is_exact_and_always_restored(self):
        import torch
        from preview import TranslationSwap
        t=torch.arange(5*4*3,dtype=torch.float32).reshape(5,4,3)
        model=types.SimpleNamespace(instances_trans=t);original=t.clone()
        backend=types.SimpleNamespace(torch=torch,trainer=types.SimpleNamespace(models={'RigidNodes':model}),
            env=types.SimpleNamespace(models={'RigidNodes':{'instances_trans':t}}),dataset=types.SimpleNamespace(test_timesteps=[2]))
        edited=t[:,3].numpy().copy();edited[2]=[500,20,.76]
        owner=TranslationSwap(backend,dict(original_car_trans=t[:,3].numpy().copy(),car_trans=edited))
        with self.assertRaisesRegex(RuntimeError,'render fault'):
            with owner.at(2,True):
                np.testing.assert_allclose(t[[1,2,3],3].numpy(),np.repeat(edited[2][None],3,axis=0));raise RuntimeError('render fault')
        self.assertTrue(torch.equal(t,original))

if __name__=='__main__':unittest.main()
