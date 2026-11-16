"""Meaningful CPU contract tests; these do not replace CUDA render acceptance."""
import copy
import unittest
import numpy as np
import torch
import detail_policy as p


class DetailTests(unittest.TestCase):
    def test_support_distinct_times_and_occlusion(self):
        ref=dict(depth=np.full((7,7),5.),mask=np.ones((7,7),dtype=bool))
        points=np.array([[0,0,5],[0,0,7],[100,0,5],[0,0,-5]])
        k=np.array([[4.,0,3],[0,4,3],[0,0,1]])
        visible=p.visible_centers(points,np.eye(4),k,ref)
        np.testing.assert_array_equal(visible,[True,False,False,False])
        a=p.SupportAccumulator(4)
        for _ in range(3):a.add(0,visible)
        np.testing.assert_array_equal(a.finish(),[False]*4)
        a=p.SupportAccumulator(4);a.add(0,visible);a.add(3,visible)
        np.testing.assert_array_equal(a.finish(),[True,False,False,False])

    def test_bounds_gradients_and_unsupported_points(self):
        keys=p.COLOR_FIELDS|p.FOOT_FIELDS
        selected={key:torch.nn.Parameter(torch.zeros(3,3 if key[1]!='_opacities' else 1)) for key in keys}
        initial={k:v.detach().clone() for k,v in selected.items()};support=torch.tensor([True,False,True])
        for v in selected.values():v.grad=torch.ones_like(v)
        p.restrict_gradients(selected,support)
        for key in p.FOOT_FIELDS:self.assertEqual(float(selected[key].grad[1].sum()),0.)
        with torch.no_grad():
            for v in selected.values():v.add_(1.)
        p.clamp_updates(selected,initial,support)
        scales=selected['Background','_scales']
        self.assertAlmostEqual(float(scales[0,0].exp()),1.05,places=6)
        self.assertEqual(float(scales[1].abs().sum()),0.)
        with torch.no_grad():scales[0].fill_(-5)
        p.clamp_updates(selected,initial,support)
        self.assertAlmostEqual(float(scales[0,0].exp()),.85,places=6)

    def test_exact_checkpoint_scope_and_trust_region(self):
        before=dict(models=dict(Background=dict(_scales=torch.zeros(2,3),_means=torch.zeros(2,3)),
            RoadNodes=dict(_scales=torch.zeros(2,3))),step=17)
        after=copy.deepcopy(before);after['models']['Background']['_scales'][0]=-.1
        allowed={('Background','_scales')};support=torch.tensor([True,False])
        self.assertEqual(p.validate_checkpoint(before,after,allowed,support),['models.Background._scales'])
        for model,field,index,value in [('Background','_scales',1,-.1),('Background','_means',0,.01),('RoadNodes','_scales',0,.01),('Background','_scales',0,-.3)]:
            invalid=copy.deepcopy(after);invalid['models'][model][field][index]=value
            with self.assertRaises(ValueError):p.validate_checkpoint(before,invalid,allowed,support)

    def test_depth_reference_detects_displacement_and_holes(self):
        depth=torch.full((12,12,1),10.,requires_grad=True);alpha=torch.ones_like(depth,requires_grad=True)
        ref=dict(depth=depth.detach()[...,0],opacity=alpha.detach()[...,0],mask=torch.ones(12,12,dtype=torch.bool))
        d,a=p.depth_terms(depth,alpha,ref);self.assertEqual(float(d+a),0.)
        metrics=p.depth_metrics(depth+1,alpha*.4,ref)
        self.assertAlmostEqual(metrics['mean_relative_depth'],.1,places=6)
        self.assertEqual(metrics['new_hole_fraction'],1.)
        loss=sum(p.depth_terms(depth+1,alpha*.9,ref));loss.backward()
        self.assertGreater(float(depth.grad.abs().sum()),0.)
        self.assertGreater(float(alpha.grad.abs().sum()),0.)

    def test_profile_matching(self):
        for key in ['epochs','dc_lr','rest_lr','l1_weight','ssim_weight','edge_weight']:
            self.assertEqual(p.PROFILES['edge_color'][key],p.PROFILES['footprint'][key])
        self.assertEqual(p.PROFILES['edge_color']['edge_weight']/.1,8.)


if __name__=='__main__':unittest.main()
