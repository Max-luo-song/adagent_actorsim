"""CPU regression: editing must move renderer and collision proxy together."""
import types
import unittest
import torch
from native_edit import translate_native

class EditTests(unittest.TestCase):
    def setUp(self):
        self.trans = torch.zeros(6,2,3)
        self.model = types.SimpleNamespace(instances_trans=self.trans)
        self.backend = types.SimpleNamespace(torch=torch, _last=None,
            trainer=types.SimpleNamespace(models={'SMPLNodes':self.model}),
            env=types.SimpleNamespace(models={'SMPLNodes':{'instances_trans':self.trans}}))
        self.request = dict(schema='framework_native_edit_v1',operation='translate',
                            group='SMPLNodes',instance=0,offset_world_m=[.5,0,0])
    def test_shared_proxy_and_trajectory(self):
        original = self.trans.clone()
        translate_native(self.backend,self.request)
        self.assertTrue(torch.equal(self.trans[:,0], original[:,0]+torch.tensor([.5,0,0])))
        self.assertTrue(torch.equal(self.trans[:,1], original[:,1]))
        self.assertIs(self.backend.env.models['SMPLNodes']['instances_trans'],self.model.instances_trans)
    def test_invalid_edit_does_not_mutate(self):
        for field,value in [('instance',True),('offset_world_m',[float('nan'),0,0]),('offset_world_m',[3,0,0])]:
            with self.assertRaises(ValueError): translate_native(self.backend,dict(self.request,**{field:value}))
        self.assertEqual(float(self.trans.abs().sum()),0.)
    def test_detached_proxy_rejected(self):
        self.backend.env.models['SMPLNodes']['instances_trans']=self.trans.clone()
        with self.assertRaises(RuntimeError): translate_native(self.backend,self.request)
    def test_runtime_edit_rejected(self):
        self.backend._last=(0,None)
        with self.assertRaises(RuntimeError): translate_native(self.backend,self.request)

if __name__ == '__main__': unittest.main()
