"""CPU boundaries around real registry/editor/session/managed asset code.

Only the scene loader and raster are doubles; no CPU image is GPU evidence.
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

import copy
pass  # Path is provided by the portable boundary above
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
import agent_bootstrap as bootstrap
from agent_catalog import Catalog
from agent_bridge import AgentBridge
from scene_edit_session import UnifiedSceneSession,saf
from managed_assets import ManagedAssets


class Backend:
    def __init__(self,spec):
        self.spec=spec;self.torch=torch;self._last=None;self.closed=False;self.fail_render=False;self.render_calls=0
        models={}
        for name in ('RigidNodes','SMPLNodes','DeformableNodes'):
            n=2 if name=='RigidNodes' else 1
            q=torch.zeros(90,n,4);q[...,0]=1
            models[name]=SimpleNamespace(instances_trans=torch.zeros(90,n,3),instances_quats=q)
        rigid=models['RigidNodes'];rigid.point_ids=torch.tensor([[0],[0],[1],[1]])
        rigid._features_rest=torch.arange(4*15*3,dtype=torch.float32).reshape(4,15,3)/1000
        self.trainer=SimpleNamespace(models=models,gaussian_classes={'Background':0,'RigidNodes':1},collect_gaussians=lambda *a,**k:None)
        self.env=SimpleNamespace(models={g:{'instances_trans':m.instances_trans,'instances_quats':m.instances_quats} for g,m in models.items()})
        self.supports_managed_assets=True;self.managed_assets=ManagedAssets(self.trainer,'cpu')
    def clear_observation_history(self):self._last=None
    def reset(self):self.managed_assets.clear();self.clear_observation_history()
    def close(self):self.closed=True;self.managed_assets.close()
    def apply(self,edit):
        previous=dict(self.managed_assets.assets)
        try:
            self.managed_assets.apply(edit);self.clear_observation_history()
        except Exception:self.managed_assets.assets=previous;raise
    def render(self,pose,timestamp):
        self.render_calls+=1
        if self.fail_render:raise RuntimeError('explicit fake raster failure')
        self._last=(timestamp,pose)
        moved=sum(float(m.instances_trans[80].sum()) for m in self.trainer.models.values())
        value=50+20*len(self.managed_assets.assets)+int(10*moved)
        c=dict(rgb=np.full((4,6,3),value,dtype=np.uint8),depth=np.ones((4,6),dtype=np.float32),K=np.eye(3),camera_to_world=pose.copy(),source_camera_id=0)
        return dict(cameras={name:dict(c) for name in self.spec.camera_order},ego_state={})


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        inp=self.root/'inputs';(inp/'ego_pose').mkdir(parents=True)
        for frame in (0,80):np.savetxt(inp/'ego_pose'/f'{frame:03d}.txt',np.eye(4))
        asset=self.root/'asset';asset.mkdir()
        xyz=torch.rand(128,3,generator=torch.Generator().manual_seed(42))*torch.tensor([4.,1.6,2.])
        q=torch.zeros(128,4);q[:,0]=1
        params=[3,xyz,torch.zeros(128,1,3),torch.zeros(128,15,3),torch.zeros(128,20),torch.full((128,3),-3.),q,torch.full((128,1),2.),None]
        torch.save((params,0),asset/'gs.pth');(asset/'wlh.json').write_text('[2,4,1.6]')
        from asset_calibration import create
        calibration=asset/'calibration.json';create(asset,calibration)
        cmd=dict(schema='scene_edit_v1',mode='diagnostic_preview',operation='insert',object_id='managed:car',asset=str(asset),asset_calibration=str(calibration),asset_to_world=np.eye(4).tolist())
        (self.root/'asset_command.json').write_text(json.dumps(cmd))
        for scene in ('552','551'):(self.root/f'{scene}.json').write_text('{}')
        data=dict(schema='agent_catalog_v1',scenes={'552':'552.json','551':'551.json'},assets={
            'realcar_white_20240604':dict(command='asset_command.json',placement_id='scene552_ahead',placement_scene_id='552',quality={'status':'cpu_test_fixture'})})
        (self.root/'catalog.json').write_text(json.dumps(data));self.catalog=Catalog(self.root/'catalog.json')
        def spec(path):
            scene=int(Path(path).stem)
            return SimpleNamespace(source=dict(scene=scene,inputs=str(inp),start_timestep=0),fps=10.,frames=90,
                camera_order=('CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT'),summary=lambda:dict(scene=scene))
        self.reader=patch.object(saf.SceneSpec,'read',side_effect=spec);self.reader.start();self.addCleanup(self.reader.stop)
        self.backends=[];self.fail_load=False
        def factory(spec):
            if self.fail_load and spec.source['scene']==551:raise RuntimeError('explicit fake load failure')
            backend=Backend(spec);self.backends.append(backend);return backend
        self.bridge=AgentBridge(self.root/'owner',catalog=self.catalog,session=UnifiedSceneSession(factory),execution_backend='cpu_test_double')
        self.addCleanup(self.bridge.close);self.counter=0
    def req(self,action,args=None,identity=None):
        self.counter+=1
        return dict(schema='agent_request_v1',request_id=identity or f'req{self.counter}',action=action,args=args or {})
    def send(self,action,args=None,identity=None):return self.bridge.handle(self.req(action,args,identity))
    def load(self):
        response=self.send('load_scene',{'scene_id':'552'});self.assertEqual(response['status'],'ok');return response['result']['generation']
    def args(self,command,revision=None,generation=None):
        return dict(command=command,expected_generation=generation or self.bridge.session.generation,
            expected_revision=self.bridge.session.revision if revision is None else revision)
    def observation(self):return dict(source_frame=80,expected_generation=self.bridge.session.generation,expected_revision=self.bridge.session.revision)
    def insert(self):return dict(operation='insert',object_id='managed:car',asset_id='realcar_white_20240604',placement_id='scene552_ahead')
    def translate(self,offset=.5):return dict(operation='translate',object_id='native:SMPLNodes:0',offset_world_m=[offset,0,0])
    def error(self,response,code):self.assertEqual(response['status'],'error');self.assertEqual(response['error']['code'],code)

    def test_queries_do_not_load_scene_or_train(self):
        for action in ('list_assets','list_scenes','get_capabilities'):
            result=self.send(action);self.assertEqual(result['status'],'ok');self.assertFalse(result['training_performed'])
        self.assertFalse(self.backends)
        self.error(self.send('list_objects'),'scene_not_loaded')

    def test_schema_types_and_nonfinite_values_are_rejected_without_mutation(self):
        values=[dict(self.req('get_capabilities'),schema='other'),dict(self.req('get_capabilities'),args=[]),
            dict(self.req('get_capabilities'),request_id='../escape'),self.req('get_capabilities',{'x':float('nan')}),self.req('unknown')]
        for value in values:
            with self.subTest(value=value):self.assertEqual(self.bridge.handle(value)['status'],'error')
        self.assertFalse(self.backends)

    def test_validate_insert_checks_real_calibration_but_allocates_no_object(self):
        self.load();result=self.send('validate_edit',self.args(self.insert()))
        self.assertEqual(result['status'],'ok');self.assertFalse(result['result']['mutation_performed'])
        self.assertFalse(self.bridge.session.objects);self.assertFalse(self.bridge.session.backend.managed_assets.assets)
        self.assertEqual(self.bridge.session.revision,0)
        self.assertTrue(self.catalog.list_assets()[0]['resources_verified'])

    def test_insert_request_replay_has_one_actual_revision_and_asset(self):
        self.load();value=self.req('edit',self.args(self.insert()),'insert_once')
        first=self.bridge.handle(value);second=self.bridge.handle(copy.deepcopy(value))
        self.assertEqual(first['status'],'ok');self.assertFalse(first['replayed']);self.assertTrue(second['replayed'])
        self.assertEqual(self.bridge.session.revision,1);self.assertEqual(len(self.bridge.session.backend.managed_assets.assets),1)
        self.assertEqual(len(self.bridge.session.edit_journal()),1)

    def test_same_id_different_payload_is_rejected_and_original_cache_remains(self):
        self.load();value=self.req('edit',self.args(self.translate()),'move_once');self.bridge.handle(value)
        changed=copy.deepcopy(value);changed['args']['command']['offset_world_m'][0]=1
        self.error(self.bridge.handle(changed),'request_id_conflict')
        self.assertTrue(self.bridge.handle(value)['replayed'])
        self.assertTrue(torch.all(self.bridge.session.backend.trainer.models['SMPLNodes'].instances_trans[:,0,0]==.5))

    def test_generation_revision_guards_and_total_translation_limit(self):
        generation=self.load();self.send('edit',self.args(self.translate(1.5)))
        self.error(self.send('edit',self.args(self.translate(),revision=0)),'stale_revision')
        self.error(self.send('edit',self.args(self.translate(),generation='old')),'stale_generation')
        self.error(self.send('edit',self.args(self.translate(.6))),'edit_limit')
        self.assertEqual(self.bridge.session.revision,1)

    def test_observe_real_packet_files_then_runtime_edit_rejected_and_reset_exact(self):
        generation=self.load();self.send('edit',self.args(self.insert()));self.send('edit',self.args(self.translate()))
        response=self.send('observe',self.observation());self.assertEqual(response['status'],'ok')
        self.assertEqual(len(response['result']['cameras']),3)
        for camera in response['result']['cameras'].values():self.assertTrue(Path(camera['rgb']).is_file());self.assertTrue(Path(camera['depth']).is_file())
        self.error(self.send('edit',self.args(self.translate())),'edit_requires_pre_run')
        reset=self.send('reset',dict(expected_generation=generation,expected_revision=2))
        self.assertEqual(reset['status'],'ok');self.assertTrue(all(reset['result']['reset_audit'].values()))
        self.assertNotEqual(generation,reset['result']['generation']);self.assertFalse(self.bridge.session.objects)
        self.error(self.send('edit',self.args(self.translate(),generation=generation)),'stale_generation')
        plain=self.send('observe',self.observation());self.assertFalse(plain['result']['edit_trust']['edited'])

    def test_pre_run_preview_allows_iterative_edits_without_deleting_managed_assets(self):
        self.load();before=self.send('preview',self.observation())
        self.assertEqual(before['status'],'ok');self.assertTrue(before['result']['preview_only'])
        self.assertFalse(self.bridge.session.editor.started);self.assertIsNone(self.bridge.session.backend._last)
        self.assertEqual(self.send('edit',self.args(self.insert()))['status'],'ok')
        after=self.send('preview',self.observation());self.assertEqual(after['status'],'ok')
        self.assertEqual(len(self.bridge.session.backend.managed_assets.assets),1)
        self.assertEqual(self.send('edit',self.args(self.translate()))['status'],'ok')
        self.assertEqual(self.send('observe',self.observation())['status'],'ok')
        self.error(self.send('preview',self.observation()),'preview_requires_pre_run')
        self.error(self.send('edit',self.args(self.translate())),'edit_requires_pre_run')

    def test_failed_preview_history_clear_invalidates_editor_instead_of_committing(self):
        self.load()
        def failed():raise RuntimeError('explicit preview history clear failure')
        self.bridge.session.backend.clear_observation_history=failed
        response=self.send('preview',self.observation())
        self.error(response,'backend_error');self.assertEqual(response['context']['phase'],'failed')
        self.assertEqual(self.bridge.session.revision,0)

    def test_observe_rejects_stale_scene_before_rendering_or_advancing_phase(self):
        self.load();args=self.observation();args['expected_generation']='old'
        self.error(self.send('observe',args),'stale_generation')
        self.assertEqual(self.backends[-1].render_calls,0)
        self.assertFalse(self.bridge.session.editor.started)
        self.error(self.send('observe',{'source_frame':80}),'invalid_request')

    def test_native_rotation_through_json_restores_quaternion_and_sh(self):
        self.load();model=self.bridge.session.backend.trainer.models['RigidNodes'];q=model.instances_quats.clone();sh=model._features_rest.clone()
        rotate=dict(operation='rotate',object_id='native:RigidNodes:1',yaw_degrees=15.)
        response=self.send('edit',self.args(rotate));self.assertEqual(response['status'],'ok')
        self.assertFalse(torch.equal(model.instances_quats,q));self.assertFalse(torch.equal(model._features_rest,sh))
        reset=self.send('reset',dict(expected_generation=self.bridge.session.generation,expected_revision=1))
        self.assertTrue(reset['result']['reset_audit']['native_radiance_restored'])
        self.assertTrue(torch.equal(model.instances_quats,q));self.assertTrue(torch.equal(model._features_rest,sh))

    def test_managed_set_pose_and_remove_update_same_registered_object(self):
        self.load();self.send('edit',self.args(self.insert()))
        pose=np.eye(4);pose[0,3]=.5
        result=self.send('edit',self.args(dict(operation='set_pose',object_id='managed:car',asset_to_world=pose.tolist())))
        self.assertEqual(result['status'],'ok');self.assertEqual(self.bridge.session.objects['car']['asset_to_world'][0][3],.5)
        result=self.send('edit',self.args(dict(operation='remove',object_id='managed:car')))
        self.assertEqual(result['status'],'ok');self.assertFalse(self.bridge.bindings)
        self.error(self.send('edit',self.args(dict(operation='remove',object_id='managed:car'))),'object_not_found')

    def test_unknown_asset_placement_and_raw_path_fields_are_rejected(self):
        self.load()
        self.error(self.send('edit',self.args(dict(self.insert(),asset_id='unregistered'))),'unknown_asset')
        self.error(self.send('edit',self.args(dict(self.insert(),placement_id='elsewhere'))),'unknown_placement')
        self.error(self.send('edit',self.args(dict(self.insert(),asset='/arbitrary/file'))),'invalid_request')
        self.assertEqual(self.bridge.session.revision,0)

    def test_catalog_placement_cannot_cross_scene_and_load_closes_old_backend(self):
        generation=self.load();old=self.backends[-1]
        response=self.send('load_scene',dict(scene_id='551',expected_generation=generation,expected_revision=0))
        self.assertEqual(response['status'],'ok');self.assertTrue(old.closed)
        self.error(self.send('validate_edit',self.args(self.insert())),'placement_scene_mismatch')
        self.assertEqual(self.bridge.session.revision,0)

    def test_changed_asset_is_rejected_by_calibration_before_loading(self):
        self.load();self.catalog.verify_asset('realcar_white_20240604')
        (self.root/'asset/wlh.json').write_text('[1,2,3]')
        self.error(self.send('edit',self.args(self.insert())),'asset_resource_error')
        self.assertFalse(self.bridge.session.objects);self.assertEqual(self.bridge.session.revision,0)

    def test_failed_raster_is_reported_and_not_implicitly_retried(self):
        self.load();self.backends[-1].fail_render=True;req=self.req('observe',self.observation(),'failed_observe')
        first=self.bridge.handle(req);second=self.bridge.handle(req)
        self.error(first,'backend_error');self.assertTrue(second['replayed']);self.assertEqual(self.backends[-1].render_calls,1)
        self.assertTrue((self.bridge.output/'errors/failed_observe.txt').is_file())
        self.assertEqual(first['context']['phase'],'observing')

    def test_failed_scene_load_can_recover_without_a_stale_owner(self):
        generation=self.load();self.fail_load=True
        self.error(self.send('load_scene',dict(scene_id='551',expected_generation=generation,expected_revision=0)),'backend_error')
        self.assertIsNone(self.bridge.context()['scene_id'])
        self.assertEqual(self.send('load_scene',{'scene_id':'552'})['status'],'ok')

    def test_old_successful_request_replay_after_reset_does_not_reapply(self):
        generation=self.load();value=self.req('edit',self.args(self.translate()),'historical_move')
        self.bridge.handle(value)
        self.send('reset',dict(expected_generation=generation,expected_revision=1))
        response=self.bridge.handle(value);self.assertTrue(response['replayed'])
        self.assertEqual(response['context']['generation'],self.bridge.session.generation)
        self.assertEqual(response['result']['generation'],generation)
        self.assertEqual(self.bridge.session.revision,0)
        self.assertTrue(torch.all(self.bridge.session.backend.trainer.models['SMPLNodes'].instances_trans==0))

    def test_close_prevents_further_requests_and_is_replayable(self):
        self.load();value=self.req('close',identity='close_once')
        self.assertEqual(self.bridge.handle(value)['status'],'ok');self.assertTrue(self.bridge.handle(value)['replayed'])
        self.error(self.send('get_capabilities'),'bridge_closed')

    def test_invalid_pose_missing_version_and_unsupported_native_operation(self):
        self.load()
        self.error(self.send('edit',dict(command=self.translate(),expected_revision=0)),'invalid_request')
        self.error(self.send('edit',self.args(dict(self.insert(),placement_id=None,asset_to_world=np.eye(4).tolist()))),'invalid_edit')
        insert=self.insert();insert.pop('placement_id');insert['asset_to_world']=[[1]]
        self.error(self.send('edit',self.args(insert)),'invalid_pose')
        self.error(self.send('edit',self.args(dict(operation='rotate',object_id='native:SMPLNodes:0',yaw_degrees=5))),'unsupported_operation')
        self.error(self.send('edit',self.args(dict(operation='delete',object_id='native:RigidNodes:1'))),'unsupported_operation')
        self.assertEqual(self.bridge.session.revision,0)

if __name__=='__main__':unittest.main()
