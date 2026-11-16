
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

import io,json,os,tempfile,unittest,urllib.error
pass  # Path is provided by the portable boundary above
from unittest.mock import patch,MagicMock
from helpers import config,FakeAdapter,reply
from core import read,write
from provider import HTTPProvider,CompletedResponse,TransientAPI,FixtureProvider
from engine import Engine
class StreamingTests(unittest.TestCase):
 def call(self,rows,finish=True):
  response=MagicMock();response.headers={'Content-Type':'text/event-stream'}
  data=b''.join(b'data: '+json.dumps(r).encode()+b'\n\n' for r in rows)+(b'data: [DONE]\n\n' if finish else b'')
  response.readline.side_effect=io.BytesIO(data).readline
  opener=MagicMock();opener.open.return_value.__enter__.return_value=response
  return opener
 def usage(self,n=63243):return dict(prompt_tokens=55051,completion_tokens=n-55051,total_tokens=n)
 def env(self):return patch.dict(os.environ,{'PATCH_API_BASE_URL':'https://example.invalid/v1','PATCH_MODEL':'fixture','PATCH_API_KEY':'SECRET'})
 def test_length_after_reasoning_reconciles_actual_receipt_without_saving_reasoning(self):
  rows=[{'choices':[{'delta':{'reasoning_content':'private reasoning'},'finish_reason':None}]},{'choices':[{'delta':{},'finish_reason':'length'}],'usage':self.usage()}]
  with tempfile.TemporaryDirectory() as d,self.env(),patch('urllib.request.build_opener',return_value=self.call(rows)):
   path=Path(d)/'call.json'
   with self.assertRaises(CompletedResponse) as ctx:HTTPProvider(config()['api']).complete([],{},lambda n:None,path)
   self.assertEqual(ctx.exception.metadata['usage_tokens'],63243);self.assertEqual(ctx.exception.metadata['content_characters'],0);self.assertGreater(ctx.exception.metadata['reasoning_characters'],0);self.assertNotIn('private reasoning',path.read_text());self.assertNotIn('SECRET',path.read_text());self.assertTrue(read(path.with_suffix('.progress.json'))['completed'])
 def test_stream_chunks_form_final_json_and_usage_is_retained(self):
  rows=[{'choices':[{'delta':{'content':'{"decision":'},'finish_reason':None}]},{'choices':[{'delta':{'content':'"diagnose"}'},'finish_reason':'stop'}]},{'choices':[],'usage':self.usage()}]
  with tempfile.TemporaryDirectory() as d,self.env(),patch('urllib.request.build_opener',return_value=self.call(rows)):
   value,meta=HTTPProvider(config()['api']).complete([],{},lambda n:None,Path(d)/'call.json');self.assertEqual(value,{'decision':'diagnose'});self.assertTrue(meta['usage_known']);self.assertEqual(meta['usage']['prompt_tokens'],55051)
 def test_eof_without_finished_json_is_never_accepted(self):
  rows=[{'choices':[{'delta':{'content':'{"partial":'},'finish_reason':None}]}]
  with tempfile.TemporaryDirectory() as d,self.env(),patch('urllib.request.build_opener',return_value=self.call(rows,False)):
   with self.assertRaises(CompletedResponse):HTTPProvider(config()['api']).complete([],{},lambda n:None,Path(d)/'call.json')
 def test_k3_uses_registered_low_reasoning_and_other_models_get_no_parameter(self):
  rows=[{'choices':[{'delta':{'content':'{}'},'finish_reason':'stop'}],'usage':self.usage()}]
  for model in ('kimi-k3','unregistered-model'):
   opener=self.call(rows)
   with tempfile.TemporaryDirectory() as d,self.env(),patch.dict(os.environ,{'PATCH_MODEL':model}),patch('urllib.request.build_opener',return_value=opener):
    value,meta=HTTPProvider(config()['api']).complete([],{},lambda n:None,Path(d)/'call.json');payload=_hugsim_expand(json.loads(opener.open.call_args.args[0].data))
    if model=='kimi-k3':self.assertEqual(payload['reasoning_effort'],'low');self.assertEqual(meta['requested_reasoning_effort'],'low')
    else:self.assertNotIn('reasoning_effort',payload);self.assertIsNone(meta['requested_reasoning_effort'])
 def test_transport_reason_is_sanitized(self):
  opener=MagicMock();opener.open.side_effect=urllib.error.URLError('SECRET DNS failure')
  with tempfile.TemporaryDirectory() as d,self.env(),patch('urllib.request.build_opener',return_value=opener):
   path=Path(d)/'call.json'
   with self.assertRaises(TransientAPI) as ctx:HTTPProvider(config()['api']).complete([],{},lambda n:None,path)
   self.assertNotIn('SECRET',str(ctx.exception));self.assertNotIn('SECRET',path.read_text())
 def test_incomplete_completion_blocks_after_one_call_and_refunds_reservation_to_actual_usage(self):
  class Limited(FixtureProvider):
   def complete(self,messages,schema,before,path):
    self.calls+=1;before(10000);raise CompletedResponse('length',dict(usage_tokens=123,reserved_tokens=10000,usage_known=True,usage={'prompt_tokens':100,'completion_tokens':23}))
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);provider=Limited(reply);e=Engine(root/'campaign',config(),provider,FakeAdapter(root));e.initialize('limited');s=e.run();self.assertEqual(provider.calls,1);self.assertEqual(s['status'],'blocked_api_response');self.assertEqual(s['usage']['charged_tokens'],123);self.assertEqual(s['usage']['reported_tokens'],123);self.assertEqual(s['usage']['completion_tokens'],23);self.assertEqual(s['usage']['gpu_dispatches'],0)
 def test_two_transport_failures_stop_instead_of_consuming_whole_budget(self):
  class Lost(FixtureProvider):
   def complete(self,messages,schema,before,path):self.calls+=1;before(100);raise TransientAPI('fixture timeout')
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);provider=Lost(reply);e=Engine(root/'campaign',config(),provider,FakeAdapter(root));e.initialize('lost');s=e.run();self.assertEqual(provider.calls,2);self.assertEqual(s['status'],'blocked_api_transport');self.assertEqual(s['usage']['charged_tokens'],200);self.assertEqual(s['usage']['reported_tokens'],0)
if __name__=='__main__':unittest.main()
