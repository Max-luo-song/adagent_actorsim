
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

import json,os,tempfile,threading,unittest
pass  # Path is provided by the portable boundary above
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from unittest.mock import patch
from helpers import config
from core import AgentError,Deferred,read
from provider import HTTPProvider,TransientAPI
class ProviderTests(unittest.TestCase):
 def call(self,payload=None,status=200):
  seen=[]
  class Handler(BaseHTTPRequestHandler):
   def log_message(self,*a):pass
   def do_POST(self):
    seen.append((self.path,_hugsim_expand(json.loads(self.rfile.read(int(self.headers['Content-Length'])))),self.headers.get('Authorization')))
    self.send_response(status);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(json.dumps(payload).encode())
  server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  return server,thread,seen
 def envelope(self,content='{"decision":"fixture"}',finish='stop'):return dict(id='fixture',choices=[dict(finish_reason=finish,message=dict(content=content))],usage=dict(total_tokens=12))
 def test_http_json_protocol_and_budget_callback(self):
  server,thread,seen=self.call(self.envelope());c=config()['api'];charged=[]
  try:
   with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'PATCH_API_BASE_URL':'http://127.0.0.1:'+str(server.server_port)+'/v1','PATCH_MODEL':'fixture-model','PATCH_API_KEY':'fixture-secret'}):
    value,meta=HTTPProvider(c).complete([dict(role='system',content='Return JSON')],{},charged.append,Path(d)/'call.json')
    self.assertEqual(value,{'decision':'fixture'});self.assertEqual(meta['usage_tokens'],12);self.assertEqual(seen[0][0],'/v1/chat/completions');self.assertEqual(seen[0][1]['model'],'fixture-model');self.assertEqual(seen[0][1]['response_format'],{'type':'json_object'});self.assertEqual(len(charged),1);self.assertNotIn('fixture-secret',(Path(d)/'call.json').read_text())
  finally:server.shutdown();server.server_close();thread.join()
 def test_transient_and_auth_failures_have_distinct_outcomes(self):
  for status,error in [(429,TransientAPI),(503,TransientAPI),(401,Deferred)]:
   server,thread,_=self.call({'error':'fixture-secret'},status)
   try:
    with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'PATCH_API_BASE_URL':'http://127.0.0.1:'+str(server.server_port),'PATCH_MODEL':'m','PATCH_API_KEY':'fixture-secret'}):
     p=Path(d)/'call.json'
     with self.assertRaises(error):HTTPProvider(config()['api']).complete([],{},lambda n:None,p)
     self.assertNotIn('fixture-secret',p.read_text());self.assertEqual(read(p)['http_status'],status)
   finally:server.shutdown();server.server_close();thread.join()
 def test_truncated_response_never_becomes_a_patch(self):
  server,thread,_=self.call(self.envelope(finish='length'))
  try:
   with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'PATCH_API_BASE_URL':'http://127.0.0.1:'+str(server.server_port),'PATCH_MODEL':'m','PATCH_API_KEY':'fixture-secret'}):
    with self.assertRaises(AgentError):HTTPProvider(config()['api']).complete([],{},lambda n:None,Path(d)/'call.json')
  finally:server.shutdown();server.server_close();thread.join()
 def test_missing_configuration_never_sends_request(self):
  c=config()['api']
  with patch.dict(os.environ,{'PATCH_API_BASE_URL':'','PATCH_MODEL':'','PATCH_API_KEY':''}):
   self.assertEqual(len(HTTPProvider(c).readiness()),3)
   with self.assertRaises(Deferred):HTTPProvider(c).complete([],{},lambda n:self.fail('Must not reserve'),Path('/not_written'))
if __name__=='__main__':unittest.main()
