
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

import io,json,subprocess,sys,unittest
pass  # Path is provided by the portable boundary above
from scene_agent_client import SceneAgentClient,JsonlTransport,tool_specs

class ClientTests(unittest.TestCase):
    def test_jsonl_transport_writes_exactly_one_request_and_reads_response(self):
        request=dict(schema='agent_request_v1',request_id='x',action='get_capabilities',args={})
        response=dict(schema='agent_response_v1',request_id='x',status='ok',context={})
        writer=io.StringIO();transport=JsonlTransport(io.StringIO(json.dumps(response)+'\n'),writer)
        self.assertEqual(transport(request),response);self.assertEqual(_hugsim_expand(json.loads(writer.getvalue())),request)
    def test_disconnect_and_invalid_response_do_not_trigger_implicit_retry(self):
        transport=JsonlTransport(io.StringIO(''),io.StringIO())
        with self.assertRaises(ConnectionError):transport({})
        client=SceneAgentClient(lambda value:{'schema':'incorrect'})
        with self.assertRaises(ValueError):client.call('get_capabilities',request_id='check')
    def test_transport_retry_uses_original_envelope_after_response_loss(self):
        calls=[]
        def transport(req):
            calls.append(req)
            if len(calls)==1:raise ConnectionError('response lost')
            return dict(schema='agent_response_v1',request_id=req['request_id'],context={},status='ok')
        c=SceneAgentClient(transport)
        with self.assertRaises(ConnectionError):c.call('get_capabilities',request_id='once')
        c.call('get_capabilities',request_id='once');self.assertEqual(calls[0],calls[1])
    def test_bad_args_rejected_without_transport(self):
        c=SceneAgentClient(lambda _:self.fail('should not contact service'))
        for args in ([],{'x':float('nan')}):
            with self.assertRaises(ValueError):c.call('get_capabilities',args)
        with self.assertRaises(ValueError):c.call('get_capabilities',request_id='../escape')
    def test_exported_contract_files_match_live_agent_tools(self):
        from task_contracts import TASK_SCHEMA,RESULT_SCHEMA,_EDIT_FIELDS
        from edit_api import FIELDS
        self.assertEqual(set(_EDIT_FIELDS),set(FIELDS))
        folder=Path(__file__).parent/'schemas'
        self.assertEqual(_hugsim_expand(json.loads((folder/'agent_tools.json').read_text())),tool_specs())
        self.assertEqual(_hugsim_expand(json.loads((folder/'scene_task_v1.schema.json').read_text())),TASK_SCHEMA)
        self.assertEqual(_hugsim_expand(json.loads((folder/'scene_task_result_v1.schema.json').read_text())),RESULT_SCHEMA)
    def test_external_client_imports_without_backend_or_torch(self):
        code='import sys; import scene_agent_client; assert "torch" not in sys.modules; assert "agent_bridge" not in sys.modules; print(len(scene_agent_client.tool_specs()))'
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=Path(__file__).parent,capture_output=True,text=True,check=True)
        self.assertEqual(result.stdout.strip(),'2')

if __name__=='__main__':unittest.main()
