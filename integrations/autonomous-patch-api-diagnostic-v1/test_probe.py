
# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

import io,json,os,ssl,socket,unittest,urllib.error
from unittest.mock import patch,MagicMock
import probe
class ProbeTests(unittest.TestCase):
 def test_missing_config_sends_no_request(self):
  with patch.dict(os.environ,{},clear=True),patch('urllib.request.build_opener') as opener:
   self.assertFalse(probe.run()['request_sent']);opener.assert_not_called()
 def test_transport_classifications(self):
  for err,expected in [(socket.gaierror(-2,'missing'),'dns_failure'),(TimeoutError('late'),'timeout'),(ssl.SSLCertVerificationError('bad'),'tls_certificate_failure'),(ConnectionRefusedError('refused'),'connection_refused')]:self.assertEqual(probe.classify(urllib.error.URLError(err)),expected)
 def request(self,opener):
  with patch.dict(os.environ,{'PATCH_API_KEY':'SECRET','PATCH_API_BASE_URL':'https://example.invalid/v1','PATCH_MODEL':'test'}),patch('urllib.request.build_opener',return_value=opener):return probe.run()
 def test_secret_redacted_from_transport(self):
  opener=MagicMock();opener.open.side_effect=urllib.error.URLError('SECRET refused');r=self.request(opener);self.assertNotIn('SECRET',json.dumps(r));self.assertEqual(r['status'],'transport_failure')
 def test_http_failure_has_bounded_provider_reason(self):
  opener=MagicMock();opener.open.side_effect=urllib.error.HTTPError('https://example.invalid',400,'bad',{},io.BytesIO(b'{"error":{"message":"SECRET wrong token parameter"}}'));r=self.request(opener);self.assertEqual(r['http_status'],400);self.assertNotIn('SECRET',json.dumps(r))
 def test_success_is_one_small_request_not_full_verification(self):
  opener=MagicMock();response=opener.open.return_value.__enter__.return_value;response.status=200;response.read.return_value=b'{"choices":[{}],"usage":{"total_tokens":8}}';r=self.request(opener);self.assertEqual(r['status'],'minimal_request_succeeded');self.assertFalse(r['full_patch_request_verified']);self.assertEqual(opener.open.call_count,1);data=_hugsim_expand(json.loads(opener.open.call_args.args[0].data));self.assertEqual(data['max_completion_tokens'],16)
if __name__=='__main__':unittest.main()
