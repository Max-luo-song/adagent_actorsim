"""One bounded minimal request through the same transport as the frozen agent."""

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

import json,os,ssl,socket,time,urllib.request,urllib.error,urllib.parse,sys
pass  # Path is provided by the portable boundary above
ROOT=Path(__file__).resolve().parents[2]
APP=ROOT/'integrations/autonomous-patch-agent-v2'

def classify(error):
 reason=getattr(error,'reason',error)
 if isinstance(reason,(TimeoutError,socket.timeout)):return 'timeout'
 if isinstance(reason,socket.gaierror):return 'dns_failure'
 if isinstance(reason,ssl.SSLCertVerificationError):return 'tls_certificate_failure'
 if isinstance(reason,ssl.SSLError):return 'tls_failure'
 if isinstance(reason,ConnectionRefusedError):return 'connection_refused'
 if isinstance(reason,ConnectionResetError):return 'connection_reset'
 return 'transport_failure'

def run():
 c=_hugsim_expand(json.loads((APP/'config.example.json').read_text()))['api']
 missing=[c[k] for k in ('base_url_env','model_env','api_key_env') if not os.environ.get(c[k])]
 if missing:return {'status':'configuration_missing','missing':missing,'request_sent':False}
 key=os.environ[c['api_key_env']];url=os.environ[c['base_url_env']].rstrip('/')
 parsed=urllib.parse.urlparse(url)
 if parsed.username or parsed.password or parsed.query or parsed.fragment:return {'status':'invalid_base_url','request_sent':False}
 if parsed.scheme!='https' and not (parsed.scheme=='http' and parsed.hostname in ('localhost','127.0.0.1','::1')):return {'status':'invalid_url_scheme','request_sent':False}
 if not url.endswith('/chat/completions'):url+='/chat/completions'
 proxies=urllib.request.getproxies()
 result={'status':'pending','host':parsed.hostname,'proxy_configured':bool(proxies.get(parsed.scheme)),'proxy_bypass':urllib.request.proxy_bypass(parsed.hostname),'max_output_tokens':16,'timeout_seconds':45,'request_sent':False,'campaign_resumed':False}
 def clean(value):return str(value).replace(key,'[REDACTED]')[:500]
 data={'model':os.environ[c['model_env']],'messages':[{'role':'user','content':'Reply OK.'}],c['token_parameter']:16,'stream':False}
 request=urllib.request.Request(url,data=json.dumps(data).encode(),headers={'Content-Type':'application/json',c['auth_header']:c['auth_prefix']+key},method='POST')
 class NoRedirect(urllib.request.HTTPRedirectHandler):
  def redirect_request(self,*args,**kwargs):return None
 handlers=[NoRedirect()]
 if parsed.hostname in ('localhost','127.0.0.1','::1'):handlers.append(urllib.request.ProxyHandler({}))
 opener=urllib.request.build_opener(*handlers);start=time.monotonic();result['request_sent']=True
 try:
  with opener.open(request,timeout=45) as response:
   result['http_status']=response.status;body=response.read(65537)
  if len(body)>65536:result['status']='response_too_large'
  else:
   try:reply=_hugsim_expand(json.loads(body.decode('utf-8')))
   except (UnicodeError,ValueError):result['status']='invalid_json_response'
   else:
    result['status']='minimal_request_succeeded' if isinstance(reply,dict) and isinstance(reply.get('choices'),list) and len(reply['choices'])>0 else 'unexpected_response_format'
    usage=reply.get('usage',{}) if isinstance(reply,dict) else {}
    result['usage']={k:v for k,v in usage.items() if k in ('prompt_tokens','completion_tokens','total_tokens') and isinstance(v,int)} if isinstance(usage,dict) else {}
    result['full_patch_request_verified']=False
 except urllib.error.HTTPError as error:
  result.update(status='http_error',http_status=error.code)
  try:
   reply=_hugsim_expand(json.loads(error.read(4096).decode('utf-8','replace')));detail=reply.get('error',{})
   if isinstance(detail,dict):result['provider_error']={k:clean(detail[k]) for k in ('type','code','message') if k in detail}
   elif isinstance(detail,str):result['provider_error']=clean(detail)
  except (ValueError,OSError,AttributeError):pass
 except (urllib.error.URLError,TimeoutError,OSError) as error:
  reason=getattr(error,'reason',error);result.update(status=classify(error),error_type=type(error).__name__,reason_type=type(reason).__name__,reason=clean(reason),errno=getattr(reason,'errno',None))
 result['seconds']=round(time.monotonic()-start,2)
 return result

if __name__=='__main__':
 try:out=run()
 except Exception as error:out={'status':'diagnostic_error','error_type':type(error).__name__}
 print(json.dumps(out,indent=2,ensure_ascii=True))
