"""Replaceable JSON model provider. No SDK dependency or key in persisted records."""
import json,os,urllib.request,urllib.error,urllib.parse
from core import AgentError,Deferred,BudgetExceeded,decode,digest,write,utc
class TransientAPI(Deferred):pass
class HTTPProvider:
 def __init__(self,config):self.config=config
 def readiness(self):
  c=self.config
  return [name for name in (c['base_url_env'],c['model_env'],c['api_key_env']) if not os.environ.get(name)]
 def complete(self,messages,schema,before_request,record):
  missing=self.readiness()
  if missing:raise Deferred('API configuration pending: '+', '.join(missing))
  c=self.config;key=os.environ[c['api_key_env']];url=os.environ[c['base_url_env']].rstrip('/')
  parts=urllib.parse.urlparse(url)
  if parts.username or parts.password or parts.query or parts.fragment:raise AgentError('API base URL must not contain credentials/query/fragment')
  if parts.scheme!='https' and not (parts.scheme=='http' and parts.hostname in ('127.0.0.1','localhost','::1')):raise AgentError('Use HTTPS or loopback HTTP')
  if not url.endswith('/chat/completions'):url+='/chat/completions'
  payload={'model':os.environ[c['model_env']],'messages':messages,c['token_parameter']:c['max_output_tokens'],'stream':False}
  if c['response_format']=='json_schema':payload['response_format']={'type':'json_schema','json_schema':{'name':'patch_decision','strict':True,'schema':schema}}
  elif c['response_format']=='json_object':payload['response_format']={'type':'json_object'}
  elif c['response_format']!='text':raise AgentError('Unknown response format')
  data=json.dumps(payload,ensure_ascii=False).encode();reservation=len(data)+512+c['max_output_tokens'];before_request(reservation)
  headers={'Content-Type':'application/json',c['auth_header']:c['auth_prefix']+key}
  req=urllib.request.Request(url,data=data,headers=headers,method='POST')
  class NoRedirect(urllib.request.HTTPRedirectHandler):
   def redirect_request(self,*args,**kwargs):return None
  handlers=[NoRedirect()]
  if parts.hostname in ('127.0.0.1','localhost','::1'):handlers.append(urllib.request.ProxyHandler({}))
  opener=urllib.request.build_opener(*handlers)
  try:
   with opener.open(req,timeout=c['timeout_seconds']) as response:
    raw=response.read(c['max_response_bytes']+1)
    if len(raw)>c['max_response_bytes']:raise AgentError('Model response exceeds configured byte limit')
   raw_text=raw.decode('utf-8')
   write(record,dict(time=utc(),reserved_tokens=reservation,external_API_called=True,raw_response=raw_text.replace(key,'[REDACTED]')))
   envelope=decode(raw_text);choice=envelope['choices'][0]
   if choice.get('finish_reason')!='stop' or choice['message'].get('refusal'):raise AgentError('Model refused or returned an incomplete response')
   content=choice['message']['content']
   if not isinstance(content,str):raise AgentError('Expected textual JSON model response')
   usage=envelope.get('usage',{});tokens=usage.get('total_tokens')
   if not isinstance(tokens,int) or isinstance(tokens,bool) or tokens<0:tokens=reservation
   meta=dict(time=utc(),request_id=envelope.get('id'),usage_tokens=tokens,reserved_tokens=reservation,model=payload['model'],response_sha256=digest(envelope),external_API_called=True)
   write(record,dict(metadata=meta,content=content.replace(key,'[REDACTED]')))
   return decode(content),meta
  except urllib.error.HTTPError as e:
   # Never persist error bodies or authorization values.
   write(record,dict(time=utc(),http_status=e.code,reserved_tokens=reservation,external_API_called=True))
   if e.code in (401,403):raise Deferred('API authentication rejected; update configured environment variable')
   if e.code==429 or e.code>=500:raise TransientAPI('API temporarily unavailable: HTTP '+str(e.code))
   raise AgentError('API request rejected: HTTP '+str(e.code))
  except (urllib.error.URLError,TimeoutError) as e:
   write(record,dict(time=utc(),error_type=type(e).__name__,reserved_tokens=reservation,outcome_unknown=True,external_API_called=True))
   raise TransientAPI('API transport unavailable; reserved request budget retained')
class FixtureProvider:
 """Test-only; the live CLI never selects it from user configuration."""
 def __init__(self,callback):self.callback=callback;self.calls=0
 def readiness(self):return []
 def complete(self,messages,schema,before_request,record):
  before_request(32);self.calls+=1;value=self.callback(messages,self.calls)
  meta={'usage_tokens':32,'reserved_tokens':32,'external_API_called':False,'mock_model':True}
  write(record,dict(metadata=meta,content=value));return value,meta
